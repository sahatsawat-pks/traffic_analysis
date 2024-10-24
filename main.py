import tkinter as tk
from tkinter import filedialog
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from typing import Dict, Iterable, List, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

ZONE_IN_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]


class DetectionsManager:
    """
    Manages detections across frames, tracking objects and counting zone transitions.

    Attributes:
        tracker_id_to_zone_id (Dict[int, int]): Maps tracker IDs to zone IDs.
        counts (Dict[int, Dict[int, Set[int]]]): Counts of zone transitions.
    """

    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        """
        Updates the detections manager with new detections.

        Args:
            detections_all (sv.Detections): All detections in the frame.
            detections_in_zones (List[sv.Detections]): Detections in each zone.
            detections_out_zones (List[sv.Detections]): Detections out of each zone.

        Returns:
            sv.Detections: Updated detections with zone information.
        """
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)

        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    """Creates polygon zones from polygons and triggering anchors."""
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    """
    Processes a video using a YOLO model and tracks objects across frames.

    Attributes:
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IOU threshold for detections.
        source_video_path (str): Path to the source video.
        target_video_path (str): Path to the target video.
        model (YOLO): YOLO model for detections.
        tracker (sv.ByteTrack): Tracker for object tracking.
        video_info (sv.VideoInfo): Video information.
        zones_in (List[sv.PolygonZone]): Polygon zones for entering objects.
        zones_out (List[sv.PolygonZone]): Polygon zones for exiting objects.
        box_annotator (sv.BoxAnnotator): Box annotator for drawing bounding boxes.
        label_annotator (sv.LabelAnnotator): Label annotator for drawing labels.
        trace_annotator (sv.TraceAnnotator): Trace annotator for drawing object traces.
        detections_manager (DetectionsManager): Detections manager for tracking objects.
    """

    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_video(self):
        """
        Processes the video using the YOLO model and tracker.

        If a target video path is provided, the processed video is written to that path.
        Otherwise, the processed video is displayed in a window.
        """
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for frame_num, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames)):
                        if frame_num % 2 != 0:  # Skip every other frame to speed up processing
                            continue
                        annotated_frame = executor.submit(self.process_frame, frame).result()
                        sink.write_frame(annotated_frame)
                        gc.collect()  # Force garbage collection to manage memory
        else:
            for frame_num, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames)):
                if frame_num % 2 != 0:  # Skip every other frame
                    continue
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """
        Annotates a frame with detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections to annotate.

        Returns:
            np.ndarray: The annotated frame.
        """
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a frame using the YOLO model and tracker."""
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


class Application(tk.Frame):
    """Application for processing videos using a YOLO model and tracker."""

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        """Creates the application widgets."""
        self.source_weights_path = tk.StringVar()
        self.source_video_path = tk.StringVar()
        self.target_video_path = tk.StringVar()
        self.confidence_threshold = tk.DoubleVar(value=0.3)
        self.iou_threshold = tk.DoubleVar(value=0.7)

        tk.Label(self, text="Source Weights Path").grid(row=0, column=0)
        tk.Entry(self, textvariable=self.source_weights_path).grid(row=0, column=1)
        tk.Button(self, text="Browse", command=self.browse_source_weights).grid(row=0, column=2)

        tk.Label(self, text="Source Video Path").grid(row=1, column=0)
        tk.Entry(self, textvariable=self.source_video_path).grid(row=1, column=1)
        tk.Button(self, text="Browse", command=self.browse_source_video).grid(row=1, column=2)

        tk.Label(self, text="Target Video Path").grid(row=2, column=0)
        tk.Entry(self, textvariable=self.target_video_path).grid(row=2, column=1)
        tk.Button(self, text="Browse", command=self.browse_target_video).grid(row=2, column=2)

        tk.Label(self, text="Confidence Threshold").grid(row=3, column=0)
        tk.Entry(self, textvariable=self.confidence_threshold).grid(row=3, column=1)

        tk.Label(self, text="IOU Threshold").grid(row=4, column=0)
        tk.Entry(self, textvariable=self.iou_threshold).grid(row=4, column=1)

        tk.Button(self, text="Start Processing", command=self.start_processing).grid(row=5, column=1)

    def browse_source_weights(self):
        """Browses for the source weights file."""
        file_path = filedialog.askopenfilename()
        self.source_weights_path.set(file_path)

    def browse_source_video(self):
        """Browses for the source video file."""
        file_path = filedialog.askopenfilename()
        self.source_video_path.set(file_path)

    def browse_target_video(self):
        """Browses for the target video file."""
        file_path = filedialog.askopenfilename()
        self.target_video_path.set(file_path)

    def start_processing(self):
        """Starts processing the video."""
        processor = VideoProcessor(
            source_weights_path=self.source_weights_path.get(),
            source_video_path=self.source_video_path.get(),
            target_video_path=self.target_video_path.get(),
            confidence_threshold=self.confidence_threshold.get(),
            iou_threshold=self.iou_threshold.get(),
        )
        processor.process_video()
        self.master.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

# Instructions for processing a video:
# 1. Run the application.
# 2. Browse for the source weights file.
# 3. Browse for the source video file.
# 4. Browse for the target video file (optional).
# 5. Set the confidence threshold and IOU threshold.
# 6. Click the "Start Processing" button.
# 7. The processed video will be written to the target video file (if provided) or displayed in a window.