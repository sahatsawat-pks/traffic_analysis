import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from typing import Dict, Iterable, List, Set

COLORS = sv.ColorPalette.DEFAULT

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
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        
    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections]
    ) -> sv.Detections:    
        
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
            
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
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]

class VideoProcessor:
    def __init__(self,
                 source_weights_path: str,
                 source_video_path: str,
                 target_video_path: str = None,
                 confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.7,
    ) -> None:
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])
        
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.detections_manager = DetectionsManager()
        
        
    def process_video(self):
        frame_generator = sv.get_video_frames_generator(self.source_video_path)
        
        for frame in frame_generator:
            processed_frame = self.process_frame(frame)
            cv2.imshow("Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()    
    
    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
        ) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
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
        result = self.model(
            frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold
            )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)
        
        detections_in_zones = []
        
        for zone_in in self.zones_in:
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
        
        detections = self.detections_manager.update(
            detections, detections_in_zones
        )
        
        
        return self.annotate_frame(frame=frame, detections=detections)
        
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        type=str,
        required=True,
        help="Path to the weights file",
    )

    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file",
    )

    parser.add_argument(
        "--target_video_path",
        type=str,
        default=None,
        help="Path to the target video file",
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.7,
        help="IoU threshold",
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        args.source_weights_path,
        args.source_video_path,
        args.target_video_path,
        args.confidence_threshold,
        args.iou_threshold,
    )
    
    processor.process_video()
                