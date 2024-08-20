import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

COLORS = sv.ColorPalette.DEFAULT

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
        
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        
        
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
        return annotated_frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(
            frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold
            )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)
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
                