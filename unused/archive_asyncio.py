# Unused code (No difference)

import cv2
import asyncio
import threading
import time
import queue
import gc
from ultralytics import YOLO
from supervision.tools.detections import Detections
from supervision.tools.zone import Point, Zone
from supervision.draw.color import Color
from supervision.draw.utils import draw_text
from supervision import Detections as svDetections
from supervision import VideoInfo, VideoSink
from supervision.draw.annotators import BoxAnnotator

# Globals
frame_queue = queue.Queue(maxsize=20)
result_queue = queue.Queue(maxsize=20)
stop_event = threading.Event()
skip_rate = 2

# Load YOLO model
model = YOLO("yolov8n.pt")
annotator = BoxAnnotator()

# Define polygon zone
polygon = [Point(100, 100), Point(300, 100), Point(300, 300), Point(100, 300)]
zone = Zone(polygon=polygon)

def capture_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % skip_rate == 0:
            if not frame_queue.full():
                frame_queue.put((frame_num, frame.copy()))
        frame_num += 1
    cap.release()
    stop_event.set()

async def process_frames():
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame_num, frame = frame_queue.get(timeout=0.5)

            # Async detection
            detections = await asyncio.to_thread(model.predict, frame, verbose=False)

            # Parse detections
            detection_result = detections[0]
            detection_sv = svDetections.from_ultralytics(detection_result)

            # Zone logic
            in_zone = zone.trigger(detection_sv)

            # Annotate
            annotated = annotator.annotate(scene=frame.copy(), detections=detection_sv)
            for i, d in enumerate(detection_sv.xyxy):
                draw_text(annotated, f"{detection_sv.class_id[i]}", (int(d[0]), int(d[1]) - 10))

            if not result_queue.full():
                result_queue.put((frame_num, annotated))
            
            del detections, detection_result, detection_sv
            gc.collect()

        except queue.Empty:
            await asyncio.sleep(0.01)

def display_results():
    while not stop_event.is_set() or not result_queue.empty():
        try:
            frame_num, frame = result_queue.get(timeout=0.5)
            cv2.imshow("YOLO + Supervision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except queue.Empty:
            time.sleep(0.01)
    cv2.destroyAllWindows()

def main(video_path):
    # Start capture thread
    threading.Thread(target=capture_frames, args=(video_path,), daemon=True).start()

    # Start display thread
    threading.Thread(target=display_results, daemon=True).start()

    # Start asyncio loop
    asyncio.run(process_frames())

    print("Finished processing.")

if __name__ == "__main__":
    main("sample.mp4")  # Replace with your actual video file
