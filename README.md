# Traffic Analysis with YOLO and ByteTrack

This project implements a traffic flow analysis system using YOLO for object detection and ByteTrack for object tracking.
The system is designed to process video footage, detect vehicles, and track their movements across predefined zones within the video.
It provides detailed annotations and zone-based tracking, making it ideal for traffic monitoring and analysis.



https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900




## Features
- **Object Detection**: Utilizes YOLO for detecting vehicles in the video.
- **Object Tracking**: Integrates ByteTrack for robust tracking of detected objects across frames.
- **Zone-Based Analysis**: Monitors predefined zones and tracks the movement of vehicles in and out of these zones.
- **Customizable Confidence and IoU Thresholds**: Allows fine-tuning of detection and tracking with customizable confidence and IoU thresholds.
- **Real-Time Video Processing**: Processes video in real-time with optional output to a target video file.
- **Visual Annotations**: Annotates frames with bounding boxes, labels, and trace lines, providing a clear visual representation of vehicle movements.

## 💻 install

- clone repository and navigate to example directory

  ```bash
  git clone https://github.com/shrimpstanot/traffic_analysis.git
  cd traffic_analysis
  ```

- setup python environment and activate it [optional]

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- install required dependencies

  ```bash
  pip install -r requirements.txt
  ```

- download `traffic_analysis.pt` and `traffic_analysis.mov` files

  ```bash
  ./setup.sh
  ```

## 🛠️ script arguments

  - `--source_weights_path`: Required. Specifies the path to the YOLO model's weights
    file, which is essential for the object detection process. This file contains the
    data that the model uses to identify objects in the video.

  - `--source_video_path`: Required. The path to the source video file that will be
    analyzed. This is the input video on which traffic flow analysis will be performed.
  - `--target_video_path` (optional): The path to save the output video with
    annotations. If not specified, the processed video will be displayed in real-time
    without being saved.
  - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO
    model to filter detections. Default is `0.3`. This determines how confident the
    model should be to recognize an object in the video.
  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
    for the model. Default is 0.7. This value is used to manage object detection
    accuracy, particularly in distinguishing between different objects.

## ⚙️ run

  ```bash
  python ultralytics_example.py \
  --source_weights_path data/traffic_analysis.pt \
  --source_video_path data/traffic_analysis.mov \
  --confidence_threshold 0.3 \
  --iou_threshold 0.5 \
  --target_video_path data/traffic_analysis_result.mov
```
