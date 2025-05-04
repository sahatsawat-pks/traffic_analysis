# Traffic Analysis with YOLO and ByteTrack

This project implements a traffic flow analysis system using YOLO for object detection and ByteTrack for object tracking.
The system is designed to process video footage, detect vehicles, and track their movements across predefined zones within the video.
It provides detailed annotations and zone-based tracking, making it ideal for traffic monitoring and analysis.

This repository includes both an original command-line implementation (`ultralytics_example.py`) and an optimized version with a graphical user interface (`main_optimize.py`).

[![Watch the video](https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900)](https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900)

## Features

-   **Object Detection**: Utilizes YOLO for detecting vehicles in the video.
-   **Object Tracking**: Integrates ByteTrack for robust tracking of detected objects across frames.
-   **Zone-Based Analysis**: Monitors predefined zones and tracks the movement of vehicles in and out of these zones.
-   **Graphical User Interface (GUI)**: The optimized version provides an easy-to-use Tkinter interface for configuration and execution.
-   **Customizable Settings**: Fine-tune detection and tracking with customizable confidence and IoU thresholds, frame downsampling, frame skipping, and thread count via the GUI or command-line arguments (original version).
-   **Optimized Processing**:
    -   **Multi-threaded Pipeline**: Efficiently reads, processes, annotates, and writes frames in parallel.
    -   **Memory Mapping (MMap)**: Uses memory-mapped file access for optimized video reading, reducing RAM usage for suitable video formats.
    -   **Half-Precision (FP16)**: Option to use FP16 for faster inference on compatible GPUs.
    -   **Resource Management**: Includes memory monitoring and thorough resource cleanup.
-   **Real-Time Video Processing**: Processes video with optional output to a target video file.
-   **Visual Annotations**: Annotates frames with bounding boxes, labels, and trace lines, providing a clear visual representation of vehicle movements and zone counts.
-   **Performance Testing**: Includes a script to compare the original and optimized implementations and measure improvements in speed and resource usage.

## 💻 Install

-   Clone the repository and navigate to the directory:
    ```bash
    git clone [https://github.com/shrimpstanot/traffic_analysis.git](https://github.com/shrimpstanot/traffic_analysis.git)
    cd traffic_analysis
    ```

-   Set up a Python environment and activate it [optional]:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # or ".\venv\Scripts\activate" on Windows
    ```

-   Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `tkinter`, `numpy`, `supervision`, `ultralytics`, `opencv-python`, `psutil`, `tqdm`, `torch`)*

-   Download `traffic_analysis.pt` (YOLO weights) and `traffic_analysis.mov` (sample video) files:
    ```bash
    ./setup.sh
    # or manually download and place them in a 'data/' subdirectory
    ```

##  GUI Usage (Optimized Version)

The optimized version (`main_optimize.py`) provides a graphical interface for easier operation.

1.  **Launch the Application**:
    ```bash
    python main_optimize.py
    ```
2.  **File Selection**:
    -   Use the "Browse" buttons to select the YOLO model weights (`.pt`), the source video file, and the target video path (where the processed video will be saved).
    -   The target path is optional; if left blank, the processed video will be displayed in a window but not saved.
3.  **Configuration**:
    -   Adjust the **Confidence Threshold**, **IOU Threshold**, **Downsample Factor**, and **Skip Frames** using the sliders. The current value is displayed next to each slider.
4.  **Advanced Settings**:
    -   **Use FP16**: Check this box to enable half-precision inference (requires a compatible GPU).
    -   **Annotate Frames**: Check this box to draw bounding boxes, tracks, and counts on the output video. Uncheck for faster processing if annotations are not needed.
    -   **Processing Threads**: Set the number of threads dedicated to processing frames (inference and tracking).
    -   **Queue Size**: Configure the maximum size of the internal queues used in the multi-threaded pipeline.
5.  **Start Processing**:
    -   Click the "Start Processing" button. A progress window will appear showing the status.
    -   You can cancel the process using the "Cancel" button in the progress window.
6.  **Completion**:
    -   Once processing is finished (or canceled), a message box will display the results, including processing time, FPS, and memory usage.

## 🛠️ Script Arguments (Original Version)

The original version (`ultralytics_example.py`) uses command-line arguments for configuration:

-   `--source_weights_path`: **Required**. Specifies the path to the YOLO model's weights file (`.pt`).
-   `--source_video_path`: **Required**. The path to the source video file to be analyzed.
-   `--target_video_path` (optional): The path to save the output video with annotations. If not specified, the processed video might be displayed in real-time (depending on the original script's implementation) or not saved.
-   `--confidence_threshold` (optional): Sets the confidence threshold for YOLO detections. Default is `0.3`.
-   `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold for Non-Max Suppression. Default is `0.7`.

*(Note: The optimized GUI version uses UI elements for these settings instead of command-line arguments.)*

## ⚙️ Run Examples

**Original Version (Command-Line)**:

```bash
python ultralytics_example.py \
--source_weights_path data/traffic_analysis.pt \
--source_video_path data/traffic_analysis.mov \
--confidence_threshold 0.3 \
--iou_threshold 0.5 \
--target_video_path data/traffic_analysis_result.mov
```

**Optimize Version (GUI)**:
```bash
python main_optimize.py \
--source_weights_path data/traffic_analysis.pt \
--source_video_path data/traffic_analysis.mov \
--confidence_threshold 0.3 \
--iou_threshold 0.5 \
--target_video_path data/traffic_analysis_optimized.mov \
--num_threads 8 \
--downsample_factor 0.8
```

**⚙️ Optimization Features**:

The optimized implementation (`main_optimize.py`) includes several improvements:

- **Graphical User Interface**: Easy configuration and execution via Tkinter.
- **Multi-threading Pipeline**: Separate threads for reading, processing, annotating (optional), and writing frames, improving parallelism.
- **Memory Mapping (MMap)**: Attempts to use memory-mapped files for video reading, reducing RAM usage for certain video formats.
- **Memory Management**: Includes monitoring of peak memory usage and specific cleanup routines to release resources (including GPU memory if applicable).
- **Thread-Safe Data Structures**: Uses locks and thread-safe queues for reliable data handling between threads.
- **Frame Downsampling**: Option to resize frames before processing to reduce computational load at the cost of potential accuracy.
- **Frame Skipping**: Option to process only every Nth frame to increase throughput.
- **Half-Precision Inference (FP16)**: Option to use FP16 computation for potentially faster inference on compatible hardware.
- **Progress Monitoring**: Real-time feedback on processing status via the GUI and console output.
- **Resource Cleanup**: Implements explicit cleanup of resources like the model, tracker, video writers, and forces garbage collection.

**Performance Testing**:
The project includes a performance testing script to compare the original implementation with the optimized version:
```bash
python performance_test.py 
```

This will run both implementations on the same video file and generate performance metrics including:

- Execution time
- Frames Per Second (FPS)
- Peak memory usage (RAM and possibly VRAM)
- CPU utilization
- Performance improvement percentages

A visual comparison chart is saved as `performance_comparison.png`.

## References

Roboflow. Supervision [Computer software]. https://github.com/roboflow/supervision
Ultralytics. YOLO [Computer software]. https://github.com/ultralytics/ultralytics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

