import tkinter as tk
from tkinter import filedialog, Scale, IntVar, BooleanVar, Checkbutton, messagebox
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from typing import Dict, Iterable, List, Set, Optional, Tuple, Any
from tqdm import tqdm
import gc
import os
import torch
import psutil
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    """

    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}
        self.lock = threading.RLock()  # Reentrant lock for thread safety

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        """
        Updates the detections manager with new detections.
        Thread-safe implementation.
        """
        with self.lock:
            # Process zone entry
            for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
                if detections_in_zone.tracker_id is not None:
                    for tracker_id in detections_in_zone.tracker_id:
                        self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

            # Process zone exit and counting
            for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
                if detections_out_zone.tracker_id is not None:
                    for tracker_id in detections_out_zone.tracker_id:
                        if tracker_id in self.tracker_id_to_zone_id:
                            zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                            self.counts.setdefault(zone_out_id, {})
                            self.counts[zone_out_id].setdefault(zone_in_id, set())
                            self.counts[zone_out_id][zone_in_id].add(tracker_id)

            # Update class IDs based on tracking
            if len(detections_all) > 0 and detections_all.tracker_id is not None:
                detections_all.class_id = np.array([
                    self.tracker_id_to_zone_id.get(tracker_id, -1) 
                    for tracker_id in detections_all.tracker_id
                ])
            else:
                detections_all.class_id = np.array([], dtype=int)

            return detections_all[detections_all.class_id != -1]
    
    def get_counts(self) -> Dict[int, Dict[int, int]]:
        """Return the counts in a thread-safe way."""
        with self.lock:
            result = {}
            for zone_out_id, zones in self.counts.items():
                result[zone_out_id] = {}
                for zone_in_id, trackers in zones.items():
                    result[zone_out_id][zone_in_id] = len(trackers)
            return result 


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


def scale_polygon(polygon: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale a polygon by the given factor."""
    centroid = np.mean(polygon, axis=0)
    return np.array([
        centroid + (point - centroid) * scale_factor 
        for point in polygon
    ], dtype=np.int32)


class MemoryMonitor:
    """
    Monitors memory usage during video processing.
    """
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        self.lock = threading.Lock()
        
    def update(self):
        current = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        with self.lock:
            self.peak_memory = max(self.peak_memory, current)
        self.peak_memory = max(self.peak_memory, current)
        return current
        
    def get_peak(self):
        with self.lock:
            return self.peak_memory
        
    def get_increase(self):
        with self.lock:
            return self.peak_memory - self.start_memory

class FrameProcessor:
    """
    Processes a single frame with YOLO detection and tracking.
    This class is designed to be used by multiple threads.
    """
    def __init__(
        self,
        model: YOLO,
        tracker: sv.ByteTrack,
        zones_in: List[sv.PolygonZone],
        zones_out: List[sv.PolygonZone],
        detections_manager: DetectionsManager,
        conf_threshold: float,
        iou_threshold: float,
        use_fp16: bool,
        downsample_factor: float = 1.0,
    ):
        self.model = model
        self.tracker = tracker
        self.zones_in = zones_in
        self.zones_out = zones_out
        self.detections_manager = detections_manager
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_fp16 = use_fp16
        self.downsample_factor = downsample_factor
        
        # Initialize thread-local storage
        self.thread_local = threading.local()
        
        # Model lock for thread safety if model is not thread-safe
        self.model_lock = threading.Lock()
        
    def resize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize frame based on downsample factor."""
        if self.downsample_factor == 1.0:
            return frame, frame.shape[:2]
            
        h, w = frame.shape[:2]
        new_h, new_w = int(h * self.downsample_factor), int(w * self.downsample_factor)
        return cv2.resize(frame, (new_w, new_h)), (h, w)
        
    def process(self, frame: np.ndarray) -> sv.Detections:
        """Process a single frame and return detections."""
        # Resize frame if needed
        resized_frame, original_dims = self.resize_frame(frame)
        
        # Use thread-local storage for per-thread caching
        
        # CPU Optimization

        if not hasattr(self.thread_local, 'device_set'):
            # Set device for this thread
            self.thread_local.device_set = True
            if torch.cuda.is_available():
                # If multiple GPUs are available, we could distribute across them
                device_id = threading.current_thread().ident % torch.cuda.device_count() if torch.cuda.device_count() > 1 else 0
                self.thread_local.device = torch.device(f'cuda:{device_id}')
            else:
                self.thread_local.device = torch.device('cpu')
        
        # Run inference with model lock to ensure thread safety
        with self.model_lock:
            with torch.inference_mode():
                # Run model inference
                results = self.model(
                    resized_frame, 
                    verbose=False, 
                    conf=self.conf_threshold, 
                    iou=self.iou_threshold,
                    half=self.use_fp16
                )[0]
                
                # Convert results to detections
                detections = sv.Detections.from_ultralytics(results)
                
                if len(detections) > 0:
                    detections.class_id = np.zeros(len(detections))
                    
                    # Update with tracker (tracker should be thread-safe)
                    detections = self.tracker.update_with_detections(detections)
                    
                    # Process zones
                    detections_in_zones = []
                    detections_out_zones = []
                    
                    for zone_in, zone_out in zip(self.zones_in, self.zones_out):
                        detections_in_zone = detections[zone_in.trigger(detections=detections)]
                        detections_in_zones.append(detections_in_zone)
                        detections_out_zone = detections[zone_out.trigger(detections=detections)]
                        detections_out_zones.append(detections_out_zone)
                    
                    # Update detections manager
                    detections = self.detections_manager.update(
                        detections, detections_in_zones, detections_out_zones
                    )

                    # Clear CUDA cache if needed
                    if self.frame_count % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return detections


class FrameAnnotator:
    """
    Handles the annotation of frames with detections.
    """
    def __init__(
        self,
        zones_in: List[sv.PolygonZone],
        zones_out: List[sv.PolygonZone],
        detections_manager: DetectionsManager,
    ):
        self.zones_in = zones_in
        self.zones_out = zones_out
        self.detections_manager = detections_manager
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=50, thickness=2
        )
        
    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Annotate a frame with detections."""
        annotated_frame = frame.copy()
        
        # Draw zones
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )
        
        # Only annotate if there are detections
        if len(detections) > 0 and detections.tracker_id is not None:
            # Generate minimal labels
            labels = [f"#{tid}" for tid in detections.tracker_id]
            
            # Apply annotations
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, labels
            )
        
        # Draw counts
        counts = self.detections_manager.get_counts()
        for zone_out_id, zone_out in enumerate(self.zones_out):
            if zone_out_id in counts:
                zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
                zone_counts = counts[zone_out_id]
                
                for i, zone_in_id in enumerate(zone_counts):
                    count = zone_counts[zone_in_id]
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )
        
        return annotated_frame

class VideoProcessor_Optimize:
    """
    Processes a video using a YOLO model and tracks objects across frames.
    Multi-threaded / Memory implementation with I/O optimization.
    """

    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        confidence_threshold: float,
        iou_threshold: float,
        downsample_factor: float = 1.0,
        skip_frames: int = 0,
        use_fp16: bool = False,
        annotate_frames: bool = True,
        num_threads: int = 4,
        queue_size: int = 32,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.downsample_factor = downsample_factor
        self.skip_frames = skip_frames
        self.use_fp16 = use_fp16
        self.annotate_frames = annotate_frames
        self.num_threads = num_threads
        self.queue_size = queue_size
        self.memory_monitor = MemoryMonitor()

        # Initialize model with optimized settings
        self.model = self.initialize_model(source_weights_path)
        self.tracker = sv.ByteTrack()

        # Get video info
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        
        # Initialize zones with proper scaling if needed
        if self.downsample_factor != 1.0:
            scaled_zones_in = [scale_polygon(p, downsample_factor) for p in ZONE_IN_POLYGONS]
            scaled_zones_out = [scale_polygon(p, downsample_factor) for p in ZONE_OUT_POLYGONS]
            self.zones_in = initiate_polygon_zones(scaled_zones_in, [sv.Position.CENTER])
            self.zones_out = initiate_polygon_zones(scaled_zones_out, [sv.Position.CENTER])
        else:
            self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
            self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])
        
        # Initialize the detection manager
        self.detections_manager = DetectionsManager()
        
        # Initialize the frame processor
        self.frame_processor = FrameProcessor(
            model=self.model,
            tracker=self.tracker,
            zones_in=self.zones_in,
            zones_out=self.zones_out,
            detections_manager=self.detections_manager,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            use_fp16=self.use_fp16,
            downsample_factor=self.downsample_factor,
        )

        # Initialize annotators only if needed
        if self.annotate_frames:
            self.frame_annotator = FrameAnnotator(
                zones_in=self.zones_in,
                zones_out=self.zones_out,
                detections_manager=self.detections_manager,
            )
        
        # Initialize thread-safe queues for pipeline
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        # Initialize threading control
        self.stop_event = threading.Event()
        self.threads = []
        
        # Statistics
        self.start_time = None
        self.frame_count = 0
        self.processed_count = 0
        self.stats_lock = threading.Lock()

    def initialize_model(self, weights_path: str) -> YOLO:
        """Initialize the YOLO model with optimized settings."""
        model = YOLO(weights_path)
        
        # Optimize model settings
        if torch.cuda.is_available():
            model.to('cuda')
            if self.use_fp16 and torch.cuda.is_available():
                model.model.half()  # Use FP16 precision
        
        return model
    
    def frame_reader(self):
        """Thread function for reading frames from video."""
        try:
            cap = cv2.VideoCapture(self.source_video_path)
            frame_idx = 0
            
            while not self.stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    break
                
                # Update frame counter
                with self.stats_lock:
                    self.frame_count += 1
                
                # Skip frames if needed
                if self.skip_frames > 0 and frame_idx % (self.skip_frames + 1) != 0:
                    frame_idx += 1
                    continue
                
                # Put frame in queue
                try:
                    self.frame_queue.put((frame_idx, frame), timeout=1.0)
                    frame_idx += 1
                except queue.Full:
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
            # Signal end of frames
            self.frame_queue.put((None, None))
            
        except Exception as e:
            logger.error(f"Error in frame reader: {e}")
            self.stop_event.set()
        finally:
            cap.release()

    def resize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize frame based on downsample factor."""
        if self.downsample_factor == 1.0:
            return frame, frame.shape[:2]
            
        h, w = frame.shape[:2]
        new_h, new_w = int(h * self.downsample_factor), int(w * self.downsample_factor)
        return cv2.resize(frame, (new_w, new_h)), (h, w)
    
    def process_frames(self):
        """Thread function for processing frames."""
        try:
            while not self.stop_event.is_set():
                # Get frame from queue
                try:
                    frame_idx, frame = self.frame_queue.get(timeout=1.0)
                    if frame_idx is None:  # End signal
                        self.frame_queue.put((None, None))  # Forward the signal
                        break
                    
                    # Process frame
                    detections = self.frame_processor.process(frame)
                    
                    # Put result in queue
                    self.result_queue.put((frame_idx, frame, detections), timeout=1.0)
                    
                    # Mark task as done
                    self.frame_queue.task_done()
                    
                    # Update processed count
                    with self.stats_lock:
                        self.processed_count += 1
                    
                    # Monitor memory
                    self.memory_monitor.update()
                    
                except queue.Empty:
                    continue
                
        except Exception as e:
            logger.error(f"Error in frame processor: {e}")
            self.stop_event.set()
        finally:
            # Signal end of processing
            self.result_queue.put((None, None, None))

    def annotate_frames(self):
        """Thread function for annotating frames."""
        try:
            while not self.stop_event.is_set():
                # Get result from queue
                try:
                    frame_idx, frame, detections = self.result_queue.get(timeout=1.0)
                    if frame_idx is None:  # End signal
                        self.result_queue.put((None, None, None))  # Forward the signal
                        break
                    
                    # Annotate frame if needed
                    if self.annotate_frames:
                        annotated_frame = self.frame_annotator.annotate(frame, detections)
                    else:
                        annotated_frame = frame
                    
                    # Put annotated frame in output queue
                    self.output_queue.put((frame_idx, annotated_frame), timeout=1.0)
                    
                    # Mark task as done
                    self.result_queue.task_done()
                    
                except queue.Empty:
                    continue
                
        except Exception as e:
            logger.error(f"Error in frame annotator: {e}")
            self.stop_event.set()
        finally:
            # Signal end of annotation
            self.output_queue.put((None, None))

    def writer(self):
        """Thread function for writing frames to video."""
        try:
            # Setup video writer
            sink = None
            if self.target_video_path:
                # Adjust output resolution if needed
                if self.downsample_factor != 1.0 and self.annotate_frames:
                    # Output will be at the processed resolution
                    h, w = int(self.video_info.height * self.downsample_factor), int(self.video_info.width * self.downsample_factor)
                    fps = self.video_info.fps
                    try:
                        sink = cv2.VideoWriter(self.target_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    except Exception as e:
                        logger.error(f"Error creating video sink: {e}")
                        sink = None
                else:
                    try:
                        sink = cv2.VideoWriter(
                            self.target_video_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            self.video_info.fps, 
                            (self.video_info.width, self.video_info.height)
                        )
                    except Exception as e:
                        logger.error(f"Error creating video sink: {e}")
                        sink = None
            
            # Process frames from output queue
            frames_written = 0
            buffer = {}  # Buffer to ensure frames are written in order
            next_frame_idx = 0
            
            while not self.stop_event.is_set():
                try:
                    frame_idx, frame = self.output_queue.get(timeout=1.0)
                    if frame_idx is None:  # End signal
                        break
                    
                    # Store frame in buffer
                    buffer[frame_idx] = frame
                    
                    # Write frames in order
                    while next_frame_idx in buffer:
                        next_frame = buffer.pop(next_frame_idx)
                        if sink is not None:
                            sink.write(next_frame)
                        frames_written += 1
                        next_frame_idx += 1
                    
                    # Mark task as done
                    self.output_queue.task_done()
                    
                    # Display frame if needed
                    if sink is None and self.annotate_frames:
                        cv2.imshow("Processed Video", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.stop_event.set()
                            break
                    
                except queue.Empty:
                    continue
                
        except Exception as e:
            logger.error(f"Error in frame writer: {e}")
            self.stop_event.set()
        finally:
            # Clean up
            if sink is not None:
                sink.release()
            if sink is None and self.annotate_frames:
                cv2.destroyAllWindows()

    def process_video(self):
        """Process video using multi-threading pipeline."""
        self.start_time = time.time()
        self.stop_event.clear()
        
        # Muitithreading (Process Optimization)

        # Start reader thread
        reader_thread = threading.Thread(target=self.frame_reader)
        reader_thread.daemon = True
        reader_thread.start()
        self.threads.append(reader_thread)

        # Start processor threads
        for _ in range(self.num_threads):
            processor_thread = threading.Thread(target=self.process_frames)
            processor_thread.daemon = True
            processor_thread.start()
            self.threads.append(processor_thread)
        
        # Start annotator thread
        annotator_thread = threading.Thread(target=self.annotate_frames)
        annotator_thread.daemon = True
        annotator_thread.start()
        self.threads.append(annotator_thread)
        
        # Start writer thread
        writer_thread = threading.Thread(target=self.writer)
        writer_thread.daemon = True
        writer_thread.start()
        self.threads.append(writer_thread)
        
        # Create a progress bar
        cap = cv2.VideoCapture(self.source_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        with tqdm(total=total_frames) as pbar:
            last_count = 0
            while any(thread.is_alive() for thread in self.threads):
                if self.stop_event.is_set():
                    break
                
                # Update progress bar
                with self.stats_lock:
                    new_count = self.frame_count
                    if new_count > last_count:
                        pbar.update(new_count - last_count)
                        last_count = new_count
                
                time.sleep(0.1)
        
        # Wait for all threads to finish
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Print statistics
        duration = time.time() - self.start_time
        peak_memory = self.memory_monitor.get_peak()
        memory_increase = self.memory_monitor.get_increase()

        print(f"\nProcessing completed:")
        print(f"- Total frames: {self.frame_count}")
        print(f"- Processed frames: {self.processed_count}")
        print(f"- Processing time: {duration:.2f} seconds")
        print(f"- FPS: {self.processed_count / duration:.2f}")
        print(f"- Peak memory usage: {peak_memory:.2f} MB")
        print(f"- Memory increase: {memory_increase:.2f} MB")
        print(f"- Threads used: {self.num_threads}")
        
        # Clean up
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        # Memory / CPU Optimization
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Move model to CPU
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
                self.result_queue.task_done()
            except queue.Empty:
                break
                
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
                self.output_queue.task_done()
            except queue.Empty:
                break
        
        # Delete large objects
        self.model = None
        self.tracker = None
        self.frame_processor = None
        if hasattr(self, 'frame_annotator'):
            self.frame_annotator = None
        
        # Force garbage collection
        gc.collect()


class Application(tk.Frame):
    """Application for processing videos using a YOLO model and tracker."""

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Video Processing Tool")
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.current_process = None
        self.progress_window = None
        self.create_widgets()
        
    def create_widgets(self):
        """Creates the application widgets."""
        # Input variables
        self.source_weights_path = tk.StringVar()
        self.source_video_path = tk.StringVar()
        self.target_video_path = tk.StringVar()
        self.confidence_threshold = tk.DoubleVar(value=0.3)
        self.iou_threshold = tk.DoubleVar(value=0.7)
        self.downsample_factor = tk.DoubleVar(value=1.0)
        self.skip_frames = tk.IntVar(value=0)
        self.use_fp16 = BooleanVar(value=False)
        self.annotate_frames = BooleanVar(value=True)
        self.num_threads = tk.IntVar(value=4)
        self.queue_size = tk.IntVar(value=32)
        
        # Create main frames
        file_frame = tk.LabelFrame(self, text="File Selection", padx=5, pady=5)
        file_frame.pack(fill=tk.X, expand=False, pady=5)
        
        config_frame = tk.LabelFrame(self, text="Configuration", padx=5, pady=5)
        config_frame.pack(fill=tk.X, expand=False, pady=5)
        
        advanced_frame = tk.LabelFrame(self, text="Advanced Settings", padx=5, pady=5)
        advanced_frame.pack(fill=tk.X, expand=False, pady=5)
        
        action_frame = tk.Frame(self, padx=5, pady=5)
        action_frame.pack(fill=tk.X, expand=False, pady=10)
        
        # File selection
        row = 0
        tk.Label(file_frame, text="Source Weights Path").grid(row=row, column=0, sticky="w", pady=2)
        tk.Entry(file_frame, textvariable=self.source_weights_path, width=40).grid(row=row, column=1, sticky="ew", padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_source_weights).grid(row=row, column=2, padx=5)
        
        row += 1
        tk.Label(file_frame, text="Source Video Path").grid(row=row, column=0, sticky="w", pady=2)
        tk.Entry(file_frame, textvariable=self.source_video_path, width=40).grid(row=row, column=1, sticky="ew", padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_source_video).grid(row=row, column=2, padx=5)
        
        row += 1
        tk.Label(file_frame, text="Target Video Path").grid(row=row, column=0, sticky="w", pady=2)
        tk.Entry(file_frame, textvariable=self.target_video_path, width=40).grid(row=row, column=1, sticky="ew", padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_target_video).grid(row=row, column=2, padx=5)
        
        # Configuration options
        row = 0
        tk.Label(config_frame, text="Confidence Threshold").grid(row=row, column=0, sticky="w", pady=2)
        conf_scale = tk.Scale(config_frame, variable=self.confidence_threshold, from_=0.0, to=1.0, 
                             resolution=0.05, orient=tk.HORIZONTAL)
        conf_scale.grid(row=row, column=1, sticky="ew")
        tk.Label(config_frame, text=f"{self.confidence_threshold.get():.2f}").grid(row=row, column=2, padx=5, sticky="w")
        conf_scale.config(command=lambda val: self.update_label(conf_scale, row, 2, f"{float(val):.2f}"))
        
        row += 1
        tk.Label(config_frame, text="IOU Threshold").grid(row=row, column=0, sticky="w", pady=2)
        iou_scale = tk.Scale(config_frame, variable=self.iou_threshold, from_=0.0, to=1.0, 
                            resolution=0.05, orient=tk.HORIZONTAL)
        iou_scale.grid(row=row, column=1, sticky="ew")
        tk.Label(config_frame, text=f"{self.iou_threshold.get():.2f}").grid(row=row, column=2, padx=5, sticky="w")
        iou_scale.config(command=lambda val: self.update_label(iou_scale, row, 2, f"{float(val):.2f}"))
        
        row += 1
        tk.Label(config_frame, text="Downsample Factor").grid(row=row, column=0, sticky="w", pady=2)
        down_scale = tk.Scale(config_frame, variable=self.downsample_factor, from_=0.1, to=1.0, 
                             resolution=0.1, orient=tk.HORIZONTAL)
        down_scale.grid(row=row, column=1, sticky="ew")
        tk.Label(config_frame, text=f"{self.downsample_factor.get():.1f}").grid(row=row, column=2, padx=5, sticky="w")
        down_scale.config(command=lambda val: self.update_label(down_scale, row, 2, f"{float(val):.1f}"))
        
        row += 1
        tk.Label(config_frame, text="Skip Frames").grid(row=row, column=0, sticky="w", pady=2)
        skip_scale = tk.Scale(config_frame, variable=self.skip_frames, from_=0, to=10, 
                             resolution=1, orient=tk.HORIZONTAL)
        skip_scale.grid(row=row, column=1, sticky="ew")
        tk.Label(config_frame, text=f"{self.skip_frames.get()}").grid(row=row, column=2, padx=5, sticky="w")
        skip_scale.config(command=lambda val: self.update_label(skip_scale, row, 2, f"{int(float(val))}"))
        
        # Advanced settings
        row = 0
        options_frame = tk.Frame(advanced_frame)
        options_frame.pack(fill=tk.X, expand=True, pady=5)
        
        tk.Checkbutton(options_frame, text="Use FP16 (half precision)", variable=self.use_fp16).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(options_frame, text="Annotate Frames", variable=self.annotate_frames).pack(side=tk.LEFT, padx=5)
        
        threading_frame = tk.Frame(advanced_frame)
        threading_frame.pack(fill=tk.X, expand=True, pady=5)
        
        tk.Label(threading_frame, text="Processing Threads:").pack(side=tk.LEFT, padx=5)
        tk.Spinbox(threading_frame, from_=1, to=16, textvariable=self.num_threads, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(threading_frame, text="Queue Size:").pack(side=tk.LEFT, padx=5)
        tk.Spinbox(threading_frame, from_=8, to=64, textvariable=self.queue_size, width=5).pack(side=tk.LEFT, padx=5)
        
        # Info section
        info_text = "This tool processes video using YOLO object detection with tracking.\n" + \
                   "Higher confidence threshold = fewer detections but more accurate.\n" + \
                   "Lower downsample factor = faster processing but less accurate."
        info_label = tk.Label(advanced_frame, text=info_text, justify=tk.LEFT, 
                             fg="gray", font=("Arial", 8))
        info_label.pack(fill=tk.X, expand=True, pady=5)
        
        # Process button
        self.process_btn = tk.Button(action_frame, text="Start Processing", command=self.start_processing, 
                                    bg="#4CAF50", fg="white", height=2)
        self.process_btn.pack(fill=tk.X, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure grid weights
        file_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(1, weight=1)

    def update_label(self, scale_widget, row, col, text):
        """Updates the label next to a scale widget."""
        scale_widget.master.grid_slaves(row=row, column=col)[0].config(text=text)

    def browse_source_weights(self):
        """Browses for the source weights file."""
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")])
        if file_path:
            self.source_weights_path.set(file_path)
            self.status_var.set(f"Selected weights: {os.path.basename(file_path)}")

    def browse_source_video(self):
        """Browses for the source video file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")])
        if file_path:
            self.source_video_path.set(file_path)
            self.status_var.set(f"Selected source video: {os.path.basename(file_path)}")
            
            # Auto-fill target path with _processed suffix
            if not self.target_video_path.get():
                base, ext = os.path.splitext(file_path)
                self.target_video_path.set(f"{base}_processed{ext}")

    def browse_target_video(self):
        """Browses for the target video file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Files", "*.mp4"), ("AVI Files", "*.avi"), ("All Files", "*.*")])
        if file_path:
            self.target_video_path.set(file_path)
            self.status_var.set(f"Selected target path: {os.path.basename(file_path)}")

    def start_processing(self):
        """Starts processing the video."""
        # Validate inputs
        if not self.source_weights_path.get():
            tk.messagebox.showwarning("Missing Input", "Please provide YOLO model weights path.")
            return
            
        if not self.source_video_path.get():
            tk.messagebox.showwarning("Missing Input", "Please provide source video path.")
            return
            
        if not os.path.exists(self.source_weights_path.get()):
            tk.messagebox.showerror("File Not Found", f"Model weights file not found: {self.source_weights_path.get()}")
            return
            
        if not os.path.exists(self.source_video_path.get()):
            tk.messagebox.showerror("File Not Found", f"Source video file not found: {self.source_video_path.get()}")
            return
            
        # Check if output directory exists
        if self.target_video_path.get():
            output_dir = os.path.dirname(self.target_video_path.get())
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except Exception as e:
                    tk.messagebox.showerror("Error", f"Cannot create output directory: {str(e)}")
                    return
        
        # Create progress window
        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title("Processing Video")
        self.progress_window.geometry("400x150")
        self.progress_window.resizable(False, False)
        self.progress_window.transient(self.master)
        self.progress_window.grab_set()
        
        # Add widgets to progress window
        frame = tk.Frame(self.progress_window, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame, text="Processing video. Please wait...", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Label(frame, text="This may take several minutes depending on the video length and settings.").pack()
        
        # Progress info
        info_frame = tk.Frame(frame)
        info_frame.pack(fill=tk.X, expand=True, pady=5)
        
        self.progress_status = tk.StringVar(value="Initializing...")
        tk.Label(info_frame, textvariable=self.progress_status).pack(side=tk.LEFT)
        
        self.cancel_button = tk.Button(frame, text="Cancel", command=self.cancel_processing)
        self.cancel_button.pack(pady=10)
        
        # Disable main window controls
        self.process_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing video...")
        
        # Start processing in a separate thread to keep UI responsive
        self.master.after(100, self._run_processing)
        
    def cancel_processing(self):
        """Cancels the current processing operation."""
        if self.current_process and hasattr(self.current_process, 'stop_event'):
            self.current_process.stop_event.set()
            self.progress_status.set("Canceling... Please wait.")
            self.cancel_button.config(state=tk.DISABLED)
    
    def _run_processing(self):
        """Run the processing operation in a background thread."""
        try:
            # Create processor
            self.current_process = VideoProcessor_Optimize(
                source_weights_path=self.source_weights_path.get(),
                source_video_path=self.source_video_path.get(),
                target_video_path=self.target_video_path.get(),
                confidence_threshold=self.confidence_threshold.get(),
                iou_threshold=self.iou_threshold.get(),
                downsample_factor=self.downsample_factor.get(),
                skip_frames=self.skip_frames.get(),
                use_fp16=self.use_fp16.get(),
                annotate_frames=self.annotate_frames.get(),
                num_threads=self.num_threads.get(),
                queue_size=self.queue_size.get(),
            )
            
            # Start monitoring thread for updating progress
            self.monitor_thread = threading.Thread(target=self._monitor_progress)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # Start processing in a separate thread
            self.process_thread = threading.Thread(target=self._process_video_thread)
            self.process_thread.daemon = True
            self.process_thread.start()
            
        except Exception as e:
            self._processing_error(f"Failed to start processing: {str(e)}")
    
    def _process_video_thread(self):
        """Thread function to run video processing."""
        try:
            self.current_process.process_video()
            
            # Update UI on main thread
            if not self.current_process.stop_event.is_set():
                self.master.after(0, self._processing_complete)
            else:
                self.master.after(0, self._processing_canceled)
                
        except Exception as e:
            self.master.after(0, lambda: self._processing_error(str(e)))
    
    def _monitor_progress(self):
        """Monitor progress and update UI."""
        while self.current_process and not self.current_process.stop_event.is_set():
            # Access frame counts safely
            with self.current_process.stats_lock:
                total = self.current_process.frame_count
                processed = self.current_process.processed_count
            
            if total > 0:
                status = f"Processing: {processed}/{total} frames"
                self.master.after(0, lambda s=status: self.progress_status.set(s))
            
            time.sleep(0.5)
    
    def _processing_complete(self):
        """Called when processing completes successfully."""
        if self.current_process:
            duration = time.time() - self.current_process.start_time
            peak_memory = self.current_process.memory_monitor.get_peak()
            
            # Close progress window
            if self.progress_window:
                self.progress_window.destroy()
                self.progress_window = None
            
            # Show completion message
            tk.messagebox.showinfo(
                "Processing Complete", 
                f"Video processing completed successfully!\n\n"
                f"Processed {self.current_process.processed_count} frames in "
                f"{duration:.2f} seconds ({self.current_process.processed_count / duration:.2f} FPS).\n\n"
                f"Peak memory usage: {peak_memory:.2f} MB"
            )
            
            # Clean up
            self._cleanup_processing()
            
            # Update status
            self.status_var.set("Processing complete")
    
    def _processing_canceled(self):
        """Called when processing is canceled."""
        # Close progress window
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
        
        tk.messagebox.showinfo("Canceled", "Video processing was canceled.")
        
        # Clean up
        self._cleanup_processing()
        
        # Update status
        self.status_var.set("Processing canceled")
    
    def _processing_error(self, error_msg):
        """Called when an error occurs during processing."""
        # Close progress window
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
        
        tk.messagebox.showerror("Error", f"An error occurred during processing:\n\n{error_msg}")
        
        # Clean up
        self._cleanup_processing()
        
        # Update status
        self.status_var.set(f"Error: {error_msg[:30]}...")
    
    def _cleanup_processing(self):
        """Clean up after processing completes, errors, or is canceled."""
        # Clean up processor
        if self.current_process:
            self.current_process.cleanup()
            self.current_process = None
        
        # Re-enable controls
        self.process_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()