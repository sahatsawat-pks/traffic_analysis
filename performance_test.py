import time
import psutil
import tracemalloc
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from main import VideoProcessor
from main_optimize import VideoProcessor_Optimize

def measure_performance(processor_class, source_weights_path, source_video_path, target_video_path, 
                        name="Original", trials=1):
    results = {
        "execution_time": [],
        "memory_increase": [],
        "peak_memory": [],
        "user_cpu_time": [],
        "system_cpu_time": []
    }
    
    for trial in range(trials):
        # Force garbage collection before each trial
        gc.collect()
        
        # Start memory and time tracking
        process = psutil.Process(os.getpid())
        start_cpu_times = process.cpu_times()
        start_time = time.time()
        tracemalloc.start()
        start_mem = process.memory_info().rss

        # Instantiate and run processor
        processor = processor_class(
            source_weights_path=source_weights_path,
            source_video_path=source_video_path,
            target_video_path=target_video_path,
            confidence_threshold=0.4,
            iou_threshold=0.5,
        )
        
        try:
            processor.process_video()
        finally:
            # Clean up resources if the optimized version
            if hasattr(processor, 'cleanup'):
                processor.cleanup()

        # Stop memory and time tracking
        end_time = time.time()
        end_mem = process.memory_info().rss
        end_cpu_times = process.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Store results
        results["execution_time"].append(end_time - start_time)
        results["memory_increase"].append((end_mem - start_mem) / 1024**2)  # MB
        results["peak_memory"].append(peak / 1024**2)  # MB
        results["user_cpu_time"].append(end_cpu_times.user - start_cpu_times.user)
        results["system_cpu_time"].append(end_cpu_times.system - start_cpu_times.system)
        
        print(f"Trial {trial+1}/{trials} completed for {name}")
    
    # Calculate averages
    avg_results = {k: np.mean(v) for k, v in results.items()}
    std_results = {k: np.std(v) for k, v in results.items()}
    
    print(f"\n=== Performance Report [{name}] ===")
    print(f"Total Execution Time     : {avg_results['execution_time']:.2f} ± {std_results['execution_time']:.2f} seconds")
    print(f"Memory Increase          : {avg_results['memory_increase']:.2f} ± {std_results['memory_increase']:.2f} MB")
    print(f"Peak Memory Usage        : {avg_results['peak_memory']:.2f} ± {std_results['peak_memory']:.2f} MB")
    print(f"User CPU Time            : {avg_results['user_cpu_time']:.2f} ± {std_results['user_cpu_time']:.2f} seconds")
    print(f"System CPU Time          : {avg_results['system_cpu_time']:.2f} ± {std_results['system_cpu_time']:.2f} seconds")
    
    return avg_results, std_results

def plot_comparison(original_results, optimized_results):
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot execution time
    axs[0, 0].bar(['Original', 'Optimized'], 
                 [original_results['execution_time'], optimized_results['execution_time']])
    axs[0, 0].set_title('Execution Time (s)')
    axs[0, 0].set_ylabel('Seconds')
    
    # Plot memory usage
    axs[0, 1].bar(['Original', 'Optimized'], 
                 [original_results['peak_memory'], optimized_results['peak_memory']])
    axs[0, 1].set_title('Peak Memory Usage (MB)')
    axs[0, 1].set_ylabel('MB')
    
    # Plot CPU time
    axs[1, 0].bar(['Original User', 'Optimized User', 'Original System', 'Optimized System'], 
                 [original_results['user_cpu_time'], optimized_results['user_cpu_time'],
                  original_results['system_cpu_time'], optimized_results['system_cpu_time']])
    axs[1, 0].set_title('CPU Usage (s)')
    axs[1, 0].set_ylabel('Seconds')
    
    # Plot memory increase
    axs[1, 1].bar(['Original', 'Optimized'], 
                 [original_results['memory_increase'], optimized_results['memory_increase']])
    axs[1, 1].set_title('Memory Usage Increase (MB)')
    axs[1, 1].set_ylabel('MB')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Use the same files for both versions to ensure fair comparison
    YOLO_WEIGHTS = "data/traffic_analysis.pt"
    SOURCE_VIDEO = "data/traffic_analysis.mov"
    OUTPUT_ORIG = "data/traffic_analysis_original.mov"
    OUTPUT_OPT = "data/traffic_analysis_optimized.mov"
    
    # Number of trials for each version
    TRIALS = 3
    
    print("Running original version...")
    orig_avg, orig_std = measure_performance(
        VideoProcessor, 
        YOLO_WEIGHTS, SOURCE_VIDEO, OUTPUT_ORIG, 
        name="Original", 
        trials=TRIALS
    )
    
    print("\nRunning optimized version...")
    opt_avg, opt_std = measure_performance(
        VideoProcessor_Optimize, 
        YOLO_WEIGHTS, SOURCE_VIDEO, OUTPUT_OPT, 
        name="Optimized", 
        trials=TRIALS
    )
    
    # Calculate improvement percentages
    time_improvement = (orig_avg['execution_time'] - opt_avg['execution_time']) / orig_avg['execution_time'] * 100
    memory_improvement = (orig_avg['peak_memory'] - opt_avg['peak_memory']) / orig_avg['peak_memory'] * 100
    
    print("\n=== Improvement Summary ===")
    print(f"Execution Time Reduction  : {time_improvement:.2f}%")
    print(f"Peak Memory Reduction     : {memory_improvement:.2f}%")
    
    # Create visual comparison
    plot_comparison(orig_avg, opt_avg)