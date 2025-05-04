import time
import psutil
import tracemalloc
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from main import VideoProcessor
from main_optimize import VideoProcessor_Optimize

def plot_comparison(original_results, optimized_results, orig_trials=None, opt_trials=None):
    # Bar plots (existing)
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))

    # Bar plot: Execution Time
    axs[0, 0].bar(['Original', 'Optimized'], 
                 [original_results['execution_time'], optimized_results['execution_time']], color=['skyblue', 'lightgreen'])
    axs[0, 0].set_title('Average Execution Time (s)')
    axs[0, 0].set_ylabel('Seconds')
    
    # Bar plot: Peak Memory Usage
    axs[0, 1].bar(['Original', 'Optimized'], 
                 [original_results['peak_memory'], optimized_results['peak_memory']], color=['skyblue', 'lightgreen'])
    axs[0, 1].set_title('Average Peak Memory Usage (MB)')
    axs[0, 1].set_ylabel('MB')
    
    # Bar plot: CPU Time
    axs[1, 0].bar(['Original User', 'Optimized User', 'Original System', 'Optimized System'], 
                 [original_results['user_cpu_time'], optimized_results['user_cpu_time'],
                  original_results['system_cpu_time'], optimized_results['system_cpu_time']],
                 color=['skyblue', 'lightgreen', 'deepskyblue', 'lightseagreen'])
    axs[1, 0].set_title('Average CPU Usage (s)')
    axs[1, 0].set_ylabel('Seconds')
    
    # Bar plot: Memory Increase
    axs[1, 1].bar(['Original', 'Optimized'], 
                 [original_results['memory_increase'], optimized_results['memory_increase']], color=['skyblue', 'lightgreen'])
    axs[1, 1].set_title('Average Memory Usage Increase (MB)')
    axs[1, 1].set_ylabel('MB')
    
    # Line plot: Execution Time per Trial
    if orig_trials and opt_trials:
        axs[2, 0].plot(orig_trials['execution_time'], label='Original', marker='o')
        axs[2, 0].plot(opt_trials['execution_time'], label='Optimized', marker='s')
        axs[2, 0].set_title('Execution Time per Trial')
        axs[2, 0].set_ylabel('Seconds')
        axs[2, 0].set_xlabel('Trial')
        axs[2, 0].legend()
    
        # Boxplot: Memory Increase Spread
        axs[2, 1].boxplot([orig_trials['memory_increase'], opt_trials['memory_increase']], labels=['Original', 'Optimized'])
        axs[2, 1].set_title('Memory Increase Spread per Trial')
        axs[2, 1].set_ylabel('MB')
    
    plt.tight_layout()
    plt.savefig('performance_comparison_extended.png')
    plt.show()

    # Radar chart (optional, separate figure)
    labels = ['Execution Time', 'Peak Memory', 'Memory Increase', 'User CPU', 'System CPU']
    orig_values = [original_results[k] for k in ['execution_time', 'peak_memory', 'memory_increase', 'user_cpu_time', 'system_cpu_time']]
    opt_values = [optimized_results[k] for k in ['execution_time', 'peak_memory', 'memory_increase', 'user_cpu_time', 'system_cpu_time']]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    orig_values += orig_values[:1]
    opt_values += opt_values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, orig_values, 'o-', label='Original')
    ax.plot(angles, opt_values, 'o-', label='Optimized')
    ax.fill(angles, orig_values, alpha=0.25)
    ax.fill(angles, opt_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Radar Chart: Performance Metrics')
    ax.legend(loc='upper right')
    plt.savefig('performance_radar_chart.png')
    plt.show()


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