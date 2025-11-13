# main_fast.py
import os
import time
import numpy as np

# Choose the fastest method
USE_PARALLEL = True  # Set to False for vectorized version

if USE_PARALLEL:
    from experiments.run_experiments import run_all_parallel_experiments
else:
    from experiments.run_experiments import run_all_vectorized_experiments

from plots.plot_results import plot_learning_curves, plot_comparison_bar_chart, plot_success_rates

def main_fast():
    """Ultra-fast main function"""
    print("🚀 ULTRA-FAST REINFORCEMENT LEARNING EXPERIMENTS")
    print("=" * 60)
    print(f"Using {'PARALLEL' if USE_PARALLEL else 'VECTORIZED'} execution")
    print(f"CPU cores available: {os.cpu_count()}")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Run experiments
    print("\n🏃‍♂️ Starting parallel experiments...")
    results = run_all_parallel_experiments() if USE_PARALLEL else run_all_vectorized_experiments()
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\n✅ All experiments completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Generate plots
    print("\n📊 Generating plots...")
    plot_learning_curves(results, save_path='plots/learning_curves_fast.png')
    plot_success_rates(results, save_path='plots/success_rates_fast.png')
    plot_comparison_bar_chart(results, save_path='plots/comparison_fast.png')
    
    print("\n🎉 Experiments completed!")
    print(f"📁 Results saved to 'plots' directory")

if __name__ == "__main__":
    # Set numpy for better performance
    np.random.seed(42)  # For reproducibility
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    
    main_fast()