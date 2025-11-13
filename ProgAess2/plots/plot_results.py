# plots/plot_results.py - COMPLETELY FIXED VERSION
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def debug_array_structure(data, name):
    """Helper function to debug array structure"""
    print(f"DEBUG {name}:")
    print(f"  Type: {type(data)}")
    if hasattr(data, 'shape'):
        print(f"  Shape: {data.shape}")
    elif hasattr(data, '__len__'):
        print(f"  Length: {len(data)}")
        if len(data) > 0:
            print(f"  First element type: {type(data[0])}")
            if hasattr(data[0], 'shape'):
                print(f"  First element shape: {data[0].shape}")

def plot_learning_curves(results_dict, save_path=None):
    """Plot learning curves for all experiments - ROBUST VERSION"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    plot_count = 0
    
    for idx, (exp_name, results) in enumerate(results_dict.items()):
        if idx >= 4:  # Only plot first 4 experiments
            break
            
        try:
            # Extract rewards with proper conversion to 1D arrays
            sarsa_rewards = results['training']['sarsa']['rewards']
            qlearn_rewards = results['training']['qlearn']['rewards']
            
            # Convert to numpy arrays and ensure they are 1D
            sarsa_rewards = np.array(sarsa_rewards).flatten()
            qlearn_rewards = np.array(qlearn_rewards).flatten()
            
            print(f"Plotting {exp_name}: SARSA length={len(sarsa_rewards)}, Q-learn length={len(qlearn_rewards)}")
            
            # Smooth the rewards if we have enough data
            window = min(50, len(sarsa_rewards) // 10)
            if window < 1:
                window = 1
            
            if len(sarsa_rewards) >= window:
                sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
                qlearn_smooth = np.convolve(qlearn_rewards, np.ones(window)/window, mode='valid')
                
                x_axis = np.arange(len(sarsa_smooth))
                axes[idx].plot(x_axis, sarsa_smooth, label='SARSA', alpha=0.8, linewidth=2)
                axes[idx].plot(x_axis, qlearn_smooth, label='Q-Learning', alpha=0.8, linewidth=2)
            else:
                # Plot raw data if not enough for smoothing
                axes[idx].plot(sarsa_rewards, label='SARSA', alpha=0.6, linewidth=1)
                axes[idx].plot(qlearn_rewards, label='Q-Learning', alpha=0.6, linewidth=1)
            
            axes[idx].set_title(f'{exp_name.replace("_", " ").title()}')
            axes[idx].set_xlabel('Episode')
            axes[idx].set_ylabel('Reward')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            plot_count += 1
            
        except Exception as e:
            print(f"Error plotting {exp_name}: {e}")
            axes[idx].set_title(f'{exp_name} - Plotting Error')
            axes[idx].text(0.5, 0.5, f'Error: {str(e)}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    # Hide unused subplots
    for idx in range(plot_count, 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.show()

def plot_success_rates(results_dict, save_path=None):
    """Plot success rates over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    plot_count = 0
    
    for idx, (exp_name, results) in enumerate(results_dict.items()):
        if idx >= 4:
            break
            
        try:
            sarsa_success = np.array(results['training']['sarsa']['success']).flatten()
            qlearn_success = np.array(results['training']['qlearn']['success']).flatten()
            
            # Calculate running success rate
            window = min(100, len(sarsa_success) // 10)
            if window < 1:
                window = 1
                
            sarsa_success_rate = []
            qlearn_success_rate = []
            
            for i in range(len(sarsa_success)):
                start_idx = max(0, i - window + 1)
                sarsa_success_rate.append(np.mean(sarsa_success[start_idx:i+1]))
                qlearn_success_rate.append(np.mean(qlearn_success[start_idx:i+1]))
            
            axes[idx].plot(sarsa_success_rate, label='SARSA', alpha=0.8)
            axes[idx].plot(qlearn_success_rate, label='Q-Learning', alpha=0.8)
            axes[idx].set_title(f'{exp_name.replace("_", " ").title()} - Success Rate')
            axes[idx].set_xlabel('Episode')
            axes[idx].set_ylabel('Success Rate')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(0, 1)
            
            plot_count += 1
            
        except Exception as e:
            print(f"Error plotting success rates for {exp_name}: {e}")
            axes[idx].set_title(f'{exp_name} - Error')
    
    for idx in range(plot_count, 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison_bar_chart(results_dict, save_path=None):
    """Plot bar chart comparing final performance"""
    algorithms = ['SARSA', 'Q_Learning']
    metrics = ['success_rate', 'avg_reward']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for metric_idx, metric in enumerate(metrics):
        metric_data = {algo: [] for algo in algorithms}
        exp_names = []
        
        for exp_name, results in results_dict.items():
            exp_names.append(exp_name.replace('_', '\n').title())
            for algo in algorithms:
                if algo in results['testing']:
                    value = results['testing'][algo][metric]
                    # Ensure value is scalar
                    if hasattr(value, '__len__'):
                        value = value[0] if len(value) > 0 else 0
                    metric_data[algo].append(float(value))
                else:
                    metric_data[algo].append(0.0)
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        for i, algo in enumerate(algorithms):
            axes[metric_idx].bar(x + i*width, metric_data[algo], width, 
                               label=algo.replace('_', '-'), alpha=0.8)
        
        axes[metric_idx].set_xlabel('Environment')
        axes[metric_idx].set_ylabel(metric.replace('_', ' ').title())
        axes[metric_idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[metric_idx].set_xticks(x + width/2)
        axes[metric_idx].set_xticklabels(exp_names)
        axes[metric_idx].legend()
        axes[metric_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()