# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_learning_curves(rewards, steps, title="Learning Curves", save_path=None):
    """Plot learning curves for rewards and steps"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate mean and std across seeds - ensure 1D arrays
    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)
    mean_steps = np.mean(steps, axis=0)
    std_steps = np.std(steps, axis=0)
    
    # Ensure arrays are 1-dimensional
    mean_rewards = np.squeeze(mean_rewards)
    std_rewards = np.squeeze(std_rewards)
    mean_steps = np.squeeze(mean_steps)
    std_steps = np.squeeze(std_steps)
    
    episodes = range(len(mean_rewards))
    
    # Reward plot
    ax1.plot(episodes, mean_rewards, 'b-', alpha=0.8, linewidth=2, label='Average Reward')
    ax1.fill_between(episodes, 
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    alpha=0.3, color='blue')
    ax1.set_title(f'{title} - Cumulative Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Steps plot
    ax2.plot(episodes, mean_steps, 'r-', alpha=0.8, linewidth=2, label='Average Steps')
    ax2.fill_between(episodes,
                    mean_steps - std_steps,
                    mean_steps + std_steps,
                    alpha=0.3, color='red')
    ax2.set_title(f'{title} - Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_state_visits(state_visits, env, title="State Visit Heatmap", save_path=None):
    """Plot heatmap of state visits"""
    # Average across all runs and reshape to grid
    avg_visits = np.mean(state_visits, axis=0)
    visit_grid = avg_visits.reshape(env.num_rows, env.num_cols)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(visit_grid, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Visits'}, square=True)
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_q_value_heatmap(q_values, env, title="Q-Values and Policy", save_path=None):
    """Plot Q-value heatmap with optimal policy overlay"""
    # Average Q-values across all runs
    avg_q = np.mean(q_values, axis=0)
    value_function = np.max(avg_q, axis=1)
    policy = np.argmax(avg_q, axis=1)
    
    value_grid = value_function.reshape(env.num_rows, env.num_cols)
    policy_grid = policy.reshape(env.num_rows, env.num_cols)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Q-value heatmap
    im1 = ax1.imshow(value_grid, cmap='viridis', interpolation='nearest')
    ax1.set_title(f'{title} - Q-Values')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1, label='Q-Value')
    
    # Add values to heatmap
    for i in range(env.num_rows):
        for j in range(env.num_cols):
            text = ax1.text(j, i, f'{value_grid[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=8)
    
    # Policy visualization
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    im2 = ax2.imshow(value_grid, cmap='viridis', interpolation='nearest', alpha=0.7)
    
    for i in range(env.num_rows):
        for j in range(env.num_cols):
            state = i * env.num_cols + j
            action = policy[state]
            ax2.text(j, i, action_symbols[action], 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white' if value_grid[i, j] < np.median(value_grid) else 'black')
    
    ax2.set_title(f'{title} - Optimal Policy')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2, label='Q-Value')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_summary(results, save_path=None):
    """Create comprehensive comparison summary of all experiments"""
    config_names = list(results.keys())
    avg_rewards = [np.mean(results[name]['final_results']['rewards']) for name in config_names]
    avg_steps = [np.mean(results[name]['final_results']['steps']) for name in config_names]
    std_rewards = [np.std(results[name]['final_results']['rewards']) for name in config_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Average rewards comparison
    bars1 = ax1.bar(range(len(config_names)), avg_rewards, color='skyblue', alpha=0.7, yerr=std_rewards, capsize=5)
    ax1.set_title('Average Final Rewards by Configuration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Reward')
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}±{std_rewards[i]:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Average steps comparison
    bars2 = ax2.bar(range(len(config_names)), avg_steps, color='lightcoral', alpha=0.7, capsize=5)
    ax2.set_title('Average Steps per Episode by Configuration', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Steps')
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Q-learning vs SARSA comparison
    ql_indices = [i for i, name in enumerate(config_names) if 'QL' in name]
    sarsa_indices = [i for i, name in enumerate(config_names) if 'SARSA' in name]
    
    ql_rewards = [avg_rewards[i] for i in ql_indices]
    sarsa_rewards = [avg_rewards[i] for i in sarsa_indices]
    
    if ql_rewards and sarsa_rewards:
        ax3.boxplot([ql_rewards, sarsa_rewards], labels=['Q-Learning', 'SARSA'])
        ax3.set_title('Q-Learning vs SARSA Reward Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Average Reward')
        ax3.grid(True, alpha=0.3)
    
    # Environment type comparison
    standard_indices = [i for i, name in enumerate(config_names) if '10x10' in name]
    fourroom_indices = [i for i, name in enumerate(config_names) if 'FourRoom' in name]
    
    standard_rewards = [avg_rewards[i] for i in standard_indices]
    fourroom_rewards = [avg_rewards[i] for i in fourroom_indices]
    
    if standard_rewards and fourroom_rewards:
        ax4.boxplot([standard_rewards, fourroom_rewards], labels=['10x10 Grid', 'Four Room'])
        ax4.set_title('Environment Type Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Reward')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()