import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from dueling_dqn.train import train_dqn_agent
from reinforce.train import train_reinforce_agent

def dqn_worker(args):
    env_name, variant, seed, num_episodes = args
    print(f"Running DuelingDQN {variant} on {env_name} - Seed {seed}")
    return train_dqn_agent(
        env_name=env_name,
        aggregation_type=variant,
        seed=seed,
        num_episodes=num_episodes
    )

def reinforce_worker(args):
    env_name, use_baseline, seed, num_episodes = args
    print(f"Running REINFORCE use_baseline={use_baseline} on {env_name} - Seed {seed}")
    return train_reinforce_agent(
        env_name=env_name,
        use_baseline=use_baseline,
        seed=seed,
        num_episodes=num_episodes
    )

def run_experiments(env_name, algorithm, variant, num_seeds=5, num_episodes=1000):
    """Run experiments with multiple seeds in parallel"""
    num_workers = min(num_seeds, os.cpu_count() or 1)
    
    if algorithm == 'DuelingDQN':
        args_list = [(env_name, variant, seed, num_episodes) for seed in range(num_seeds)]
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(dqn_worker, args_list)
    elif algorithm == 'REINFORCE':
        use_baseline = (variant == 'with_baseline')
        args_list = [(env_name, use_baseline, seed, num_episodes) for seed in range(num_seeds)]
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(reinforce_worker, args_list)
    
    all_returns = [result[0] for result in results]
    all_losses = [result[1] for result in results]
    
    # Pad sequences to equal length
    max_len = max(len(r) for r in all_returns)
    padded_returns = []
    for r in all_returns:
        if len(r) < max_len:
            padded_r = np.pad(r, (0, max_len - len(r)), 'constant', constant_values=r[-1])
        else:
            padded_r = r
        padded_returns.append(padded_r)
    
    return np.array(padded_returns), all_losses


def experiment_worker(args):
    env_name, algorithm, variant, seed, num_episodes = args
    print(f"Running {algorithm} ({variant}) on {env_name} - Seed {seed}")
    if algorithm == 'DuelingDQN':
        returns, _ = train_dqn_agent(
            env_name=env_name,
            aggregation_type=variant,
            seed=seed,
            num_episodes=num_episodes
        )
        return returns
    elif algorithm == 'REINFORCE':
        use_baseline = (variant == 'with_baseline')
        returns, _ = train_reinforce_agent(
            env_name=env_name,
            use_baseline=use_baseline,
            seed=seed,
            num_episodes=num_episodes
        )
        return returns


def plot_results(results_dict, title, ylabel='Episode Return'):
    """Plot results with mean and variance"""
    plt.figure(figsize=(10, 6))
    
    for label, data in results_dict.items():
        returns = data['returns']
        mean = np.mean(returns, axis=0)
        std = np.std(returns, axis=0)
        
        episodes = np.arange(len(mean))
        plt.plot(episodes, mean, label=label, linewidth=2)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
