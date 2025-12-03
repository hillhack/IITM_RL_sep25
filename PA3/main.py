import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from collections import defaultdict
from analysis import experiment_worker, plot_results

environments = ['CartPole-v1', 'Acrobot-v1']

print("Creating all experiment jobs across all environments and variants...")

jobs = []
num_seeds = 5
num_episodes = 500

for env in environments:
    # Dueling DQN Mean
    for seed in range(num_seeds):
        jobs.append((env, 'DuelingDQN', 'mean', seed, num_episodes))
    # Dueling DQN Max
    for seed in range(num_seeds):
        jobs.append((env, 'DuelingDQN', 'max', seed, num_episodes))
    # REINFORCE without baseline
    for seed in range(num_seeds):
        jobs.append((env, 'REINFORCE', 'without_baseline', seed, num_episodes))
    # REINFORCE with baseline
    for seed in range(num_seeds):
        jobs.append((env, 'REINFORCE', 'with_baseline', seed, num_episodes))

print(f"Running {len(jobs)} jobs in parallel using {os.cpu_count() or 4} cores...")

with mp.Pool(processes=os.cpu_count() or 4) as pool:
    raw_returns_list = pool.map(experiment_worker, jobs)

# Group results by env and label
from collections import defaultdict
env_variant_returns = defaultdict(list)
label_map = {
    'DuelingDQN': {'mean': 'DuelingDQN (Mean)', 'max': 'DuelingDQN (Max)'},
    'REINFORCE': {'without_baseline': 'REINFORCE (no baseline)', 'with_baseline': 'REINFORCE (with baseline)'}
}

for i, job in enumerate(jobs):
    env, algo, variant, _, _ = job
    label = label_map[algo][variant]
    env_variant_returns[(env, label)].append(raw_returns_list[i])

# Pad and prepare results
results = {}
for (env, label), returns_list in env_variant_returns.items():
    if env not in results:
        results[env] = {}
    
    # Pad sequences to equal length
    max_len = max(len(r) for r in returns_list)
    padded_returns = []
    for r in returns_list:
        if len(r) < max_len:
            padded_r = np.pad(r, (0, max_len - len(r)), 'constant', constant_values=r[-1])
        else:
            padded_r = np.array(r)
        padded_returns.append(padded_r)
    
    results[env][label] = {'returns': np.array(padded_returns)}

# Plot per environment
for env in environments:
    print(f"\n{'='*60}")
    print(f"Plotting results for {env}")
    print('='*60)
    
    env_results = results[env]
    
    # Dueling DQN comparison
    dqn_results = {
        'Mean Aggregation': env_results['DuelingDQN (Mean)'],
        'Max Aggregation': env_results['DuelingDQN (Max)']
    }
    plot_results(dqn_results, f'Dueling DQN Variants - {env}')
    
    # REINFORCE comparison
    reinforce_results = {
        'Without Baseline': env_results['REINFORCE (no baseline)'],
        'With Baseline': env_results['REINFORCE (with baseline)']
    }
    plot_results(reinforce_results, f'REINFORCE Variants - {env}')
