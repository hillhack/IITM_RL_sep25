# Add this at the VERY TOP of run_optimized.py
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt

# run_optimized.py - Fix the parallel processing issues
import mlflow
import numpy as np
from env import create_standard_grid, create_four_room
from agents import SarsaAgent, QLearningAgent
from visualization import plot_learning_curves, plot_state_visits, plot_q_value_heatmap, plot_comparison_summary
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_hyperparameter_combinations_optimized(config):
    """Generate optimized hyperparameter combinations"""
    combinations = []
    
    # Pre-generate all combinations
    for alpha in config.learning_rates:
        for gamma in config.discount_factors:
            # Epsilon-greedy combinations
            for epsilon in config.epsilons:
                combinations.append({
                    'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon, 
                    'temp': 1.0, 'exploration_strategy': 'epsilon_greedy'
                })
            # Softmax combinations  
            for temp in config.temperatures:
                combinations.append({
                    'alpha': alpha, 'gamma': gamma, 'epsilon': 0.1,
                    'temp': temp, 'exploration_strategy': 'softmax'
                })
    
    return combinations

def evaluate_single_hyperparameter(args):
    """Evaluate single hyperparameter combination (for parallel processing)"""
    # Set matplotlib backend for child processes
    import matplotlib
    matplotlib.use('Agg')
    
    env_config, agent_class, hp, num_episodes, num_seeds, seed_offset = args
    
    # Create environment (lightweight)
    if 'goal_change' in env_config:
        from env import create_four_room
        env = create_four_room(
            start_state=np.array(env_config['start_state']),
            goal_change=env_config['goal_change'],
            transition_prob=env_config.get('transition_prob', 1.0)
        )
    else:
        from env import create_standard_grid
        env = create_standard_grid(
            start_state=np.array(env_config['start_state']),
            transition_prob=env_config.get('transition_prob', 0.7),
            wind=env_config.get('wind', False)
        )
    
    seed_rewards = []
    
    for seed in range(num_seeds):
        np.random.seed(seed + seed_offset)
        agent = agent_class(
            num_states=env.num_states,
            num_actions=env.num_actions,
            alpha=hp['alpha'],
            gamma=hp['gamma'],
            epsilon=hp['epsilon'],
            temp=hp['temp'],
            exploration_strategy=hp['exploration_strategy']
        )
        
        total_rewards = run_training_episodes_fast(env, agent, num_episodes, 100)
        avg_reward = np.mean(total_rewards[-100:])  # Last 100 episodes
        seed_rewards.append(avg_reward)
    
    return hp, np.mean(seed_rewards), np.std(seed_rewards)

def run_training_episodes_fast(env, agent, num_episodes, max_steps):
    """Optimized training episodes with pre-allocated arrays"""
    rewards = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        if agent.__class__.__name__ == 'SarsaAgent':
            rewards[episode] = run_sarsa_episode_fast(env, agent, max_steps)
        else:
            rewards[episode] = run_q_learning_episode_fast(env, agent, max_steps)
    
    return rewards

def run_sarsa_episode_fast(env, agent, max_steps):
    """Optimized SARSA episode"""
    state = env.reset()
    action = agent.choose_action(state)
    total_reward = 0
    
    for step in range(max_steps):
        next_state, reward = env.step(state, action)
        next_action = agent.choose_action(next_state)
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward, next_state, next_action, done)
        
        state, action = next_state, next_action
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def run_q_learning_episode_fast(env, agent, max_steps):
    """Optimized Q-learning episode"""
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward = env.step(state, action)
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def evaluate_single_seed(args):
    """Evaluate single seed (for parallel processing)"""
    # Set matplotlib backend for child processes
    import matplotlib
    matplotlib.use('Agg')
    
    env_config, agent_class, best_hyperparams, num_episodes, seed = args
    
    # Create environment
    if 'goal_change' in env_config:
        from env import create_four_room
        env = create_four_room(
            start_state=np.array(env_config['start_state']),
            goal_change=env_config['goal_change'],
            transition_prob=env_config.get('transition_prob', 1.0)
        )
    else:
        from env import create_standard_grid
        env = create_standard_grid(
            start_state=np.array(env_config['start_state']),
            transition_prob=env_config.get('transition_prob', 0.7),
            wind=env_config.get('wind', False)
        )
    
    np.random.seed(seed)
    agent = agent_class(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=best_hyperparams['alpha'],
        gamma=best_hyperparams['gamma'],
        epsilon=best_hyperparams['epsilon'],
        temp=best_hyperparams['temp'],
        exploration_strategy=best_hyperparams['exploration_strategy']
    )
    
    seed_rewards = []
    seed_steps = []
    
    for episode in range(num_episodes):
        if agent_class.__name__ == 'SarsaAgent':
            reward, steps = run_sarsa_episode_with_steps_fast(env, agent, 100)
        else:
            reward, steps = run_q_learning_episode_with_steps_fast(env, agent, 100)
        
        seed_rewards.append(reward)
        seed_steps.append(steps)
    
    return seed_rewards, seed_steps, agent.state_visits.copy(), agent.Q.copy()

# Fix the episode functions to extract scalar rewards
def run_sarsa_episode_fast(env, agent, max_steps):
    """Optimized SARSA episode"""
    state = env.reset()
    action = agent.choose_action(state)
    total_reward = 0
    
    for step in range(max_steps):
        next_state, reward = env.step(state, action)
        # Extract scalar from reward array
        reward_scalar = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
        next_action = agent.choose_action(next_state)
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward_scalar, next_state, next_action, done)
        
        state, action = next_state, next_action
        total_reward += reward_scalar
        
        if done:
            break
    
    return total_reward

def run_q_learning_episode_fast(env, agent, max_steps):
    """Optimized Q-learning episode"""
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward = env.step(state, action)
        # Extract scalar from reward array
        reward_scalar = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward_scalar, next_state, done)
        
        state = next_state
        total_reward += reward_scalar
        
        if done:
            break
    
    return total_reward

def run_sarsa_episode_with_steps_fast(env, agent, max_steps):
    """Optimized SARSA episode with steps"""
    state = env.reset()
    action = agent.choose_action(state)
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        next_state, reward = env.step(state, action)
        # Extract scalar from reward array
        reward_scalar = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
        next_action = agent.choose_action(next_state)
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward_scalar, next_state, next_action, done)
        
        state, action = next_state, next_action
        total_reward += reward_scalar
        steps += 1
        
        if done:
            break
    
    return total_reward, steps

def run_q_learning_episode_with_steps_fast(env, agent, max_steps):
    """Optimized Q-learning episode with steps"""
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward = env.step(state, action)
        # Extract scalar from reward array
        reward_scalar = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward_scalar, next_state, done)
        
        state = next_state
        total_reward += reward_scalar
        steps += 1
        
        if done:
            break
    
    return total_reward, steps

def run_hyperparameter_tuning_parallel(env_config, agent_class, hyperparams, num_episodes, num_seeds=5):
    """Run hyperparameter tuning in parallel"""
    print(f"  Parallel tuning with {mp.cpu_count()} cores...")
    
    # Filter hyperparameters based on exploration strategy
    strategy = env_config['exploration_strategy']
    filtered_hyperparams = [hp for hp in hyperparams if hp['exploration_strategy'] == strategy]
    
    best_reward = -np.inf
    best_hyperparams = None
    
    # Prepare arguments for parallel processing
    args_list = []
    for hp in filtered_hyperparams:
        args_list.append((env_config, agent_class, hp, num_episodes, num_seeds, len(args_list)*1000))
    
    # Use all available CPU cores
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(args_list))) as executor:
        futures = [executor.submit(evaluate_single_hyperparameter, args) for args in args_list]
        
        for future in as_completed(futures):
            hp, mean_reward, std_reward = future.result()
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_hyperparams = hp.copy()
                print(f"    New best: {mean_reward:.2f} ± {std_reward:.2f} - {hp}")
    
    return best_hyperparams, best_reward

def run_final_evaluation_parallel(env_config, agent_class, best_hyperparams, num_episodes, num_seeds):
    """Run final evaluation in parallel"""
    print(f"  Parallel final evaluation with {mp.cpu_count()} cores...")
    
    # Prepare arguments
    args_list = [(env_config, agent_class, best_hyperparams, num_episodes, i) for i in range(num_seeds)]
    
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(args_list))) as executor:
        futures = [executor.submit(evaluate_single_seed, args) for args in args_list]
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    # Combine results
    all_rewards = np.array([r[0] for r in results])
    all_steps = np.array([r[1] for r in results])
    all_state_visits = np.array([r[2] for r in results])
    all_q_values = np.array([r[3] for r in results])
    
    return {
        'rewards': all_rewards,
        'steps': all_steps,
        'state_visits': all_state_visits,
        'q_values': all_q_values
    }

class OptimizedGridWorldExperimentRunner:
    def __init__(self):
        from configs import ExperimentConfig
        self.config = ExperimentConfig()
        self.results = {}
    
    def run_all_experiments(self):
        """Run all experiments with optimal performance"""
        print("=== OPTIMIZED GRID WORLD EXPERIMENTS ===")
        print(f"Using {os.cpu_count()} CPU cores")
        print(f"Total configurations: 20")
        
        start_time = time.time()
        
        # Pre-generate all hyperparameter combinations
        all_hyperparams = get_hyperparameter_combinations_optimized(self.config)
        print(f"Hyperparameter combinations per config: {len(all_hyperparams)}")
        
        # Run 10x10 Grid World experiments
        self.run_10x10_experiments_parallel(all_hyperparams)
        
        # Run Four Room experiments  
        self.run_four_room_experiments_parallel(all_hyperparams)
        
        total_time = time.time() - start_time
        print(f"\n=== ALL EXPERIMENTS COMPLETED IN {total_time/60:.1f} MINUTES ===")
        
        # Generate summary
        self.generate_summary_report()
    
    def run_10x10_experiments_parallel(self, all_hyperparams):
        """Run 10x10 experiments in optimized manner"""
        print("\n" + "="*50)
        print("10x10 GRID WORLD EXPERIMENTS")
        print("="*50)
        
        # Q-learning configurations
        ql_configs = [
            {'transition_prob': 0.7, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 0.7, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 0.7, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 0.7, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': False},
        ]
        
        # SARSA configurations
        sarsa_configs = [
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': True},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': True},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': True},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': True},
        ]
        
        # Run Q-learning configurations
        for i, config in enumerate(ql_configs):
            config_name = f"10x10_QL_{i+1}"
            print(f"\n[{i+1}/8] Q-Learning: {config}")
            self.run_single_configuration_optimized(config, QLearningAgent, config_name, all_hyperparams)
        
        # Run SARSA configurations
        for i, config in enumerate(sarsa_configs):
            config_name = f"10x10_SARSA_{i+1}"
            print(f"\n[{i+1}/8] SARSA: {config}")
            self.run_single_configuration_optimized(config, SarsaAgent, config_name, all_hyperparams)
    
    def run_four_room_experiments_parallel(self, all_hyperparams):
        """Run Four Room experiments in optimized manner"""
        print("\n" + "="*50)
        print("FOUR ROOM EXPERIMENTS")
        print("="*50)
        
        four_room_configs = [
            {'goal_change': False, 'exploration_strategy': 'epsilon_greedy'},
            {'goal_change': True, 'exploration_strategy': 'epsilon_greedy'},
        ]
        
        # Add common parameters
        for config in four_room_configs:
            config.update({'transition_prob': 1.0, 'start_state': [[8,0]]})
        
        # Run Q-learning configurations
        for i, config in enumerate(four_room_configs):
            config_name = f"FourRoom_QL_{i+1}"
            print(f"\n[{i+1}/2] Q-Learning: {config}")
            self.run_single_configuration_optimized(config, QLearningAgent, config_name, all_hyperparams)
        
        # Run SARSA configurations
        for i, config in enumerate(four_room_configs):
            config_name = f"FourRoom_SARSA_{i+1}"
            print(f"\n[{i+1}/2] SARSA: {config}")
            self.run_single_configuration_optimized(config, SarsaAgent, config_name, all_hyperparams)
    
    def run_single_configuration_optimized(self, config, agent_class, config_name, all_hyperparams):
        """Optimized single configuration runner"""
        # Minimal MLflow logging to reduce overhead
        mlflow.start_run(run_name=config_name, nested=True)
        
        try:
            # Create environment
            if 'goal_change' in config:
                env = create_four_room(
                    start_state=np.array(config['start_state']),
                    goal_change=config['goal_change'],
                    transition_prob=config['transition_prob']
                )
            else:
                env = create_standard_grid(
                    start_state=np.array(config['start_state']),
                    transition_prob=config['transition_prob'],
                    wind=config.get('wind', False)
                )
            
            # Parallel hyperparameter tuning
            best_hyperparams, best_reward = run_hyperparameter_tuning_parallel(
                config, agent_class, all_hyperparams,
                self.config.num_episodes_tuning, self.config.num_seeds_tuning
            )
            
            # Parallel final evaluation
            final_results = run_final_evaluation_parallel(
                config, agent_class, best_hyperparams,
                self.config.num_episodes_final, self.config.num_seeds_final
            )
            
            # Store results
            self.results[config_name] = {
                'config': config,
                'best_hyperparams': best_hyperparams,
                'final_results': final_results,
                'env': env
            }
            
            # Generate visualizations
            self.generate_visualizations_optimized(config_name, final_results, env)
            
            # Batch MLflow logging
            mlflow.log_params(config)
            mlflow.log_params({'best_' + k: v for k, v in best_hyperparams.items()})
            mlflow.log_metric('best_mean_reward', best_reward)
            
            print(f"  ✓ Completed {config_name} | Reward: {best_reward:.2f}")
            
        finally:
            mlflow.end_run()
    
    def generate_visualizations_optimized(self, config_name, results, env):
        """Optimized visualization generation"""
        ensure_dir('plots')
        
        # Generate plots
        plot_learning_curves(
            results['rewards'], results['steps'],
            title=f"{config_name} - Learning Curves",
            save_path=f"plots/{config_name}_learning_curves.png"
        )
        
        plot_state_visits(
            results['state_visits'], env,
            title=f"{config_name} - State Visit Heatmap", 
            save_path=f"plots/{config_name}_state_visits.png"
        )
        
        plot_q_value_heatmap(
            results['q_values'], env,
            title=f"{config_name} - Q-Values and Policy",
            save_path=f"plots/{config_name}_q_values_policy.png"
        )
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("FINAL SUMMARY REPORT")
        print("="*60)
        
        plot_comparison_summary(self.results, save_path="plots/summary_comparison.png")
        
        # Print best configurations
        self.print_best_configurations()
        
        print(f"\nResults saved to MLflow and plots/ directory")
        print("Use 'mlflow ui' to view detailed results")

    def print_best_configurations(self):
        """Print best configurations"""
        ql_results = {k: v for k, v in self.results.items() if 'QL' in k}
        sarsa_results = {k: v for k, v in self.results.items() if 'SARSA' in k}
        
        if ql_results:
            best_ql = max(ql_results.items(), key=lambda x: np.mean(x[1]['final_results']['rewards']))
            reward = np.mean(best_ql[1]['final_results']['rewards'])
            print(f"\n🏆 BEST Q-LEARNING: {best_ql[0]}")
            print(f"   Average Reward: {reward:.2f}")
            print(f"   Hyperparams: {best_ql[1]['best_hyperparams']}")
        
        if sarsa_results:
            best_sarsa = max(sarsa_results.items(), key=lambda x: np.mean(x[1]['final_results']['rewards']))
            reward = np.mean(best_sarsa[1]['final_results']['rewards'])
            print(f"\n🏆 BEST SARSA: {best_sarsa[0]}")
            print(f"   Average Reward: {reward:.2f}")
            print(f"   Hyperparams: {best_sarsa[1]['best_hyperparams']}")

def main():
    # Optimized MLflow setup
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("GridWorld_Optimized")
    
    runner = OptimizedGridWorldExperimentRunner()
    runner.run_all_experiments()

if __name__ == "__main__":
    main()