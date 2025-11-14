import mlflow
import numpy as np
from agents import SarsaAgent, QLearningAgent
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import itertools

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
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(evaluate_single_hyperparameter, args) for args in args_list]
        
        for future in as_completed(futures):
            hp, mean_reward, std_reward = future.result()
            
            # Log to MLflow in batch later to reduce overhead
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_hyperparams = hp.copy()
                print(f"    New best: {mean_reward:.2f} ± {std_reward:.2f} - {hp}")
    
    return best_hyperparams, best_reward

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

def run_final_evaluation_parallel(env_config, agent_class, best_hyperparams, num_episodes, num_seeds):
    """Run final evaluation in parallel"""
    print(f"  Parallel final evaluation with {mp.cpu_count()} cores...")
    
    # Prepare arguments
    args_list = [(env_config, agent_class, best_hyperparams, num_episodes, i) for i in range(num_seeds)]
    
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
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

def evaluate_single_seed(args):
    """Evaluate single seed (for parallel processing)"""
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

def run_sarsa_episode_with_steps_fast(env, agent, max_steps):
    """Optimized SARSA episode with steps"""
    state = env.reset()
    action = agent.choose_action(state)
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        next_state, reward = env.step(state, action)
        next_action = agent.choose_action(next_state)
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward, next_state, next_action, done)
        
        state, action = next_state, next_action
        total_reward += reward
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
        
        done = (next_state in env.goal_states_seq) or (step == max_steps - 1)
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    return total_reward, steps