# experiments/run_experiments_parallel.py
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent
from env import create_standard_grid, create_four_room

def run_single_episode(args):
    """Run a single episode - optimized for parallel execution"""
    env_config, agent_type, episode_idx, max_steps = args
    
    # Create fresh environment and agent for this episode
    if env_config['type'] == 'standard':
        env = create_standard_grid(**env_config['params'])
    else:
        env = create_four_room(**env_config['params'])
    
    if agent_type == 'sarsa':
        agent = SarsaAgent(env.num_states, env.num_actions, **env_config['agent_params'])
    else:
        agent = QLearningAgent(env.num_states, env.num_actions, **env_config['agent_params'])
    
    # Decay epsilon
    initial_epsilon = env_config['agent_params']['epsilon']
    decayed_epsilon = initial_epsilon * (1 - episode_idx / env_config['num_episodes'])
    agent.set_epsilon(max(0.01, decayed_epsilon))
    
    # Run episode
    state = env.reset()
    total_reward = 0
    steps = 0
    reached_goal = False
    
    action = agent.choose_action(state)
    
    for step in range(max_steps):
        next_state, reward = env.step(state, action)
        total_reward += float(reward)
        steps += 1
        
        if next_state in env.goal_states_seq:
            reached_goal = True
            agent.update(state, action, reward, next_state)
            break
        
        if agent_type == 'sarsa':
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            action = next_action
        else:
            agent.update(state, action, reward, next_state)
            action = agent.choose_action(next_state)
        
        state = next_state
    
    return {
        'episode': episode_idx,
        'agent_type': agent_type,
        'reward': total_reward,
        'steps': steps,
        'success': reached_goal,
        'q_table': agent.get_q_table() if episode_idx == env_config['num_episodes'] - 1 else None
    }

def run_parallel_experiment(env_config, num_episodes=1000, max_steps=100):
    """Run experiment in parallel for both agents"""
    print(f"Running parallel experiment: {env_config['name']}")
    
    # Prepare arguments for parallel execution
    sarsa_args = [(env_config, 'sarsa', i, max_steps) for i in range(num_episodes)]
    qlearn_args = [(env_config, 'qlearn', i, max_steps) for i in range(num_episodes)]
    
    all_args = sarsa_args + qlearn_args
    
    # Use all available cores
    num_workers = min(mp.cpu_count(), len(all_args))
    print(f"Using {num_workers} CPU cores for {len(all_args)} tasks")
    
    results = {'sarsa': [], 'qlearn': []}
    final_q_tables = {'sarsa': None, 'qlearn': None}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_args = {executor.submit(run_single_episode, arg): arg for arg in all_args}
        
        completed = 0
        for future in as_completed(future_to_args):
            result = future.result()
            agent_type = result['agent_type']
            
            # Store results
            results[agent_type].append({
                'reward': result['reward'],
                'steps': result['steps'],
                'success': result['success']
            })
            
            # Store final Q-table
            if result['q_table'] is not None:
                final_q_tables[agent_type] = result['q_table']
            
            completed += 1
            if completed % 500 == 0:
                print(f"  Completed {completed}/{len(all_args)} episodes")
    
    # Create final agents with learned Q-tables
    if env_config['type'] == 'standard':
        env = create_standard_grid(**env_config['params'])
    else:
        env = create_four_room(**env_config['params'])
    
    sarsa_agent = SarsaAgent(env.num_states, env.num_actions, **env_config['agent_params'])
    qlearn_agent = QLearningAgent(env.num_states, env.num_actions, **env_config['agent_params'])
    
    if final_q_tables['sarsa'] is not None:
        sarsa_agent.q_table = final_q_tables['sarsa']
    if final_q_tables['qlearn'] is not None:
        qlearn_agent.q_table = final_q_tables['qlearn']
    
    return {
        'env': env,
        'sarsa_agent': sarsa_agent,
        'qlearn_agent': qlearn_agent,
        'training': {
            'sarsa': {
                'rewards': [r['reward'] for r in results['sarsa']],
                'steps': [r['steps'] for r in results['sarsa']],
                'success': [r['success'] for r in results['sarsa']]
            },
            'qlearn': {
                'rewards': [r['reward'] for r in results['qlearn']],
                'steps': [r['steps'] for r in results['qlearn']],
                'success': [r['success'] for r in results['qlearn']]
            }
        }
    }

def run_all_parallel_experiments():
    """Run all experiments in parallel"""
    experiments_config = {
        'std_no_wind': {
            'name': 'Standard Grid (No Wind)',
            'type': 'standard',
            'params': {'wind': False, 'transition_prob': 0.7},
            'agent_params': {'learning_rate': 0.1, 'discount_factor': 0.99, 'epsilon': 0.2},
            'num_episodes': 2000  # Can run more episodes since it's parallel
        },
        'std_with_wind': {
            'name': 'Standard Grid (With Wind)',
            'type': 'standard',
            'params': {'wind': True, 'transition_prob': 0.7},
            'agent_params': {'learning_rate': 0.1, 'discount_factor': 0.99, 'epsilon': 0.2},
            'num_episodes': 2000
        },
        'four_room_fixed': {
            'name': 'Four Room (Fixed Goal)',
            'type': 'four_room',
            'params': {'goal_change': False, 'transition_prob': 1.0},
            'agent_params': {'learning_rate': 0.1, 'discount_factor': 0.99, 'epsilon': 0.2},
            'num_episodes': 2000
        },
        'four_room_changing': {
            'name': 'Four Room (Changing Goal)',
            'type': 'four_room',
            'params': {'goal_change': True, 'transition_prob': 1.0},
            'agent_params': {'learning_rate': 0.3, 'discount_factor': 0.9, 'epsilon': 0.4},
            'num_episodes': 3000  # More episodes for harder environment
        }
    }
    
    all_results = {}
    
    for exp_name, config in experiments_config.items():
        print(f"\n{'='*60}")
        print(f"PARALLEL EXPERIMENT: {config['name']}")
        print(f"{'='*60}")
        
        try:
            import time
            start_time = time.time()
            
            results = run_parallel_experiment(config, config['num_episodes'])
            all_results[exp_name] = results
            
            end_time = time.time()
            print(f"✓ Completed in {end_time - start_time:.2f} seconds")
            
            # Test the learned policies
            test_results = test_learned_policy_parallel(results)
            print(f"Test Results:")
            print(f"  SARSA: Success={test_results['sarsa']['success_rate']:.1%}, "
                  f"Avg Reward={test_results['sarsa']['avg_reward']:.1f}")
            print(f"  Q-Learning: Success={test_results['qlearn']['success_rate']:.1%}, "
                  f"Avg Reward={test_results['qlearn']['avg_reward']:.1f}")
                  
        except Exception as e:
            print(f"✗ Error in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results

def test_learned_policy_parallel(experiment_results, num_tests=100):
    """Test the learned policy"""
    env = experiment_results['env']
    sarsa_agent = experiment_results['sarsa_agent']
    qlearn_agent = experiment_results['qlearn_agent']
    
    test_results = {'sarsa': {'rewards': [], 'success': []}, 
                   'qlearn': {'rewards': [], 'success': []}}
    
    # Test both agents
    for agent_name, agent in [('sarsa', sarsa_agent), ('qlearn', qlearn_agent)]:
        for _ in range(num_tests):
            state = env.reset()
            total_reward = 0
            reached_goal = False
            
            for step in range(100):
                action = agent.choose_action(state, greedy=True)
                next_state, reward = env.step(state, action)
                total_reward += reward
                
                if next_state in env.goal_states_seq:
                    reached_goal = True
                    break
                
                state = next_state
            
            test_results[agent_name]['rewards'].append(total_reward)
            test_results[agent_name]['success'].append(reached_goal)
        
        test_results[agent_name]['success_rate'] = np.mean(test_results[agent_name]['success'])
        test_results[agent_name]['avg_reward'] = np.mean(test_results[agent_name]['rewards'])
    
    return test_results