import numpy as np
import matplotlib.pyplot as plt

def analyze_policy(env, agent, agent_name, num_tests=100):
    """
    Analyze the learned policy
    
    Args:
        env: Environment
        agent: Trained agent
        agent_name: Name of the agent
        num_tests: Number of test episodes
    
    Returns:
        analysis: Dictionary with analysis results
    """
    rewards, steps, successes = [], [], []
    state_visits = np.zeros(env.num_states)
    
    for _ in range(num_tests):
        state = env.reset()
        total_reward = 0
        step_count = 0
        success = False
        
        while step_count < 100:
            state_visits[state] += 1
            action = agent.choose_action(state, greedy=True)
            next_state, reward = env.step(state, action)
            
            total_reward += reward
            step_count += 1
            
            if next_state in env.goal_states_seq:
                success = True
                break
            
            state = next_state
        
        rewards.append(total_reward)
        steps.append(step_count)
        successes.append(success)
    
    # Calculate policy metrics
    success_rate = np.mean(successes)
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    
    # Get optimal actions
    optimal_actions = np.argmax(agent.get_q_table(), axis=1)
    
    analysis = {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'state_visits': state_visits,
        'optimal_actions': optimal_actions,
        'q_table': agent.get_q_table()
    }
    
    print(f"\n{agent_name} Policy Analysis:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Max Q-value: {np.max(agent.get_q_table()):.2f}")
    print(f"Min Q-value: {np.min(agent.get_q_table()):.2f}")
    
    return analysis

def compare_policies(env, sarsa_agent, qlearn_agent):
    """Compare policies of both agents"""
    sarsa_analysis = analyze_policy(env, sarsa_agent, "SARSA")
    qlearn_analysis = analyze_policy(env, qlearn_agent, "Q-Learning")
    
    # Policy agreement
    policy_agreement = np.mean(
        sarsa_analysis['optimal_actions'] == qlearn_analysis['optimal_actions']
    )
    print(f"\nPolicy Agreement: {policy_agreement:.2%}")
    
    return sarsa_analysis, qlearn_analysis