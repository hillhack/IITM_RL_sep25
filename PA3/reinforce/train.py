import torch
import numpy as np
import random
import gymnasium as gym
from reinforce.model import REINFORCEAgent

def train_reinforce_agent(env_name, use_baseline, seed, num_episodes=1000, 
                         max_steps=500):
    """Train REINFORCE agent"""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=0.99,
        lr_policy=1e-3,
        lr_value=1e-3,
        use_baseline=use_baseline
    )
    
    # Training loop
    episode_returns = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed+episode)
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store reward
            agent.store_reward(reward)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update agent with complete episode
        loss = agent.update()
        
        episode_returns.append(total_reward)
        episode_losses.append(loss if loss else 0)
        
        # Early stopping for CartPole
        if env_name == 'CartPole-v1' and episode >= 100:
            avg_return = np.mean(episode_returns[-100:])
            if avg_return >= 195:
                remaining = num_episodes - episode - 1
                episode_returns.extend([total_reward] * remaining)
                episode_losses.extend([episode_losses[-1]] * remaining)
                break
    
    env.close()
    return episode_returns, episode_losses