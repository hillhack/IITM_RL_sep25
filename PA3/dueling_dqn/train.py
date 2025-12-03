import torch
import numpy as np
import random
import gymnasium as gym
from collections import deque
from collections import namedtuple
from .model import DuelingDQNAgent

def train_dqn_agent(env_name, aggregation_type, seed, num_episodes=1000, 
                    max_steps=500, epsilon_start=1.0, epsilon_end=0.01, 
                    epsilon_decay=0.995):
    """Train Dueling DQN agent"""
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
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        target_update=10,
        aggregation_type=aggregation_type
    )
    
    # Training loop
    epsilon = epsilon_start
    episode_returns = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed+episode)
        total_reward = 0
        episode_loss = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss:
                episode_loss += loss
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        episode_returns.append(total_reward)
        episode_losses.append(episode_loss / (step + 1) if step > 0 else 0)
        
        # Early stopping for CartPole (solved at 195 average over 100 episodes)
        if env_name == 'CartPole-v1' and episode >= 100:
            avg_return = np.mean(episode_returns[-100:])
            if avg_return >= 195:
                # Fill remaining episodes with final return
                remaining = num_episodes - episode - 1
                episode_returns.extend([total_reward] * remaining)
                episode_losses.extend([episode_losses[-1]] * remaining)
                break
    
    env.close()
    return episode_returns, episode_losses
