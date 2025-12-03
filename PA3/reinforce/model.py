import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import torch.distributions

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return F.softmax(self.network(state), dim=-1)


class ValueNetwork(nn.Module):
    """Value network for baseline in REINFORCE"""
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class REINFORCEAgent:
    """REINFORCE Agent with optional baseline"""
    def __init__(self, state_dim, action_dim, device, gamma=0.99, 
                 lr_policy=1e-3, lr_value=1e-3, use_baseline=True):
        
        self.device = device
        torch.set_num_threads(1)
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        
        # Value network for baseline (if used)
        if use_baseline:
            self.value_net = ValueNetwork(state_dim).to(device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def select_action(self, state):
        """Select action using policy distribution"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        # Store for episode
        self.episode_states.append(state)
        self.episode_actions.append(action)
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.episode_rewards.append(reward)
    
    def update(self):
        """Update policy using episode returns"""
        if len(self.episode_rewards) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        rewards = np.array(self.episode_rewards)
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get log probabilities
        action_probs = self.policy_net(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        if self.use_baseline:
            # Calculate baseline values
            values = self.value_net(states).squeeze()
            
            # Advantage = returns - baseline
            advantages = returns - values.detach()
            
            # Policy loss
            policy_loss = -(log_probs * advantages).mean()
            
            # Value loss (TD(0) update)
            value_loss = F.mse_loss(values, returns)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()
            
            loss = policy_loss.item() + value_loss.item()
        else:
            # REINFORCE without baseline
            policy_loss = -(log_probs * returns).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            
            loss = policy_loss.item()
        
        # Clear episode buffer
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return loss
