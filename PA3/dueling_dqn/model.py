import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from collections import deque, namedtuple

class DuelingDQNNetwork(nn.Module):
    """Neural network for Dueling DQN with configurable advantage aggregation"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, aggregation_type='mean'):
        super(DuelingDQNNetwork, self).__init__()
        self.aggregation_type = aggregation_type
        
        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q-value aggregation based on type
        if self.aggregation_type == 'mean':
            # Type-1: Mean-normalized advantage
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        elif self.aggregation_type == 'max':
            # Type-2: Max-normalized advantage
            q_values = value + (advantages - advantages.max(dim=1, keepdim=True)[0])
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
        
        return q_values


class DuelingDQNAgent:
    """Dueling DQN Agent with experience replay and target network"""
    def __init__(self, state_dim, action_dim, device, gamma=0.99, 
                 lr=1e-3, batch_size=64, buffer_size=10000,
                 target_update=10, aggregation_type='mean'):
        
        self.device = device
        torch.set_num_threads(1)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.aggregation_type = aggregation_type
        
        # Main and target networks
        self.policy_net = DuelingDQNNetwork(state_dim, action_dim, 
                                           aggregation_type=aggregation_type).to(device)
        self.target_net = DuelingDQNNetwork(state_dim, action_dim,
                                           aggregation_type=aggregation_type).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', 
                                    ['state', 'action', 'reward', 'next_state', 'done'])
        
        # Training metrics
        self.steps = 0
    
    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randrange(self.policy_net.advantage_stream[-1].out_features)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        exp = self.experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
    
    def update(self):
        """Update network using experience replay"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        experiences = random.sample(self.buffer, self.batch_size)
        batch = self.experience(*zip(*experiences))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
