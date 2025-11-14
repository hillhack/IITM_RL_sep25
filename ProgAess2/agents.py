# agents.py
import numpy as np
import random

class BaseAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1, temp=1.0, exploration_strategy='epsilon_greedy'):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.temp = temp
        self.exploration_strategy = exploration_strategy
        self.Q = np.zeros((num_states, num_actions))
        self.state_visits = np.zeros(num_states)
    
    def softmax(self, x):
        exp_x = np.exp((x - np.max(x)) / self.temp)
        return exp_x / np.sum(exp_x)
    
    def choose_action(self, state, training=True):
        self.state_visits[state] += 1
        
        if not training:
            return np.argmax(self.Q[state])
        
        if self.exploration_strategy == 'epsilon_greedy':
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
            else:
                return np.argmax(self.Q[state])
        
        elif self.exploration_strategy == 'softmax':
            probabilities = self.softmax(self.Q[state])
            return np.random.choice(self.num_actions, p=probabilities)
    
    def get_policy(self):
        return np.argmax(self.Q, axis=1)
    
    def get_value_function(self):
        return np.max(self.Q, axis=1)
    
    def reset_state_visits(self):
        self.state_visits = np.zeros(self.num_states)

class SarsaAgent(BaseAgent):
    def update(self, state, action, reward, next_state, next_action, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
        
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

class QLearningAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])