import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=1.0, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))
    
    def choose_action(self, state, greedy=False):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon and not greedy:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])
    
    @abstractmethod
    def update(self, state, action, reward, next_state, next_action=None):
        """Update Q-values - to be implemented by subclasses"""
        pass
    
    def get_q_table(self):
        """Return the current Q-table"""
        return self.q_table
    
    def set_epsilon(self, epsilon):
        """Set exploration rate"""
        self.epsilon = epsilon
    
    def set_learning_rate(self, alpha):
        """Set learning rate"""
        self.alpha = alpha