from .base_agent import BaseAgent
import numpy as np

class SarsaAgent(BaseAgent):
    """On-policy SARSA algorithm implementation"""
    
    def update(self, state, action, reward, next_state, next_action=None):
        """
        SARSA update: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Current action
            reward: Reward received
            next_state: Next state
            next_action: Next action (for SARSA)
        """
        if next_action is None:
            next_action = self.choose_action(next_state)
        
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        
        self.q_table[state, action] += self.alpha * td_error
        return next_action