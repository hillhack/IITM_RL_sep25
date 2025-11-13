from .base_agent import BaseAgent
import numpy as np

class QLearningAgent(BaseAgent):
    """Off-policy Q-Learning algorithm implementation"""
    
    def update(self, state, action, reward, next_state, next_action=None):
        """
        Q-Learning update: Q(s,a) = Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Current action
            reward: Reward received
            next_state: Next state
            next_action: Not used in Q-learning (for interface compatibility)
        """
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        self.q_table[state, action] += self.alpha * td_error
        return self.choose_action(next_state)