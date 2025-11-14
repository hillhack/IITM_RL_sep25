# configs.py
import numpy as np

class ExperimentConfig:
    def __init__(self):
        # Hyperparameter search space - FULL RANGES
        self.learning_rates = [0.001, 0.01, 0.1, 1.0]
        self.discount_factors = [0.7, 0.8, 0.9, 1.0]
        self.epsilons = [0.001, 0.01, 0.05, 0.1]
        self.temperatures = [0.01, 0.1, 1, 2]
        
        # Experiment parameters - ADJUSTED FOR FULL RUN
        self.num_episodes_tuning = 500  # Full tuning
        self.num_episodes_final = 100    # Final evaluation
        self.max_steps_per_episode = 100
        self.num_seeds_tuning = 5        # 5 seeds for tuning as required
        self.num_seeds_final = 100       # 100 seeds for final evaluation as required
        
        # 10x10 Grid World configurations for Q-learning
        self.q_learning_configs_10x10 = [
            {'transition_prob': 0.7, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 0.7, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 0.7, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 0.7, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': False},
        ]
        
        # 10x10 Grid World configurations for SARSA
        self.sarsa_configs_10x10 = [
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': False},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'epsilon_greedy', 'wind': True},
            {'transition_prob': 1.0, 'start_state': [[0,4]], 'exploration_strategy': 'softmax', 'wind': True},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'epsilon_greedy', 'wind': True},
            {'transition_prob': 1.0, 'start_state': [[3,6]], 'exploration_strategy': 'softmax', 'wind': True},
        ]
        
        # Four Room configurations
        self.four_room_configs = [
            {'goal_change': False, 'exploration_strategy': 'epsilon_greedy'},
            {'goal_change': True, 'exploration_strategy': 'epsilon_greedy'},
        ]