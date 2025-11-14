# test_single.py
import mlflow
import numpy as np
from env import create_standard_grid
from agents import QLearningAgent
from experiments import run_hyperparameter_tuning, run_final_evaluation, get_hyperparameter_combinations
from configs import ExperimentConfig
from visualization import plot_learning_curves, plot_state_visits, plot_q_value_heatmap
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# test_single.py - Update the config
def test_single_configuration():
    """Test a single configuration to debug"""
    mlflow.set_experiment("GridWorld_Test")
    
    # Simple configuration - use string directly
    config = {
        'transition_prob': 0.7,
        'start_state': [[0,4]],
        'exploration_strategy': 'epsilon_greedy',  # String, not enum
        'wind': False
    }
    
    # Create environment
    env = create_standard_grid(
        start_state=np.array(config['start_state']),
        transition_prob=config['transition_prob'],
        wind=config['wind']
    )
    
    # Reduced hyperparameter search for testing
    test_config = ExperimentConfig()
    test_config.learning_rates = [0.1, 1.0]
    test_config.discount_factors = [0.9, 1.0]
    test_config.epsilons = [0.1]
    test_config.temperatures = [1.0]
    
    hyperparams = get_hyperparameter_combinations(test_config)
    
    print("Testing hyperparameter tuning...")
    best_hyperparams, best_reward = run_hyperparameter_tuning(
        env, QLearningAgent, config, hyperparams,
        num_episodes=100, num_seeds=2
    )
    
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best reward: {best_reward}")
    
    # Final evaluation
    print("Running final evaluation...")
    final_results = run_final_evaluation(
        env, QLearningAgent, best_hyperparams, config,
        num_episodes=50, num_seeds=3
    )
    
    # Generate visualizations
    ensure_dir('test_plots')
    plot_learning_curves(
        final_results['rewards'], 
        final_results['steps'],
        title="Test Configuration - Learning Curves",
        save_path="test_plots/test_learning_curves.png"
    )
    
    plot_state_visits(
        final_results['state_visits'],
        env,
        title="Test Configuration - State Visit Heatmap",
        save_path="test_plots/test_state_visits.png"
    )
    
    plot_q_value_heatmap(
        final_results['q_values'],
        env,
        title="Test Configuration - Q-Values and Policy",
        save_path="test_plots/test_q_values_policy.png"
    )
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_single_configuration()