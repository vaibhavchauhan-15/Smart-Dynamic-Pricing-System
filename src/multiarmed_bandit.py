"""
Multi-armed Bandit module for SmartDynamic pricing system.
This module implements various MAB algorithms for price testing and optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib
import os
from datetime import datetime
from loguru import logger


class BanditArm:
    """
    Bandit arm class representing a price point with reward distribution.
    """
    
    def __init__(self, price, alpha=1.0, beta=1.0):
        """
        Initialize bandit arm with Beta prior.
        
        Args:
            price (float): Price point for this arm
            alpha (float): Alpha parameter for Beta distribution
            beta (float): Beta parameter for Beta distribution
        """
        self.price = price
        self.alpha = alpha
        self.beta = beta
        self.pulls = 0
        self.rewards = []
        
    def update(self, reward):
        """
        Update arm statistics after receiving a reward.
        
        Args:
            reward (float): Reward value (normalized between 0 and 1)
        """
        self.pulls += 1
        self.rewards.append(reward)
        
        # Update Beta distribution parameters
        self.alpha += reward
        self.beta += (1.0 - reward)
        
    def mean(self):
        """
        Get the mean of the posterior distribution.
        
        Returns:
            float: Mean reward estimate
        """
        return self.alpha / (self.alpha + self.beta)
    
    def sample(self):
        """
        Sample a value from the posterior distribution.
        
        Returns:
            float: Sampled value
        """
        return np.random.beta(self.alpha, self.beta)
    
    def confidence_bounds(self, confidence=0.95):
        """
        Get confidence interval for the arm's reward estimate.
        
        Args:
            confidence (float): Confidence level (0-1)
            
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        lower = stats.beta.ppf((1 - confidence) / 2, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - (1 - confidence) / 2, self.alpha, self.beta)
        return lower, upper


class PricingBandit:
    """
    Multi-armed bandit for price optimization.
    Implements several MAB strategies including epsilon-greedy, UCB, and Thompson sampling.
    """
    
    def __init__(self, prices, strategy="thompson", config=None):
        """
        Initialize the pricing bandit with arms for each price point.
        
        Args:
            prices (list): List of price points
            strategy (str): MAB strategy ('epsilon_greedy', 'ucb', 'thompson')
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.strategy = strategy.lower()
        self.prices = sorted(prices)
        self.arms = {price: BanditArm(price) for price in prices}
        self.t = 0  # Time step counter
        self.selections = []
        self.rewards = []
        
        # Strategy-specific parameters
        self.epsilon = self.config.get("epsilon", 0.1)  # For epsilon-greedy
        self.ucb_c = self.config.get("ucb_c", 2.0)  # Exploration constant for UCB
        
        logger.info(f"Initialized PricingBandit with {len(prices)} price points and {strategy} strategy")
    
    def select_arm(self):
        """
        Select an arm based on the chosen strategy.
        
        Returns:
            float: Selected price
        """
        self.t += 1
        
        # Initial exploration phase - try each arm at least once
        unexplored_arms = [price for price, arm in self.arms.items() if arm.pulls == 0]
        if unexplored_arms:
            selected_price = np.random.choice(unexplored_arms)
            logger.debug(f"Exploring new arm with price {selected_price}")
            return selected_price
            
        # Apply strategy
        if self.strategy == "epsilon_greedy":
            # Epsilon-greedy strategy
            if np.random.random() < self.epsilon:
                # Explore: select randomly
                selected_price = np.random.choice(self.prices)
                logger.debug(f"Epsilon exploration: selected price {selected_price}")
            else:
                # Exploit: select the best arm so far
                selected_price = max(self.arms.items(), key=lambda x: x[1].mean())[0]
                logger.debug(f"Epsilon exploitation: selected price {selected_price}")
                
        elif self.strategy == "ucb":
            # Upper Confidence Bound strategy
            ucb_values = {}
            for price, arm in self.arms.items():
                if arm.pulls == 0:
                    ucb_values[price] = float('inf')
                else:
                    # UCB formula: mean + c * sqrt(ln(t) / pulls)
                    exploration = self.ucb_c * np.sqrt(np.log(self.t) / arm.pulls)
                    ucb_values[price] = arm.mean() + exploration
            
            selected_price = max(ucb_values.items(), key=lambda x: x[1])[0]
            logger.debug(f"UCB selection: selected price {selected_price}")
            
        elif self.strategy == "thompson":
            # Thompson sampling strategy
            samples = {price: arm.sample() for price, arm in self.arms.items()}
            selected_price = max(samples.items(), key=lambda x: x[1])[0]
            logger.debug(f"Thompson sampling: selected price {selected_price}")
            
        else:
            # Default to random selection
            logger.warning(f"Unknown strategy: {self.strategy}, using random selection")
            selected_price = np.random.choice(self.prices)
        
        self.selections.append(selected_price)
        return selected_price
    
    def update(self, price, reward):
        """
        Update the bandit after receiving a reward for the selected price.
        
        Args:
            price (float): The price that was selected
            reward (float): The observed reward (should be normalized between 0-1)
        """
        # Ensure reward is normalized between 0 and 1
        reward = max(0.0, min(1.0, reward))
        
        # Update the corresponding arm
        if price in self.arms:
            self.arms[price].update(reward)
            self.rewards.append(reward)
            logger.debug(f"Updated arm with price {price}, reward {reward}")
        else:
            logger.warning(f"Price {price} not found in arms")
    
    def get_optimal_price(self):
        """
        Get the current best price based on mean reward estimates.
        
        Returns:
            float: Best price
        """
        if not any(arm.pulls > 0 for arm in self.arms.values()):
            logger.warning("No arms have been pulled yet, returning middle price")
            return np.median(self.prices)
            
        best_price = max(self.arms.items(), key=lambda x: x[1].mean() if x[1].pulls > 0 else -float('inf'))[0]
        logger.info(f"Current optimal price: {best_price}")
        return best_price
    
    def get_arm_stats(self):
        """
        Get statistics for all arms.
        
        Returns:
            pd.DataFrame: DataFrame with arm statistics
        """
        stats_data = []
        
        for price, arm in self.arms.items():
            lower, upper = arm.confidence_bounds()
            stats_data.append({
                'price': price,
                'pulls': arm.pulls,
                'mean_reward': arm.mean(),
                'lower_bound': lower,
                'upper_bound': upper
            })
        
        return pd.DataFrame(stats_data).sort_values(by='price')
    
    def plot_arm_distributions(self):
        """
        Plot the posterior distributions of all arms.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.linspace(0, 1, 1000)
            
            for price, arm in self.arms.items():
                if arm.pulls > 0:  # Only plot arms that have been pulled
                    y = stats.beta.pdf(x, arm.alpha, arm.beta)
                    ax.plot(x, y, label=f"Price: {price} (Mean: {arm.mean():.3f})")
            
            ax.set_xlabel("Reward Probability")
            ax.set_ylabel("Density")
            ax.set_title("Posterior Distributions by Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting arm distributions: {e}")
            return None
    
    def plot_reward_history(self):
        """
        Plot the history of rewards over time.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            if not self.selections or not self.rewards:
                logger.warning("No history to plot")
                return None
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot price selections over time
            time_steps = range(1, len(self.selections) + 1)
            ax1.plot(time_steps, self.selections, marker='o', linestyle='-', alpha=0.6)
            ax1.set_ylabel("Selected Price")
            ax1.set_title("Price Selection History")
            ax1.grid(True, alpha=0.3)
            
            # Plot rewards over time
            ax2.plot(time_steps[:-1], self.rewards, marker='s', color='green', linestyle='-', alpha=0.6)
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Reward")
            ax2.set_title("Reward History")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting reward history: {e}")
            return None
    
    def save(self, path):
        """
        Save the bandit state.
        
        Args:
            path (str): Path to save the bandit
            
        Returns:
            bool: Success status
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            state = {
                'prices': self.prices,
                'arms': self.arms,
                'strategy': self.strategy,
                'config': self.config,
                't': self.t,
                'selections': self.selections,
                'rewards': self.rewards
            }
            
            with open(path, 'wb') as f:
                joblib.dump(state, f)
                
            logger.info(f"Bandit state saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving bandit: {e}")
            return False
    
    @classmethod
    def load(cls, path):
        """
        Load a saved bandit.
        
        Args:
            path (str): Path to the saved bandit
            
        Returns:
            PricingBandit: Loaded bandit instance
        """
        try:
            with open(path, 'rb') as f:
                state = joblib.load(f)
                
            instance = cls(
                prices=state['prices'],
                strategy=state['strategy'],
                config=state['config']
            )
            
            instance.arms = state['arms']
            instance.t = state['t']
            instance.selections = state['selections']
            instance.rewards = state['rewards']
            
            logger.info(f"Bandit loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading bandit: {e}")
            return None


class ContextualPricingBandit:
    """
    Contextual multi-armed bandit for price optimization with context.
    """
    
    def __init__(self, prices, context_dims, strategy="linucb", config=None):
        """
        Initialize the contextual pricing bandit.
        
        Args:
            prices (list): List of price points
            context_dims (int): Number of context dimensions
            strategy (str): MAB strategy ('linucb', 'neural')
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.strategy = strategy.lower()
        self.prices = sorted(prices)
        self.context_dims = context_dims
        
        # LinUCB parameters
        self.alpha = self.config.get("alpha", 1.0)  # Exploration parameter
        
        # Initialize model parameters for each arm
        if self.strategy == "linucb":
            self.arms = {}
            for price in prices:
                self.arms[price] = {
                    'A': np.identity(context_dims),  # A matrix (d x d)
                    'b': np.zeros(context_dims),     # b vector (d x 1)
                    'theta': np.zeros(context_dims), # theta (d x 1)
                    'pulls': 0,
                    'rewards': []
                }
        elif self.strategy == "neural":
            # Neural contextual bandit would require a neural network for each arm
            # or a single network with multiple outputs
            # This is a simplified placeholder
            self.arms = {price: {'pulls': 0, 'rewards': []} for price in prices}
            logger.warning("Neural contextual bandit is a placeholder and not fully implemented")
        else:
            logger.error(f"Unsupported contextual bandit strategy: {strategy}")
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        self.t = 0
        self.selections = []
        self.rewards = []
        self.contexts = []
        
        logger.info(f"Initialized ContextualPricingBandit with {len(prices)} prices and {context_dims} context dimensions")
    
    def _get_linucb_arm(self, context):
        """
        Select arm using LinUCB algorithm.
        
        Args:
            context (np.array): Context vector
            
        Returns:
            float: Selected price
        """
        ucb_values = {}
        
        for price, arm in self.arms.items():
            # Skip arms with no pulls during initial exploration
            if arm['pulls'] == 0:
                ucb_values[price] = float('inf')
                continue
                
            A_inv = np.linalg.inv(arm['A'])
            theta = A_inv.dot(arm['b'])
            
            # Store theta for future reference
            arm['theta'] = theta
            
            # Calculate UCB
            ucb = theta.dot(context) + self.alpha * np.sqrt(context.dot(A_inv).dot(context))
            ucb_values[price] = ucb
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def select_arm(self, context):
        """
        Select an arm based on context and strategy.
        
        Args:
            context (np.array): Context features
            
        Returns:
            float: Selected price
        """
        self.t += 1
        
        # Convert context to numpy array if needed
        if not isinstance(context, np.ndarray):
            context = np.array(context)
        
        # Reshape context if needed
        if context.shape != (self.context_dims,):
            context = context.reshape(self.context_dims)
        
        # Store context
        self.contexts.append(context)
        
        # Initial exploration phase - try each arm at least once
        unexplored_arms = [price for price, arm in self.arms.items() if arm['pulls'] == 0]
        if unexplored_arms:
            selected_price = np.random.choice(unexplored_arms)
            logger.debug(f"Exploring new arm with price {selected_price}")
            return selected_price
        
        # Apply strategy
        if self.strategy == "linucb":
            selected_price = self._get_linucb_arm(context)
        elif self.strategy == "neural":
            # Placeholder for neural contextual bandit
            # In practice, would use a neural network to predict rewards
            selected_price = np.random.choice(self.prices)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using random selection")
            selected_price = np.random.choice(self.prices)
        
        self.selections.append(selected_price)
        return selected_price
    
    def update(self, price, reward, context=None):
        """
        Update the bandit after receiving a reward.
        
        Args:
            price (float): Selected price
            reward (float): Observed reward (should be normalized between 0-1)
            context (np.array, optional): Context features (if not provided, uses the last one)
        """
        # Ensure reward is normalized between 0 and 1
        reward = max(0.0, min(1.0, reward))
        
        # Use the last context if not provided
        if context is None:
            if not self.contexts:
                logger.error("No context available for update")
                return
            context = self.contexts[-1]
        
        # Convert context to numpy array if needed
        if not isinstance(context, np.ndarray):
            context = np.array(context)
        
        # Reshape context if needed
        if context.shape != (self.context_dims,):
            context = context.reshape(self.context_dims)
        
        # Update the corresponding arm
        if price in self.arms:
            arm = self.arms[price]
            arm['pulls'] += 1
            arm['rewards'].append(reward)
            self.rewards.append(reward)
            
            if self.strategy == "linucb":
                # Update A matrix and b vector
                arm['A'] += np.outer(context, context)
                arm['b'] += reward * context
                
            logger.debug(f"Updated arm with price {price}, reward {reward}")
        else:
            logger.warning(f"Price {price} not found in arms")
    
    def get_optimal_price(self, context):
        """
        Get the optimal price for a given context.
        
        Args:
            context (np.array): Context features
            
        Returns:
            float: Best price for the context
        """
        if not any(arm['pulls'] > 0 for arm in self.arms.values()):
            logger.warning("No arms have been pulled yet, returning middle price")
            return np.median(self.prices)
        
        # Convert context to numpy array if needed
        if not isinstance(context, np.ndarray):
            context = np.array(context)
        
        # Reshape context if needed
        if context.shape != (self.context_dims,):
            context = context.reshape(self.context_dims)
        
        if self.strategy == "linucb":
            estimated_rewards = {}
            
            for price, arm in self.arms.items():
                if arm['pulls'] > 0:
                    # Calculate expected reward using learned theta
                    A_inv = np.linalg.inv(arm['A'])
                    theta = A_inv.dot(arm['b'])
                    estimated_rewards[price] = theta.dot(context)
                else:
                    estimated_rewards[price] = -float('inf')
            
            best_price = max(estimated_rewards.items(), key=lambda x: x[1])[0]
            
        elif self.strategy == "neural":
            # Placeholder - would use neural network prediction
            best_price = max(self.arms.items(), key=lambda x: np.mean(x[1]['rewards']) if x[1]['pulls'] > 0 else -float('inf'))[0]
        
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using price with highest average reward")
            best_price = max(self.arms.items(), key=lambda x: np.mean(x[1]['rewards']) if x[1]['pulls'] > 0 else -float('inf'))[0]
        
        logger.info(f"Optimal price for context: {best_price}")
        return best_price
    
    def get_arm_stats(self):
        """
        Get statistics for all arms.
        
        Returns:
            pd.DataFrame: DataFrame with arm statistics
        """
        stats_data = []
        
        for price, arm in self.arms.items():
            mean_reward = np.mean(arm['rewards']) if arm['rewards'] else 0
            stats_data.append({
                'price': price,
                'pulls': arm['pulls'],
                'mean_reward': mean_reward
            })
        
        return pd.DataFrame(stats_data).sort_values(by='price')
    
    def save(self, path):
        """
        Save the contextual bandit state.
        
        Args:
            path (str): Path to save the bandit
            
        Returns:
            bool: Success status
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            state = {
                'prices': self.prices,
                'context_dims': self.context_dims,
                'arms': self.arms,
                'strategy': self.strategy,
                'config': self.config,
                't': self.t,
                'selections': self.selections,
                'rewards': self.rewards,
                'contexts': self.contexts
            }
            
            with open(path, 'wb') as f:
                joblib.dump(state, f)
                
            logger.info(f"Contextual bandit state saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving contextual bandit: {e}")
            return False
    
    @classmethod
    def load(cls, path):
        """
        Load a saved contextual bandit.
        
        Args:
            path (str): Path to the saved bandit
            
        Returns:
            ContextualPricingBandit: Loaded bandit instance
        """
        try:
            with open(path, 'rb') as f:
                state = joblib.load(f)
                
            instance = cls(
                prices=state['prices'],
                context_dims=state['context_dims'],
                strategy=state['strategy'],
                config=state['config']
            )
            
            instance.arms = state['arms']
            instance.t = state['t']
            instance.selections = state['selections']
            instance.rewards = state['rewards']
            instance.contexts = state['contexts']
            
            logger.info(f"Contextual bandit loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading contextual bandit: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    try:
        # Define price points
        prices = [89.99, 94.99, 99.99, 104.99, 109.99]
        
        # Initialize bandit
        bandit = PricingBandit(prices, strategy="thompson")
        
        # Simulate bandit performance
        n_rounds = 100
        
        # Simulate true reward means (unknown to the algorithm)
        true_means = {
            89.99: 0.3,
            94.99: 0.5,
            99.99: 0.7,  # Best price
            104.99: 0.6,
            109.99: 0.4
        }
        
        for i in range(n_rounds):
            # Select price
            price = bandit.select_arm()
            
            # Simulate reward (with noise)
            true_mean = true_means[price]
            reward = np.random.beta(true_mean*10, (1-true_mean)*10)  # Simulate noisy rewards
            
            # Update bandit
            bandit.update(price, reward)
        
        # Print final arm statistics
        stats = bandit.get_arm_stats()
        print("Arm Statistics:")
        print(stats)
        
        # Get best price
        best_price = bandit.get_optimal_price()
        print(f"Optimal price: {best_price}")
        
        # Plot arm distributions
        fig1 = bandit.plot_arm_distributions()
        plt.show()
        
        # Plot reward history
        fig2 = bandit.plot_reward_history()
        plt.show()
        
    except Exception as e:
        print(f"Error in example: {e}")
