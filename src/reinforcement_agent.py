"""
Reinforcement Learning Agent for SmartDynamic pricing system.
This module implements RL approaches for optimal pricing.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import os
import joblib
from loguru import logger


class PricingEnvironment(gym.Env):
    """
    Custom Gym environment for price optimization.
    Simulates how pricing decisions affect demand and revenue.
    """
    
    def __init__(self, config):
        """
        Initialize the pricing environment.
        
        Args:
            config (dict): Environment configuration parameters
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.demand_model = config.get("demand_model", None)
        self.price_range = config.get("price_range", (50, 150))
        self.price_step = config.get("price_step", 5)
        self.max_steps = config.get("max_steps", 30)
        self.product_id = config.get("product_id", None)
        self.context_features = config.get("context_features", {})
        self.current_step = 0
        
        # Calculate price levels based on range and step
        self.price_levels = np.arange(self.price_range[0], self.price_range[1] + self.price_step, self.price_step)
        self.n_price_levels = len(self.price_levels)
        
        # Define action space (discrete price levels)
        self.action_space = spaces.Discrete(self.n_price_levels)
        
        # Define observation space
        n_features = 4 + len(self.context_features)  # Base features + context features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.current_price = None
        self.price_history = []
        self.demand_history = []
        self.revenue_history = []
        
        logger.info(f"Initialized pricing environment with {self.n_price_levels} price levels")
    
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            np.array: Current state observation
        """
        # Base features: current price, price change, avg demand, price trend
        current_price_normalized = (self.current_price - self.price_range[0]) / (self.price_range[1] - self.price_range[0])
        price_change = 0 if len(self.price_history) < 2 else (self.current_price - self.price_history[-2]) / self.price_range[1]
        avg_demand = np.mean(self.demand_history[-5:]) if self.demand_history else 0.5
        price_trend = 0
        if len(self.price_history) >= 5:
            # Calculate short-term price trend
            recent_prices = self.price_history[-5:]
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0] / self.price_range[1]
            
        base_features = [current_price_normalized, price_change, avg_demand, price_trend]
        
        # Add context features
        context_values = list(self.context_features.values())
        
        return np.array(base_features + context_values, dtype=np.float32)
    
    def _calculate_demand(self, price):
        """
        Calculate demand based on price and context.
        Uses demand model if provided, otherwise a simple elasticity model.
        
        Args:
            price (float): Current price
            
        Returns:
            float: Estimated demand
        """
        if self.demand_model is not None:
            # Prepare features for demand model
            features = {'price': price}
            features.update(self.context_features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Get demand prediction
            demand = float(self.demand_model.predict(features_df)[0])
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, self.config.get("demand_noise", 0.05) * demand)
            demand = max(0, demand + noise)
            
        else:
            # Simple demand model based on price elasticity
            base_demand = self.config.get("base_demand", 100)
            price_elasticity = self.config.get("price_elasticity", -1.5)
            
            # Apply elasticity formula: % change in demand = elasticity * % change in price
            reference_price = self.config.get("reference_price", np.mean(self.price_range))
            price_ratio = price / reference_price
            demand = base_demand * (price_ratio ** price_elasticity)
            
            # Add context effects (e.g., weekend effect, season)
            for feature, value in self.context_features.items():
                if feature == "is_weekend" and value == 1:
                    # Increased weekend demand
                    demand *= self.config.get("weekend_factor", 1.2)
                elif feature == "is_holiday" and value == 1:
                    # Increased holiday demand
                    demand *= self.config.get("holiday_factor", 1.5)
                    
            # Add randomness
            noise = np.random.normal(0, self.config.get("demand_noise", 0.1) * demand)
            demand = max(0, demand + noise)
            
        return demand
    
    def _calculate_reward(self, price, demand):
        """
        Calculate reward based on price, demand, and business objectives.
        
        Args:
            price (float): Current price
            demand (float): Current demand
            
        Returns:
            float: Reward value
        """
        # Calculate revenue
        revenue = price * demand
        
        # Calculate profit (assuming a cost)
        unit_cost = self.config.get("unit_cost", self.price_range[0] * 0.6)
        profit = (price - unit_cost) * demand
        
        # Calculate customer satisfaction penalty for high prices
        max_reasonable_price = self.config.get("max_reasonable_price", self.price_range[1] * 0.9)
        satisfaction_penalty = max(0, (price / max_reasonable_price - 1) * self.config.get("satisfaction_weight", 0.2) * revenue)
        
        # Calculate price change penalty to discourage excessive fluctuations
        price_stability_penalty = 0
        if len(self.price_history) >= 2:
            price_change = abs(price - self.price_history[-2])
            price_stability_penalty = price_change * self.config.get("stability_weight", 0.1) * revenue
        
        # Final reward
        reward = profit - satisfaction_penalty - price_stability_penalty
        
        # Normalize reward
        max_possible_reward = (self.price_range[1] - unit_cost) * self.config.get("base_demand", 100)
        normalized_reward = reward / max_possible_reward
        
        return normalized_reward, revenue, profit
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options
            
        Returns:
            tuple: Initial observation and info
        """
        super().reset(seed=seed)
        
        # Reset internal variables
        self.current_step = 0
        
        # Start at a random price within the range
        start_price_index = self.np_random.integers(0, self.n_price_levels)
        self.current_price = self.price_levels[start_price_index]
        
        # Reset histories
        self.price_history = [self.current_price]
        self.demand_history = []
        self.revenue_history = []
        
        # Update context features if dynamic
        if callable(self.config.get("update_context")):
            self.context_features = self.config["update_context"](self.current_step)
        
        # Get initial observation
        self.state = self._get_observation()
        
        return self.state, {}
    
    def step(self, action):
        """
        Take a step in the environment by setting a new price.
        
        Args:
            action (int): Index of price level
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert action to price
        self.current_price = self.price_levels[action]
        self.price_history.append(self.current_price)
        
        # Calculate demand based on price
        demand = self._calculate_demand(self.current_price)
        self.demand_history.append(demand)
        
        # Calculate reward
        reward, revenue, profit = self._calculate_reward(self.current_price, demand)
        self.revenue_history.append(revenue)
        
        # Update context features if dynamic
        if callable(self.config.get("update_context")):
            self.context_features = self.config["update_context"](self.current_step)
        
        # Update state
        self.state = self._get_observation()
        
        # Increment step counter
        self.current_step += 1
        
        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Info dictionary with additional data
        info = {
            "price": self.current_price,
            "demand": demand,
            "revenue": revenue,
            "profit": profit,
            "step": self.current_step
        }
        
        return self.state, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment (visualization).
        
        Returns:
            None
        """
        if len(self.price_history) <= 1:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot prices
        steps = list(range(len(self.price_history)))
        ax1.plot(steps, self.price_history, marker='o', label='Price')
        ax1.set_ylabel('Price')
        ax1.set_title('Price and Demand Over Time')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        # Plot demand and revenue
        if self.demand_history:
            ax2.plot(steps[:-1], self.demand_history, marker='s', color='green', label='Demand')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Demand')
            ax2.grid(True)
            ax2.legend(loc='upper left')
            
            # Add revenue as line on secondary axis
            ax3 = ax2.twinx()
            ax3.plot(steps[:-1], self.revenue_history, marker='d', color='purple', linestyle='--', label='Revenue')
            ax3.set_ylabel('Revenue')
            ax3.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


class ReinforcementAgent:
    """
    Reinforcement Learning Agent for dynamic pricing optimization.
    """
    
    def __init__(self, env_config=None, agent_type="ppo", agent_config=None):
        """
        Initialize the RL agent.
        
        Args:
            env_config (dict, optional): Configuration for pricing environment
            agent_type (str): Type of RL agent ('ppo', 'a2c', 'dqn')
            agent_config (dict, optional): Configuration for RL algorithm
        """
        self.env_config = env_config or {}
        self.agent_type = agent_type.lower()
        self.agent_config = agent_config or {}
        
        # Create the environment
        self.env = PricingEnvironment(self.env_config)
        
        # Wrap in DummyVecEnv as required by Stable-Baselines
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Create the agent
        self._create_agent()
        
        logger.info(f"Initialized {self.agent_type} reinforcement learning agent")
    
    def _create_agent(self):
        """Create the RL agent based on specified type"""
        try:
            if self.agent_type == "ppo":
                self.model = PPO(
                    policy="MlpPolicy",
                    env=self.vec_env,
                    learning_rate=self.agent_config.get("learning_rate", 3e-4),
                    n_steps=self.agent_config.get("n_steps", 2048),
                    batch_size=self.agent_config.get("batch_size", 64),
                    n_epochs=self.agent_config.get("n_epochs", 10),
                    gamma=self.agent_config.get("gamma", 0.99),
                    gae_lambda=self.agent_config.get("gae_lambda", 0.95),
                    clip_range=self.agent_config.get("clip_range", 0.2),
                    ent_coef=self.agent_config.get("ent_coef", 0.0),
                    verbose=self.agent_config.get("verbose", 0)
                )
            elif self.agent_type == "a2c":
                self.model = A2C(
                    policy="MlpPolicy",
                    env=self.vec_env,
                    learning_rate=self.agent_config.get("learning_rate", 7e-4),
                    n_steps=self.agent_config.get("n_steps", 5),
                    gamma=self.agent_config.get("gamma", 0.99),
                    ent_coef=self.agent_config.get("ent_coef", 0.0),
                    verbose=self.agent_config.get("verbose", 0)
                )
            elif self.agent_type == "dqn":
                self.model = DQN(
                    policy="MlpPolicy",
                    env=self.vec_env,
                    learning_rate=self.agent_config.get("learning_rate", 1e-4),
                    buffer_size=self.agent_config.get("buffer_size", 10000),
                    learning_starts=self.agent_config.get("learning_starts", 1000),
                    batch_size=self.agent_config.get("batch_size", 32),
                    gamma=self.agent_config.get("gamma", 0.99),
                    target_update_interval=self.agent_config.get("target_update_interval", 500),
                    exploration_fraction=self.agent_config.get("exploration_fraction", 0.1),
                    exploration_final_eps=self.agent_config.get("exploration_final_eps", 0.05),
                    verbose=self.agent_config.get("verbose", 0)
                )
            else:
                logger.error(f"Unsupported agent type: {self.agent_type}")
                raise ValueError(f"Unsupported agent type: {self.agent_type}")
                
        except Exception as e:
            logger.error(f"Error creating reinforcement learning agent: {e}")
            raise
    
    def train(self, total_timesteps):
        """
        Train the reinforcement learning agent.
        
        Args:
            total_timesteps (int): Total number of training timesteps
            
        Returns:
            self: Trained agent
        """
        try:
            # Set up evaluation callback
            eval_env = PricingEnvironment(self.env_config)
            eval_env = DummyVecEnv([lambda: eval_env])
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.agent_config.get("save_path", "./models"),
                log_path=self.agent_config.get("log_path", "./logs"),
                eval_freq=self.agent_config.get("eval_freq", 10000),
                n_eval_episodes=self.agent_config.get("n_eval_episodes", 5),
                deterministic=True,
                render=False
            )
            
            # Train the agent
            logger.info(f"Starting training for {total_timesteps} timesteps")
            self.model.learn(total_timesteps=total_timesteps, callback=eval_callback)
            logger.info("Training completed")
            
            return self
            
        except Exception as e:
            logger.error(f"Error training the agent: {e}")
            raise
    
    def get_optimal_price(self, context_features=None):
        """
        Get the optimal price recommendation for a specific context.
        
        Args:
            context_features (dict, optional): Context features for pricing decision
            
        Returns:
            float: Optimal price
        """
        try:
            # Update environment context
            if context_features:
                self.env.context_features = context_features
            
            # Reset environment to get initial observation
            obs, _ = self.env.reset()
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Convert action to price
            optimal_price = self.env.price_levels[action]
            
            logger.info(f"Recommended optimal price: {optimal_price}")
            return optimal_price
            
        except Exception as e:
            logger.error(f"Error getting optimal price: {e}")
            return None
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes (int): Number of evaluation episodes
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Create evaluation environment
            eval_env = PricingEnvironment(self.env_config)
            eval_env = DummyVecEnv([lambda: eval_env])
            
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=n_episodes,
                deterministic=True
            )
            
            metrics = {
                'mean_reward': mean_reward,
                'std_reward': std_reward
            }
            
            logger.info(f"Evaluation results - Mean reward: {mean_reward:.4f}, Std: {std_reward:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating the agent: {e}")
            return None
    
    def simulate_pricing_strategy(self, n_steps=30, context_update_func=None):
        """
        Simulate the pricing strategy over time.
        
        Args:
            n_steps (int): Number of simulation steps
            context_update_func (callable, optional): Function to update context at each step
            
        Returns:
            dict: Simulation results
        """
        try:
            # Create simulation environment
            sim_env = PricingEnvironment(self.env_config)
            
            # Reset environment
            obs, _ = sim_env.reset()
            
            # Initialize result containers
            prices = []
            demands = []
            revenues = []
            profits = []
            rewards = []
            contexts = []
            
            # Run simulation
            for step in range(n_steps):
                # Update context if function provided
                if context_update_func:
                    context = context_update_func(step)
                    sim_env.context_features = context
                    contexts.append(context)
                    
                    # Update observation with new context
                    obs = sim_env._get_observation()
                
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = sim_env.step(action)
                
                # Store results
                prices.append(info["price"])
                demands.append(info["demand"])
                revenues.append(info["revenue"])
                profits.append(info["profit"])
                rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            # Compile results
            results = {
                'prices': prices,
                'demands': demands,
                'revenues': revenues,
                'profits': profits,
                'rewards': rewards,
                'contexts': contexts if contexts else None,
                'total_revenue': sum(revenues),
                'total_profit': sum(profits),
                'avg_reward': sum(rewards) / len(rewards) if rewards else 0
            }
            
            logger.info(f"Simulation completed - Total revenue: {results['total_revenue']:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error simulating pricing strategy: {e}")
            return None
    
    def plot_simulation_results(self, results):
        """
        Plot simulation results.
        
        Args:
            results (dict): Simulation results from simulate_pricing_strategy
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot prices
            steps = list(range(len(results['prices'])))
            ax1.plot(steps, results['prices'], marker='o', label='Price')
            ax1.set_ylabel('Price')
            ax1.set_title('Dynamic Pricing Simulation Results')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            
            # Plot demand
            ax2.plot(steps, results['demands'], marker='s', color='green', label='Demand')
            ax2.set_ylabel('Demand')
            ax2.grid(True)
            ax2.legend(loc='upper left')
            
            # Plot revenue and profit
            ax3.plot(steps, results['revenues'], marker='d', color='purple', label='Revenue')
            ax3.plot(steps, results['profits'], marker='x', color='blue', linestyle='--', label='Profit')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Value')
            ax3.grid(True)
            ax3.legend(loc='upper left')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting simulation results: {e}")
            return None
    
    def save(self, path):
        """
        Save the trained agent.
        
        Args:
            path (str): Path to save the agent
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.model.save(path)
            
            # Save environment configuration separately
            with open(f"{path}_env_config.pkl", 'wb') as f:
                joblib.dump(self.env_config, f)
                
            # Save agent metadata
            metadata = {
                'agent_type': self.agent_type,
                'agent_config': self.agent_config
            }
            
            with open(f"{path}_metadata.pkl", 'wb') as f:
                joblib.dump(metadata, f)
                
            logger.info(f"Agent successfully saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving agent: {e}")
            return False
    
    @classmethod
    def load(cls, path):
        """
        Load a trained agent.
        
        Args:
            path (str): Path to the saved agent
            
        Returns:
            ReinforcementAgent: Loaded agent instance
        """
        try:
            # Load environment configuration
            with open(f"{path}_env_config.pkl", 'rb') as f:
                env_config = joblib.load(f)
                
            # Load agent metadata
            with open(f"{path}_metadata.pkl", 'rb') as f:
                metadata = joblib.load(f)
                
            # Create instance with environment config
            instance = cls(
                env_config=env_config,
                agent_type=metadata['agent_type'],
                agent_config=metadata['agent_config']
            )
            
            # Load the model based on agent type
            if metadata['agent_type'] == "ppo":
                instance.model = PPO.load(path, env=instance.vec_env)
            elif metadata['agent_type'] == "a2c":
                instance.model = A2C.load(path, env=instance.vec_env)
            elif metadata['agent_type'] == "dqn":
                instance.model = DQN.load(path, env=instance.vec_env)
            else:
                logger.error(f"Unsupported agent type: {metadata['agent_type']}")
                return None
                
            logger.info(f"Agent successfully loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading agent: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    try:
        # Define environment configuration
        env_config = {
            "price_range": (80, 150),
            "price_step": 5,
            "max_steps": 30,
            "base_demand": 100,
            "price_elasticity": -1.5,
            "unit_cost": 60,
            "demand_noise": 0.1,
            "context_features": {
                "is_weekend": 0,
                "is_holiday": 0,
                "season": 0.5  # normalized season (0=winter, 0.33=spring, 0.66=summer, 1=fall)
            }
        }
        
        # Define agent configuration
        agent_config = {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "verbose": 1
        }
        
        # Create agent
        agent = ReinforcementAgent(env_config=env_config, agent_type="ppo", agent_config=agent_config)
        
        # Train agent (in a real scenario, use more timesteps)
        agent.train(total_timesteps=10000)
        
        # Evaluate agent
        metrics = agent.evaluate(n_episodes=5)
        print(f"Evaluation metrics: {metrics}")
        
        # Get optimal price for specific context
        weekend_context = env_config["context_features"].copy()
        weekend_context["is_weekend"] = 1
        optimal_price = agent.get_optimal_price(context_features=weekend_context)
        print(f"Optimal weekend price: {optimal_price}")
        
        # Simulate pricing strategy
        def update_context(step):
            # Example context updating function
            context = env_config["context_features"].copy()
            context["is_weekend"] = 1 if step % 7 >= 5 else 0
            context["is_holiday"] = 1 if step == 15 else 0
            return context
            
        results = agent.simulate_pricing_strategy(n_steps=30, context_update_func=update_context)
        
        # Plot results
        fig = agent.plot_simulation_results(results)
        plt.show()
        
    except Exception as e:
        print(f"Error in example: {e}")
