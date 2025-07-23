"""
Price Optimizer module for SmartDynamic pricing system.
This module combines different pricing strategies and provides a unified interface.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import matplotlib.pyplot as plt
from loguru import logger


class PriceOptimizer:
    """
    Price optimization engine that combines multiple pricing strategies.
    This is the main interface for price recommendations.
    """
    
    def __init__(self, config=None):
        """
        Initialize price optimizer with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary for pricing strategies
        """
        self.config = config or {}
        self.models = {}
        self.strategies = {}
        self.default_strategy = self.config.get("default_strategy", "demand_based")
        self.price_bounds = self.config.get("price_bounds", {})  # Per-product price bounds
        self.global_bounds = self.config.get("global_bounds", (0, float('inf')))  # Default bounds
        
        # Custom price adjustments
        self.psychological_price_func = self.config.get("psychological_price_func", self._default_psychological_price)
        
        logger.info("Price Optimizer initialized")
    
    def add_strategy(self, name, strategy_type, model=None, config=None):
        """
        Add a pricing strategy to the optimizer.
        
        Args:
            name (str): Strategy name
            strategy_type (str): Type of strategy ('demand_based', 'rl', 'bandit', 'competitor', 'rules')
            model (object, optional): Model object for the strategy
            config (dict, optional): Strategy-specific configuration
            
        Returns:
            bool: Success status
        """
        try:
            self.strategies[name] = {
                'type': strategy_type,
                'model': model,
                'config': config or {}
            }
            
            if model:
                self.models[name] = model
                
            logger.info(f"Added {strategy_type} pricing strategy: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy {name}: {e}")
            return False
    
    def remove_strategy(self, name):
        """
        Remove a pricing strategy.
        
        Args:
            name (str): Strategy name
            
        Returns:
            bool: Success status
        """
        if name in self.strategies:
            del self.strategies[name]
            if name in self.models:
                del self.models[name]
                
            logger.info(f"Removed pricing strategy: {name}")
            return True
        else:
            logger.warning(f"Strategy not found: {name}")
            return False
    
    def set_default_strategy(self, name):
        """
        Set the default pricing strategy.
        
        Args:
            name (str): Strategy name
            
        Returns:
            bool: Success status
        """
        if name in self.strategies:
            self.default_strategy = name
            logger.info(f"Set default strategy to: {name}")
            return True
        else:
            logger.warning(f"Strategy not found: {name}")
            return False
    
    def get_price_recommendation(self, product_id, features=None, strategy=None):
        """
        Get price recommendation for a product based on features.
        
        Args:
            product_id (str): Product ID
            features (dict, optional): Features for pricing decision
            strategy (str, optional): Strategy name to use (uses default if not specified)
            
        Returns:
            dict: Price recommendation with metadata
        """
        try:
            # Use specified strategy or default
            strategy_name = strategy or self.default_strategy
            
            if strategy_name not in self.strategies:
                logger.warning(f"Strategy {strategy_name} not found, using first available strategy")
                strategy_name = next(iter(self.strategies)) if self.strategies else None
                
                if not strategy_name:
                    logger.error("No pricing strategies available")
                    return None
            
            strategy_info = self.strategies[strategy_name]
            price = None
            
            # Get raw price from strategy
            if strategy_info['type'] == 'demand_based':
                price = self._get_demand_based_price(product_id, features, strategy_info)
                
            elif strategy_info['type'] == 'rl':
                price = self._get_rl_price(product_id, features, strategy_info)
                
            elif strategy_info['type'] == 'bandit':
                price = self._get_bandit_price(product_id, features, strategy_info)
                
            elif strategy_info['type'] == 'competitor':
                price = self._get_competitor_based_price(product_id, features, strategy_info)
                
            elif strategy_info['type'] == 'rules':
                price = self._get_rule_based_price(product_id, features, strategy_info)
            
            else:
                logger.error(f"Unsupported strategy type: {strategy_info['type']}")
                return None
            
            if price is None:
                logger.error(f"Failed to get price recommendation for product {product_id}")
                return None
            
            # Apply price bounds
            price = self._apply_price_bounds(product_id, price)
            
            # Apply psychological pricing
            psychological_price = self.psychological_price_func(price)
            
            # Create result with metadata
            result = {
                'product_id': product_id,
                'raw_price': price,
                'recommended_price': psychological_price,
                'strategy': strategy_name,
                'strategy_type': strategy_info['type'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features_used': features
            }
            
            logger.info(f"Price recommendation for product {product_id}: {psychological_price}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting price recommendation: {e}")
            return None
    
    def get_bulk_recommendations(self, product_ids, features_dict=None, strategy=None):
        """
        Get price recommendations for multiple products.
        
        Args:
            product_ids (list): List of product IDs
            features_dict (dict, optional): Dict mapping product IDs to feature dicts
            strategy (str, optional): Strategy name to use
            
        Returns:
            pd.DataFrame: DataFrame with price recommendations
        """
        try:
            features_dict = features_dict or {}
            results = []
            
            for product_id in product_ids:
                features = features_dict.get(product_id, {})
                recommendation = self.get_price_recommendation(product_id, features, strategy)
                
                if recommendation:
                    results.append(recommendation)
            
            if not results:
                logger.warning("No recommendations generated")
                return None
                
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error getting bulk recommendations: {e}")
            return None
    
    def _get_demand_based_price(self, product_id, features, strategy_info):
        """Get price based on demand forecasting model"""
        model = strategy_info['model']
        config = strategy_info['config']
        
        if not model:
            logger.error("No demand model available")
            return None
        
        try:
            # Calculate optimal price based on demand curve
            # This is a simplified calculation - in reality would use price elasticity
            base_price = config.get("base_price", 100)
            
            # Get demand at different price points
            price_range = np.linspace(
                config.get("min_price", base_price * 0.7),
                config.get("max_price", base_price * 1.3),
                config.get("price_points", 10)
            )
            
            revenues = []
            for price in price_range:
                # Create features with this price
                test_features = features.copy() if features else {}
                test_features['price'] = price
                
                # Predict demand at this price
                demand = model.predict(pd.DataFrame([test_features]))[0]
                
                # Calculate revenue
                revenue = price * demand
                revenues.append(revenue)
            
            # Get price that maximizes revenue
            optimal_idx = np.argmax(revenues)
            optimal_price = price_range[optimal_idx]
            
            return optimal_price
            
        except Exception as e:
            logger.error(f"Error in demand-based pricing: {e}")
            return None
    
    def _get_rl_price(self, product_id, features, strategy_info):
        """Get price from reinforcement learning model"""
        model = strategy_info['model']
        
        if not model:
            logger.error("No RL model available")
            return None
        
        try:
            # Convert features to the format expected by RL model
            context_features = {}
            if features:
                # Map features to context expected by RL model
                # This mapping should be customized based on RL model requirements
                for key, value in features.items():
                    if key in strategy_info['config'].get('feature_mapping', {}):
                        mapped_key = strategy_info['config']['feature_mapping'][key]
                        context_features[mapped_key] = value
                    else:
                        context_features[key] = value
            
            # Get optimal price from RL model
            optimal_price = model.get_optimal_price(context_features)
            return optimal_price
            
        except Exception as e:
            logger.error(f"Error in RL-based pricing: {e}")
            return None
    
    def _get_bandit_price(self, product_id, features, strategy_info):
        """Get price from multi-armed bandit model"""
        model = strategy_info['model']
        
        if not model:
            logger.error("No bandit model available")
            return None
        
        try:
            # Check if it's a contextual bandit
            if hasattr(model, 'context_dims') and features:
                # Prepare context vector
                context_vector = []
                
                # Map features to context expected by contextual bandit
                feature_order = strategy_info['config'].get('feature_order', [])
                
                if feature_order:
                    # Use predefined order
                    for feature_name in feature_order:
                        context_vector.append(features.get(feature_name, 0))
                else:
                    # Just use all features in arbitrary order
                    context_vector = list(features.values())
                
                # Get optimal price from contextual bandit
                optimal_price = model.get_optimal_price(context_vector)
                
            else:
                # Regular (non-contextual) bandit
                optimal_price = model.get_optimal_price()
                
            return optimal_price
            
        except Exception as e:
            logger.error(f"Error in bandit-based pricing: {e}")
            return None
    
    def _get_competitor_based_price(self, product_id, features, strategy_info):
        """Get price based on competitor prices"""
        config = strategy_info['config']
        
        try:
            # Check if competitor price is provided in features
            if features and 'competitor_price' in features:
                competitor_price = features['competitor_price']
                
                # Apply pricing strategy relative to competitor
                strategy = config.get("competitor_strategy", "match")
                
                if strategy == "match":
                    # Match competitor price
                    return competitor_price
                    
                elif strategy == "undercut":
                    # Undercut by percentage
                    undercut_pct = config.get("undercut_percentage", 5) / 100
                    return competitor_price * (1 - undercut_pct)
                    
                elif strategy == "premium":
                    # Premium pricing by percentage
                    premium_pct = config.get("premium_percentage", 10) / 100
                    return competitor_price * (1 + premium_pct)
                    
                elif strategy == "margin_based":
                    # Ensure minimum margin
                    if 'cost' in features:
                        min_margin_pct = config.get("min_margin_percentage", 20) / 100
                        cost = features['cost']
                        min_price = cost / (1 - min_margin_pct)
                        return max(min_price, competitor_price)
            
            logger.warning("Competitor price not available in features")
            return None
            
        except Exception as e:
            logger.error(f"Error in competitor-based pricing: {e}")
            return None
    
    def _get_rule_based_price(self, product_id, features, strategy_info):
        """Get price based on business rules"""
        config = strategy_info['config']
        rules = config.get("rules", [])
        
        try:
            # Start with base price
            base_price = config.get("base_price", 100)
            
            if features:
                if 'cost' in features:
                    # Calculate price based on cost and target margin
                    target_margin = config.get("target_margin", 0.4)
                    base_price = features['cost'] / (1 - target_margin)
            
            # Apply rules in sequence
            final_price = base_price
            
            for rule in rules:
                rule_type = rule.get("type")
                
                if rule_type == "markup":
                    # Apply percentage markup
                    markup_pct = rule.get("percentage", 0) / 100
                    final_price *= (1 + markup_pct)
                    
                elif rule_type == "discount":
                    # Apply discount if condition is met
                    condition = rule.get("condition")
                    if self._evaluate_condition(condition, features):
                        discount_pct = rule.get("percentage", 0) / 100
                        final_price *= (1 - discount_pct)
                        
                elif rule_type == "time_based":
                    # Time-based pricing rules (e.g., happy hour, seasonal)
                    current_time = datetime.now()
                    time_condition = rule.get("time_condition")
                    if self._evaluate_time_condition(time_condition, current_time):
                        adjustment_pct = rule.get("adjustment_percentage", 0) / 100
                        final_price *= (1 + adjustment_pct)
                
                elif rule_type == "volume_discount":
                    # Volume-based discount
                    if 'quantity' in features:
                        quantity = features['quantity']
                        tiers = rule.get("tiers", [])
                        
                        # Find applicable tier
                        for tier in sorted(tiers, key=lambda x: x.get("min_qty", 0), reverse=True):
                            if quantity >= tier.get("min_qty", 0):
                                discount_pct = tier.get("discount_percentage", 0) / 100
                                final_price *= (1 - discount_pct)
                                break
                                
            return final_price
            
        except Exception as e:
            logger.error(f"Error in rule-based pricing: {e}")
            return None
    
    def _evaluate_condition(self, condition, features):
        """
        Evaluate if a condition is met based on features.
        
        Args:
            condition (dict): Condition specification
            features (dict): Feature values
            
        Returns:
            bool: Whether condition is met
        """
        if not condition or not features:
            return False
            
        feature_name = condition.get("feature")
        operator = condition.get("operator", "==")
        value = condition.get("value")
        
        if feature_name not in features:
            return False
            
        feature_value = features[feature_name]
        
        # Evaluate condition
        if operator == "==":
            return feature_value == value
        elif operator == "!=":
            return feature_value != value
        elif operator == ">":
            return feature_value > value
        elif operator == ">=":
            return feature_value >= value
        elif operator == "<":
            return feature_value < value
        elif operator == "<=":
            return feature_value <= value
        elif operator == "in":
            return feature_value in value
        elif operator == "not_in":
            return feature_value not in value
            
        return False
    
    def _evaluate_time_condition(self, time_condition, current_time):
        """
        Evaluate if a time condition is met.
        
        Args:
            time_condition (dict): Time condition specification
            current_time (datetime): Current time
            
        Returns:
            bool: Whether time condition is met
        """
        if not time_condition:
            return False
            
        condition_type = time_condition.get("type")
        
        if condition_type == "day_of_week":
            days = time_condition.get("days", [])
            return current_time.weekday() in days
            
        elif condition_type == "hour_range":
            start_hour = time_condition.get("start_hour", 0)
            end_hour = time_condition.get("end_hour", 24)
            return start_hour <= current_time.hour < end_hour
            
        elif condition_type == "date_range":
            start_date_str = time_condition.get("start_date")
            end_date_str = time_condition.get("end_date")
            
            if start_date_str and end_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                return start_date <= current_time.date() <= end_date
                
        elif condition_type == "season":
            season = time_condition.get("season")
            current_month = current_time.month
            
            if season == "winter":
                return current_month in [12, 1, 2]
            elif season == "spring":
                return current_month in [3, 4, 5]
            elif season == "summer":
                return current_month in [6, 7, 8]
            elif season == "fall":
                return current_month in [9, 10, 11]
                
        return False
    
    def _apply_price_bounds(self, product_id, price):
        """
        Apply minimum and maximum price bounds.
        
        Args:
            product_id (str): Product ID
            price (float): Raw price
            
        Returns:
            float: Price within bounds
        """
        # Get product-specific bounds if available, otherwise use global bounds
        bounds = self.price_bounds.get(product_id, self.global_bounds)
        min_price, max_price = bounds
        
        # Apply bounds
        bounded_price = max(min_price, min(price, max_price))
        
        if bounded_price != price:
            logger.debug(f"Price {price} for product {product_id} adjusted to {bounded_price} due to bounds")
            
        return bounded_price
    
    def _default_psychological_price(self, price):
        """
        Apply psychological pricing (e.g., $9.99 instead of $10).
        
        Args:
            price (float): Raw price
            
        Returns:
            float: Psychological price
        """
        if price >= 1000:
            # For high prices, round to nearest 99
            return np.floor(price / 100) * 100 - 1
        elif price >= 100:
            # For medium prices, use .99
            return np.floor(price) - 0.01
        else:
            # For low prices, use .99 or .49 or .95
            return np.floor(price) - 0.01
    
    def update_model(self, strategy_name, new_model):
        """
        Update a model for a specific strategy.
        
        Args:
            strategy_name (str): Strategy name
            new_model (object): New model
            
        Returns:
            bool: Success status
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name]['model'] = new_model
            self.models[strategy_name] = new_model
            logger.info(f"Updated model for strategy: {strategy_name}")
            return True
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
            return False
    
    def set_price_bounds(self, product_id, min_price, max_price):
        """
        Set price bounds for a specific product.
        
        Args:
            product_id (str): Product ID
            min_price (float): Minimum price
            max_price (float): Maximum price
            
        Returns:
            bool: Success status
        """
        try:
            self.price_bounds[product_id] = (min_price, max_price)
            logger.info(f"Set price bounds for product {product_id}: [{min_price}, {max_price}]")
            return True
        except Exception as e:
            logger.error(f"Error setting price bounds: {e}")
            return False
    
    def set_global_bounds(self, min_price, max_price):
        """
        Set global price bounds.
        
        Args:
            min_price (float): Minimum price
            max_price (float): Maximum price
            
        Returns:
            bool: Success status
        """
        try:
            self.global_bounds = (min_price, max_price)
            logger.info(f"Set global price bounds: [{min_price}, {max_price}]")
            return True
        except Exception as e:
            logger.error(f"Error setting global bounds: {e}")
            return False
    
    def save(self, path):
        """
        Save the optimizer state.
        
        Args:
            path (str): Path to save the optimizer
            
        Returns:
            bool: Success status
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save models separately if they have their own save methods
            for name, model in self.models.items():
                if hasattr(model, 'save'):
                    model_path = f"{path}_{name}_model"
                    model.save(model_path)
                    
                    # Replace model with path for serialization
                    self.models[name] = model_path
                    self.strategies[name]['model'] = model_path
            
            # Save optimizer state
            state = {
                'strategies': self.strategies,
                'default_strategy': self.default_strategy,
                'price_bounds': self.price_bounds,
                'global_bounds': self.global_bounds,
                'config': self.config
            }
            
            with open(path, 'wb') as f:
                joblib.dump(state, f)
                
            logger.info(f"Price optimizer saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving price optimizer: {e}")
            return False
    
    @classmethod
    def load(cls, path):
        """
        Load a saved optimizer.
        
        Args:
            path (str): Path to the saved optimizer
            
        Returns:
            PriceOptimizer: Loaded optimizer instance
        """
        try:
            with open(path, 'rb') as f:
                state = joblib.load(f)
                
            # Create instance with config
            instance = cls(config=state['config'])
            
            # Restore state
            instance.strategies = state['strategies']
            instance.default_strategy = state['default_strategy']
            instance.price_bounds = state['price_bounds']
            instance.global_bounds = state['global_bounds']
            
            # Load models from paths
            for name, strategy_info in instance.strategies.items():
                model_path = strategy_info['model']
                if isinstance(model_path, str) and os.path.exists(f"{model_path}_metadata.pkl"):
                    # Try to load with appropriate method based on strategy type
                    if strategy_info['type'] == 'demand_based':
                        from .demand_forecasting import DemandForecaster
                        model = DemandForecaster.load_model(model_path)
                    elif strategy_info['type'] == 'rl':
                        from .reinforcement_agent import ReinforcementAgent
                        model = ReinforcementAgent.load(model_path)
                    elif strategy_info['type'] == 'bandit':
                        from .multiarmed_bandit import PricingBandit, ContextualPricingBandit
                        if os.path.exists(f"{model_path}.meta"):
                            model = PricingBandit.load(model_path)
                        else:
                            model = ContextualPricingBandit.load(model_path)
                    else:
                        model = joblib.load(model_path)
                        
                    instance.models[name] = model
                    instance.strategies[name]['model'] = model
            
            logger.info(f"Price optimizer loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading price optimizer: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    try:
        # Create price optimizer
        optimizer = PriceOptimizer(config={
            "default_strategy": "rule_based",
            "global_bounds": (10, 200)
        })
        
        # Add rule-based strategy
        rule_config = {
            "base_price": 100,
            "target_margin": 0.4,
            "rules": [
                {
                    "type": "time_based",
                    "time_condition": {
                        "type": "day_of_week",
                        "days": [5, 6]  # Weekend (Saturday, Sunday)
                    },
                    "adjustment_percentage": 10  # 10% premium on weekends
                },
                {
                    "type": "discount",
                    "condition": {
                        "feature": "customer_segment",
                        "operator": "==",
                        "value": "loyal"
                    },
                    "percentage": 5  # 5% discount for loyal customers
                }
            ]
        }
        
        optimizer.add_strategy("rule_based", "rules", config=rule_config)
        
        # Set price bounds for specific products
        optimizer.set_price_bounds("product123", 50, 150)
        
        # Get price recommendation
        features = {
            "cost": 60,
            "customer_segment": "loyal",
            "quantity": 5
        }
        
        recommendation = optimizer.get_price_recommendation("product123", features)
        print("Price Recommendation:", recommendation)
        
        # Get bulk recommendations
        product_ids = ["product123", "product456"]
        features_dict = {
            "product123": features,
            "product456": {"cost": 75, "customer_segment": "new", "quantity": 1}
        }
        
        bulk_recommendations = optimizer.get_bulk_recommendations(product_ids, features_dict)
        print("\nBulk Recommendations:")
        print(bulk_recommendations)
        
    except Exception as e:
        print(f"Error in example: {e}")
