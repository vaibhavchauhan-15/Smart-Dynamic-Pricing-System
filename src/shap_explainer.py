"""
SHAP Explainer module for SmartDynamic pricing system.
This module implements explainability for pricing decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os
from loguru import logger


class ShapExplainer:
    """
    SHAP-based explanation generator for pricing models.
    Provides transparency and insights into pricing decisions.
    """
    
    def __init__(self, model=None, model_type=None, feature_names=None):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: The model to explain
            model_type (str, optional): Type of model ('tree', 'linear', 'kernel', 'deep', etc.)
            feature_names (list, optional): Names of features
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
        logger.info("SHAP Explainer initialized")
    
    def set_model(self, model, model_type=None):
        """
        Set the model to explain.
        
        Args:
            model: The model to explain
            model_type (str, optional): Type of model
            
        Returns:
            bool: Success status
        """
        try:
            self.model = model
            if model_type:
                self.model_type = model_type
            
            # Reset explainer since model changed
            self.explainer = None
            self.shap_values = None
            
            logger.info("Model set for explanation")
            return True
            
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            return False
    
    def set_feature_names(self, feature_names):
        """
        Set feature names for explanations.
        
        Args:
            feature_names (list): Names of features
            
        Returns:
            bool: Success status
        """
        try:
            self.feature_names = feature_names
            logger.info(f"Set {len(feature_names)} feature names")
            return True
        except Exception as e:
            logger.error(f"Error setting feature names: {e}")
            return False
    
    def fit(self, background_data):
        """
        Fit the SHAP explainer with background data.
        
        Args:
            background_data (pd.DataFrame): Background data for SHAP
            
        Returns:
            self: Fitted explainer
        """
        try:
            # Store the background data
            self.background_data = background_data
            
            # Convert feature names if needed
            if self.feature_names and len(self.feature_names) == background_data.shape[1]:
                if isinstance(background_data, pd.DataFrame):
                    background_data.columns = self.feature_names
                
            # Create explainer based on model type
            if not self.model_type and hasattr(self.model, 'predict_proba'):
                self.model_type = 'tree'  # Default to tree for scikit-learn models
                
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model, background_data)
                
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, background_data)
                
            elif self.model_type == 'kernel':
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
                
            elif self.model_type == 'deep':
                self.explainer = shap.DeepExplainer(self.model, background_data)
                
            elif self.model_type == 'gradient':
                self.explainer = shap.GradientExplainer(self.model, background_data)
                
            else:
                # Default to Kernel Explainer which works with any model
                predict_function = self.model.predict if hasattr(self.model, 'predict') else self.model
                self.explainer = shap.KernelExplainer(predict_function, background_data)
            
            logger.info(f"SHAP explainer fitted with {len(background_data)} background samples")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting SHAP explainer: {e}")
            return None
    
    def explain(self, data):
        """
        Generate SHAP explanations for data.
        
        Args:
            data (pd.DataFrame): Data to explain
            
        Returns:
            dict: Dictionary with SHAP values and related information
        """
        try:
            if not self.explainer:
                logger.error("Explainer not fitted. Call fit() first.")
                return None
            
            # Convert feature names if needed
            if self.feature_names and len(self.feature_names) == data.shape[1]:
                if isinstance(data, pd.DataFrame):
                    data.columns = self.feature_names
            
            # Generate SHAP values
            self.shap_values = self.explainer.shap_values(data)
            
            # Prepare explanation result
            result = {
                'shap_values': self.shap_values,
                'expected_value': self.explainer.expected_value,
                'data': data,
                'feature_names': self.feature_names or list(data.columns) if isinstance(data, pd.DataFrame) else None
            }
            
            if isinstance(self.shap_values, list):
                # For multi-output models, we have a list of SHAP values arrays
                result['num_outputs'] = len(self.shap_values)
            
            logger.info(f"Generated SHAP explanations for {len(data)} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return None
    
    def explain_price_recommendation(self, price_optimizer, product_id, features, strategy=None):
        """
        Explain a specific price recommendation.
        
        Args:
            price_optimizer: Price optimizer object
            product_id (str): Product ID
            features (dict): Features for pricing decision
            strategy (str, optional): Strategy name
            
        Returns:
            dict: Explanation of price recommendation
        """
        try:
            # Get the pricing recommendation
            recommendation = price_optimizer.get_price_recommendation(product_id, features, strategy)
            
            if not recommendation:
                logger.error("Failed to get price recommendation")
                return None
            
            # Get the model and strategy info
            strategy_name = recommendation['strategy']
            strategy_type = recommendation['strategy_type']
            
            # Check if this strategy has an explainable model
            if strategy_name not in price_optimizer.models or strategy_type not in ['demand_based', 'rl', 'bandit']:
                logger.warning(f"Strategy {strategy_name} does not have an explainable model")
                
                # Return basic explanation
                return {
                    'recommendation': recommendation,
                    'explanation_type': 'basic',
                    'factors': features,
                    'message': f"Price determined using {strategy_type} strategy."
                }
            
            # Get the model
            model = price_optimizer.models[strategy_name]
            
            # Set up explainer for this model if needed
            if self.model != model:
                self.set_model(model, strategy_type)
                
                # Need background data to fit explainer
                if self.background_data is None:
                    logger.warning("No background data available for SHAP explanation")
                    return {
                        'recommendation': recommendation,
                        'explanation_type': 'basic',
                        'factors': features,
                        'message': f"Price determined using {strategy_type} strategy, but no SHAP explanations available."
                    }
                
                self.fit(self.background_data)
            
            # Create a DataFrame with the features
            features_df = pd.DataFrame([features])
            
            # Generate SHAP explanations
            shap_result = self.explain(features_df)
            
            if not shap_result:
                logger.warning("Failed to generate SHAP explanations")
                return {
                    'recommendation': recommendation,
                    'explanation_type': 'basic',
                    'factors': features,
                    'message': f"Price determined using {strategy_type} strategy, but SHAP explanation failed."
                }
            
            # Extract the SHAP values for this prediction
            shap_values = shap_result['shap_values']
            
            if isinstance(shap_values, list):
                # For multi-output models, use the first output (typically price)
                shap_values = shap_values[0]
            
            # Get feature importances
            feature_importance = {}
            for i, feature in enumerate(shap_result['feature_names']):
                feature_importance[feature] = float(abs(shap_values[0][i]))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top contributing factors
            top_factors = []
            for feature, importance in sorted_features:
                direction = "+" if shap_values[0][shap_result['feature_names'].index(feature)] > 0 else "-"
                value = features.get(feature, "N/A")
                top_factors.append({
                    'feature': feature,
                    'importance': importance,
                    'direction': direction,
                    'value': value
                })
            
            # Calculate base value (expected price without features)
            base_value = float(shap_result['expected_value']) if not isinstance(shap_result['expected_value'], list) else float(shap_result['expected_value'][0])
            
            # Create explanation
            explanation = {
                'recommendation': recommendation,
                'explanation_type': 'shap',
                'base_value': base_value,
                'top_factors': top_factors[:5],  # Top 5 factors
                'feature_importance': dict(sorted_features),
                'shap_values': shap_values[0].tolist(),
                'expected_value': float(shap_result['expected_value']) if not isinstance(shap_result['expected_value'], list) else float(shap_result['expected_value'][0])
            }
            
            logger.info(f"Generated SHAP explanation for {product_id}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining price recommendation: {e}")
            return None
    
    def plot_summary(self, data=None, plot_type="bar", max_display=10):
        """
        Plot SHAP summary visualization.
        
        Args:
            data (pd.DataFrame, optional): Data to explain (uses previously explained data if None)
            plot_type (str): Type of plot ('bar', 'beeswarm', 'heatmap')
            max_display (int): Maximum number of features to display
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Generate SHAP values if data is provided
            if data is not None:
                explanation = self.explain(data)
                shap_values = explanation['shap_values']
                feature_names = explanation['feature_names']
            else:
                if self.shap_values is None:
                    logger.error("No SHAP values available. Call explain() first.")
                    return None
                shap_values = self.shap_values
                feature_names = self.feature_names
            
            # Check if multi-output and take first output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Create appropriate plot
            if plot_type == "bar":
                shap.summary_plot(shap_values, data if data is not None else self.background_data, 
                                  feature_names=feature_names, plot_type="bar", max_display=max_display)
                
            elif plot_type == "beeswarm":
                shap.summary_plot(shap_values, data if data is not None else self.background_data, 
                                  feature_names=feature_names, max_display=max_display)
                
            elif plot_type == "heatmap":
                if hasattr(shap, "plots"):
                    shap.plots.heatmap(shap_values, max_display=max_display)
                else:
                    logger.warning("Heatmap plot requires newer SHAP version. Using beeswarm plot instead.")
                    shap.summary_plot(shap_values, data if data is not None else self.background_data, 
                                      feature_names=feature_names, max_display=max_display)
            
            fig = plt.gcf()
            plt.tight_layout()
            
            logger.info(f"Created {plot_type} SHAP summary plot")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP plot: {e}")
            return None
    
    def plot_dependence(self, feature, interaction_feature="auto", data=None):
        """
        Plot SHAP dependence plot for a feature.
        
        Args:
            feature (str): Feature to plot
            interaction_feature (str): Feature for interaction coloring
            data (pd.DataFrame, optional): Data to explain
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Generate SHAP values if data is provided
            if data is not None:
                explanation = self.explain(data)
                shap_values = explanation['shap_values']
                feature_names = explanation['feature_names']
            else:
                if self.shap_values is None:
                    logger.error("No SHAP values available. Call explain() first.")
                    return None
                shap_values = self.shap_values
                feature_names = self.feature_names
            
            # Check if multi-output and take first output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            # Get feature index
            if feature_names is not None:
                if feature in feature_names:
                    feature_idx = feature_names.index(feature)
                else:
                    logger.error(f"Feature {feature} not found in feature names")
                    return None
            else:
                try:
                    feature_idx = int(feature)
                except:
                    logger.error("Feature must be index when feature_names not available")
                    return None
                    
            # Get interaction feature index
            if interaction_feature != "auto" and feature_names is not None:
                if interaction_feature in feature_names:
                    interaction_idx = feature_names.index(interaction_feature)
                else:
                    logger.error(f"Interaction feature {interaction_feature} not found")
                    interaction_feature = "auto"
            
            # Create dependence plot
            x_data = data if data is not None else self.background_data
            
            if interaction_feature == "auto":
                shap.dependence_plot(feature_idx, shap_values, x_data, feature_names=feature_names)
            else:
                interaction_idx = feature_names.index(interaction_feature) if feature_names else int(interaction_feature)
                shap.dependence_plot(feature_idx, shap_values, x_data, feature_names=feature_names, interaction_index=interaction_idx)
            
            fig = plt.gcf()
            plt.tight_layout()
            
            logger.info(f"Created SHAP dependence plot for {feature}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dependence plot: {e}")
            return None
    
    def plot_waterfall(self, instance_idx=0, data=None):
        """
        Plot SHAP waterfall plot for a specific instance.
        
        Args:
            instance_idx (int): Index of instance to explain
            data (pd.DataFrame, optional): Data to explain
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Generate SHAP values if data is provided
            if data is not None:
                explanation = self.explain(data)
                shap_values = explanation['shap_values']
                expected_value = explanation['expected_value']
                feature_names = explanation['feature_names']
            else:
                if self.shap_values is None:
                    logger.error("No SHAP values available. Call explain() first.")
                    return None
                shap_values = self.shap_values
                expected_value = self.explainer.expected_value
                feature_names = self.feature_names
            
            # Check if multi-output and take first output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                expected_value = expected_value[0] if isinstance(expected_value, list) else expected_value
            
            # Create waterfall plot
            x_data = data if data is not None else self.background_data
            
            if hasattr(shap, "plots"):
                # New SHAP version
                shap.plots.waterfall(shap.Explanation(values=shap_values[instance_idx], 
                                                   base_values=expected_value, 
                                                   data=x_data.iloc[instance_idx] if isinstance(x_data, pd.DataFrame) else x_data[instance_idx],
                                                   feature_names=feature_names))
            else:
                # Legacy SHAP version
                shap.force_plot(expected_value, shap_values[instance_idx], 
                               x_data.iloc[instance_idx] if isinstance(x_data, pd.DataFrame) else x_data[instance_idx],
                               feature_names=feature_names, matplotlib=True)
            
            fig = plt.gcf()
            plt.tight_layout()
            
            logger.info(f"Created SHAP waterfall plot for instance {instance_idx}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
            return None
    
    def save(self, path):
        """
        Save the explainer.
        
        Args:
            path (str): Path to save the explainer
            
        Returns:
            bool: Success status
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save explainer state (except the model)
            state = {
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            
            # Save background data if available
            if self.background_data is not None:
                bg_data_path = f"{path}_background_data"
                joblib.dump(self.background_data, bg_data_path)
                state['background_data_path'] = bg_data_path
            
            # Save SHAP values if available
            if self.shap_values is not None:
                shap_values_path = f"{path}_shap_values"
                joblib.dump(self.shap_values, shap_values_path)
                state['shap_values_path'] = shap_values_path
            
            # Save state
            with open(path, 'wb') as f:
                joblib.dump(state, f)
                
            logger.info(f"SHAP explainer saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving explainer: {e}")
            return False
    
    @classmethod
    def load(cls, path, model=None):
        """
        Load a saved explainer.
        
        Args:
            path (str): Path to the saved explainer
            model (optional): Model for the explainer
            
        Returns:
            ShapExplainer: Loaded explainer instance
        """
        try:
            with open(path, 'rb') as f:
                state = joblib.load(f)
                
            # Create instance
            instance = cls(model=model, model_type=state['model_type'], feature_names=state['feature_names'])
            
            # Load background data if available
            if 'background_data_path' in state and os.path.exists(state['background_data_path']):
                instance.background_data = joblib.load(state['background_data_path'])
                
                # Try to fit the explainer if model is provided
                if model is not None:
                    instance.fit(instance.background_data)
            
            # Load SHAP values if available
            if 'shap_values_path' in state and os.path.exists(state['shap_values_path']):
                instance.shap_values = joblib.load(state['shap_values_path'])
            
            logger.info(f"SHAP explainer loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading explainer: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    try:
        import xgboost as xgb
        from sklearn.datasets import make_regression
        
        # Generate synthetic data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        df = pd.DataFrame(X, columns=feature_names)
        
        # Train a simple model
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3)
        model.fit(X, y)
        
        # Create explainer
        explainer = ShapExplainer(model=model, model_type='tree', feature_names=feature_names)
        
        # Fit explainer
        explainer.fit(df)
        
        # Generate explanations
        explanation = explainer.explain(df.iloc[:10])  # Explain first 10 instances
        
        # Plot summary
        fig1 = explainer.plot_summary(max_display=5)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()
        
        # Plot dependence for the most important feature
        fig2 = explainer.plot_dependence('feature_1', 'feature_2')
        plt.title("Dependence Plot")
        plt.tight_layout()
        plt.show()
        
        # Plot waterfall for first instance
        fig3 = explainer.plot_waterfall(instance_idx=0)
        plt.title("Explanation for Single Prediction")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in example: {e}")
