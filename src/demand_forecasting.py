"""
Demand Forecasting module for SmartDynamic pricing system.
This module provides time-series forecasting models for product demand.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
from datetime import datetime, timedelta
from loguru import logger


class DemandForecaster:
    """
    Demand forecasting class that implements multiple time series forecasting methods.
    Supports Prophet, LSTM, XGBoost, and ARIMA models.
    """
    
    def __init__(self, model_type="prophet", config=None):
        """
        Initialize the demand forecaster with specified model type.
        
        Args:
            model_type (str): Type of forecasting model ('prophet', 'lstm', 'xgboost', 'arima')
            config (dict, optional): Model-specific configuration parameters
        """
        self.model_type = model_type.lower()
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = self.config.get("sequence_length", 7)  # For LSTM
        self.forecast_horizon = self.config.get("forecast_horizon", 7)  # Default forecast days
        self.features = []
        
        logger.info(f"Initialized {self.model_type} demand forecaster")
        
    def _prepare_prophet_data(self, df, date_col, target_col, features=None):
        """Prepare data for Prophet model"""
        # Prophet requires 'ds' (date) and 'y' (target) columns
        prophet_df = df.rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Add any additional regressor columns if specified
        if features:
            self.features = [f for f in features if f in df.columns]
        
        return prophet_df
    
    def _prepare_lstm_data(self, df, date_col, target_col, features=None):
        """Prepare sequences for LSTM model"""
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Identify features to use
        if features:
            self.features = [f for f in features if f in df.columns]
        else:
            self.features = [target_col]
        
        # Scale features
        data = df[self.features].values
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i+self.sequence_length])
            y.append(scaled_data[i+self.sequence_length, df.columns.get_loc(target_col)])
        
        return np.array(X), np.array(y)
    
    def _prepare_xgboost_data(self, df, date_col, target_col, features=None):
        """Prepare data for XGBoost model"""
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Identify features to use
        if features:
            self.features = [f for f in features if f in df.columns]
        else:
            self.features = [c for c in df.columns if c != target_col and c != date_col]
        
        X = df[self.features].values
        y = df[target_col].values
        
        return X, y
    
    def _build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def fit(self, df, date_col, target_col, features=None):
        """
        Fit the forecasting model on historical data.
        
        Args:
            df (pd.DataFrame): Input DataFrame with historical data
            date_col (str): Name of date column
            target_col (str): Name of target column to forecast
            features (list, optional): Additional features for the model
            
        Returns:
            self: Fitted model
        """
        try:
            if self.model_type == "prophet":
                prophet_df = self._prepare_prophet_data(df, date_col, target_col, features)
                
                # Initialize and fit Prophet model
                model = Prophet(
                    yearly_seasonality=self.config.get("yearly_seasonality", True),
                    weekly_seasonality=self.config.get("weekly_seasonality", True),
                    daily_seasonality=self.config.get("daily_seasonality", False),
                    seasonality_mode=self.config.get("seasonality_mode", "additive"),
                    interval_width=self.config.get("interval_width", 0.95)
                )
                
                # Add additional regressors
                for feature in self.features:
                    model.add_regressor(feature)
                
                # Fit model
                model.fit(prophet_df)
                self.model = model
                
            elif self.model_type == "lstm":
                X, y = self._prepare_lstm_data(df, date_col, target_col, features)
                
                # Build LSTM model
                self.model = self._build_lstm_model((X.shape[1], X.shape[2]))
                
                # Define early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get("patience", 5),
                    restore_best_weights=True
                )
                
                # Train model
                self.model.fit(
                    X, y,
                    epochs=self.config.get("epochs", 50),
                    batch_size=self.config.get("batch_size", 32),
                    validation_split=self.config.get("validation_split", 0.2),
                    callbacks=[early_stopping],
                    verbose=self.config.get("verbose", 0)
                )
                
            elif self.model_type == "xgboost":
                X, y = self._prepare_xgboost_data(df, date_col, target_col, features)
                
                # Configure XGBoost model
                params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': self.config.get("learning_rate", 0.1),
                    'max_depth': self.config.get("max_depth", 6),
                    'min_child_weight': self.config.get("min_child_weight", 1),
                    'subsample': self.config.get("subsample", 0.8),
                    'colsample_bytree': self.config.get("colsample_bytree", 0.8),
                    'n_estimators': self.config.get("n_estimators", 100)
                }
                
                self.model = xgb.XGBRegressor(**params)
                self.model.fit(X, y)
                
            elif self.model_type == "arima":
                # Sort by date
                df = df.sort_values(by=date_col)
                
                # ARIMA requires just the target series
                series = df[target_col]
                
                # Extract ARIMA parameters
                p = self.config.get("p", 1)
                d = self.config.get("d", 1)
                q = self.config.get("q", 0)
                
                # Fit ARIMA model
                self.model = ARIMA(series, order=(p, d, q))
                self.model = self.model.fit()
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return None
            
            logger.info(f"Successfully fitted {self.model_type} model")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting {self.model_type} model: {e}")
            raise
    
    def predict(self, periods=None, future_features=None):
        """
        Generate demand forecasts for future periods.
        
        Args:
            periods (int, optional): Number of periods to forecast
            future_features (pd.DataFrame, optional): DataFrame with future feature values
            
        Returns:
            pd.DataFrame: DataFrame with forecast values
        """
        if periods is None:
            periods = self.forecast_horizon
            
        try:
            if self.model_type == "prophet":
                # Create future DataFrame
                future = self.model.make_future_dataframe(periods=periods, freq='D')
                
                # Add regressor values if provided
                if future_features is not None and len(self.features) > 0:
                    for feature in self.features:
                        if feature in future_features.columns:
                            future[feature] = future_features[feature]
                
                # Make prediction
                forecast = self.model.predict(future)
                return forecast
                
            elif self.model_type == "lstm":
                # For LSTM, we need the last sequence from our data
                # This would need to be implemented with the actual last sequence
                # and feature engineering for future dates
                logger.warning("LSTM forecast requires last sequence data and future features")
                return None
                
            elif self.model_type == "xgboost":
                # For XGBoost, we need feature values for future periods
                if future_features is None:
                    logger.warning("XGBoost forecast requires future feature values")
                    return None
                    
                # Prepare future features
                future_X = future_features[self.features].values
                
                # Make prediction
                predictions = self.model.predict(future_X)
                
                # Create forecast DataFrame
                forecast = future_features.copy()
                forecast['forecast'] = predictions
                return forecast
                
            elif self.model_type == "arima":
                # Generate forecasts with ARIMA
                forecast = self.model.forecast(steps=periods)
                return forecast
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating forecast with {self.model_type}: {e}")
            return None
    
    def evaluate(self, test_df, date_col, target_col, features=None):
        """
        Evaluate model performance on test data.
        
        Args:
            test_df (pd.DataFrame): Test DataFrame
            date_col (str): Name of date column
            target_col (str): Name of target column
            features (list, optional): Feature columns
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            if self.model_type == "prophet":
                # Prepare test data in Prophet format
                prophet_df = self._prepare_prophet_data(test_df, date_col, target_col, features)
                
                # Generate predictions for test dates
                forecast = self.predict(periods=0)
                
                # Merge with actual values
                evaluation = pd.merge(
                    prophet_df[['ds', 'y']], 
                    forecast[['ds', 'yhat']], 
                    on='ds'
                )
                
                # Calculate metrics
                mae = mean_absolute_error(evaluation['y'], evaluation['yhat'])
                rmse = np.sqrt(mean_squared_error(evaluation['y'], evaluation['yhat']))
                
            elif self.model_type == "lstm":
                # Prepare test sequences
                X_test, y_test = self._prepare_lstm_data(test_df, date_col, target_col, features)
                
                # Generate predictions
                predictions = self.model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                
            elif self.model_type == "xgboost":
                # Prepare test data
                X_test, y_test = self._prepare_xgboost_data(test_df, date_col, target_col, features)
                
                # Generate predictions
                predictions = self.model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                
            elif self.model_type == "arima":
                # For ARIMA, we compare forecasts with actual values
                # This would need actual implementation with real data
                logger.warning("ARIMA evaluation not fully implemented")
                return None
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return None
                
            metrics = {
                'mae': mae,
                'rmse': rmse
            }
            
            logger.info(f"Model evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.model_type} model: {e}")
            return None
    
    def plot_forecast(self, forecast=None, actual_df=None, date_col=None, target_col=None, 
                      title=None, figsize=(12, 6)):
        """
        Plot the forecasted values along with actual values if provided.
        
        Args:
            forecast (pd.DataFrame, optional): Forecast DataFrame
            actual_df (pd.DataFrame, optional): DataFrame with actual values
            date_col (str, optional): Name of date column in actual_df
            target_col (str, optional): Name of target column in actual_df
            title (str, optional): Plot title
            figsize (tuple, optional): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            fig = plt.figure(figsize=figsize)
            
            if self.model_type == "prophet":
                if forecast is None:
                    logger.warning("No forecast provided for plotting")
                    return fig
                    
                # Plot Prophet forecast
                ax = fig.add_subplot(111)
                
                # Plot forecast
                ax.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                                color='blue', alpha=0.2, label='95% Confidence Interval')
                
                # Plot actual values if provided
                if actual_df is not None and date_col and target_col:
                    ax.plot(actual_df[date_col], actual_df[target_col], 
                            color='red', marker='o', linestyle='None', label='Actual')
                
            else:
                # Generic plotting for other model types
                ax = fig.add_subplot(111)
                
                # Plot forecast if provided
                if forecast is not None:
                    if isinstance(forecast, pd.DataFrame) and 'ds' in forecast.columns and 'yhat' in forecast.columns:
                        ax.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
                    elif isinstance(forecast, pd.Series):
                        ax.plot(forecast.index, forecast.values, color='blue', label='Forecast')
                    else:
                        ax.plot(range(len(forecast)), forecast, color='blue', label='Forecast')
                
                # Plot actual values if provided
                if actual_df is not None and date_col and target_col:
                    ax.plot(actual_df[date_col], actual_df[target_col], 
                            color='red', marker='o', label='Actual')
            
            # Set title and labels
            ax.set_title(title or f"{self.model_type.capitalize()} Demand Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Demand")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            return None
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: Success status
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if self.model_type == "prophet":
                with open(path, 'wb') as f:
                    joblib.dump(self.model, f)
                    
            elif self.model_type in ["lstm", "xgboost"]:
                self.model.save(path)
                
            elif self.model_type == "arima":
                with open(path, 'wb') as f:
                    joblib.dump(self.model, f)
                    
            else:
                logger.error(f"Saving not implemented for model type: {self.model_type}")
                return False
                
            # Save model metadata
            metadata = {
                'model_type': self.model_type,
                'features': self.features,
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon,
                'date_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(f"{path}.meta", 'wb') as f:
                joblib.dump(metadata, f)
                
            logger.info(f"Model successfully saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load_model(cls, path):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            DemandForecaster: Loaded model instance
        """
        try:
            # Load metadata
            with open(f"{path}.meta", 'rb') as f:
                metadata = joblib.load(f)
                
            # Create instance with correct model type
            instance = cls(model_type=metadata['model_type'])
            instance.features = metadata['features']
            instance.sequence_length = metadata.get('sequence_length', 7)
            instance.forecast_horizon = metadata.get('forecast_horizon', 7)
            
            # Load the model based on type
            if metadata['model_type'] == "prophet":
                with open(path, 'rb') as f:
                    instance.model = joblib.load(f)
                    
            elif metadata['model_type'] == "lstm":
                from tensorflow.keras.models import load_model
                instance.model = load_model(path)
                
            elif metadata['model_type'] == "xgboost":
                instance.model = xgb.XGBRegressor()
                instance.model.load_model(path)
                
            elif metadata['model_type'] == "arima":
                with open(path, 'rb') as f:
                    instance.model = joblib.load(f)
                    
            else:
                logger.error(f"Loading not implemented for model type: {metadata['model_type']}")
                return None
                
            logger.info(f"Model successfully loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    try:
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=100)
        data = {
            'date': dates,
            'demand': np.sin(np.arange(100)/10) * 50 + 100 + np.random.normal(0, 10, 100),
            'price': np.random.uniform(80, 120, 100),
            'promo': np.random.choice([0, 1], 100, p=[0.8, 0.2])
        }
        df = pd.DataFrame(data)
        
        # Initialize forecaster
        forecaster = DemandForecaster(model_type="prophet", config={
            "yearly_seasonality": False,
            "weekly_seasonality": True
        })
        
        # Fit model
        forecaster.fit(df, date_col='date', target_col='demand', features=['price', 'promo'])
        
        # Generate forecast
        forecast = forecaster.predict(periods=30)
        
        print(f"Generated forecast for next {30} days")
        
        # Plot forecast
        fig = forecaster.plot_forecast(forecast=forecast, actual_df=df, 
                                       date_col='date', target_col='demand',
                                       title="Example Prophet Forecast")
        plt.show()
        
    except Exception as e:
        print(f"Error in example: {e}")
