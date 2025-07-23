"""
Feature Engineering module for SmartDynamic pricing system.
This module handles all feature creation, transformation, and encoding operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import category_encoders as ce
from scipy import stats
from loguru import logger
import holidays


class FeatureEngineering:
    """
    Feature Engineering class for creating advanced features for pricing models.
    """
    
    def __init__(self):
        """Initialize feature engineering pipeline components."""
        self.scalers = {}
        self.encoders = {}
        self.country_holidays = holidays.US()  # Default to US holidays - can be changed
        logger.info("Feature Engineering module initialized")
        
    def fit_transform(self, df, categorical_cols=None, numerical_cols=None):
        """
        Fit and transform the DataFrame with feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            categorical_cols (list): Categorical columns to encode
            numerical_cols (list): Numerical columns to scale
            
        Returns:
            pd.DataFrame: Transformed DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add time-based features
        if 'date' in df.columns:
            result_df = self.add_time_features(result_df, 'date')
            
        # Add price-related features
        if all(col in df.columns for col in ['price', 'original_price']):
            result_df = self.add_price_features(result_df)
            
        # Process categorical features
        if categorical_cols:
            result_df = self.encode_categorical_features(result_df, categorical_cols)
            
        # Scale numerical features
        if numerical_cols:
            result_df = self.scale_numerical_features(result_df, numerical_cols)
        
        return result_df
    
    def add_time_features(self, df, date_col):
        """
        Extract time-based features from date column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_col (str): Name of the date column
            
        Returns:
            pd.DataFrame: DataFrame with additional time features
        """
        df = df.copy()
        
        # Convert date column to datetime if not already
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract basic time components
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['quarter'] = df[date_col].dt.quarter
        
        # Add cyclical encoding for month and day of week
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        # Check for holidays
        df['is_holiday'] = df[date_col].apply(lambda x: 1 if x in self.country_holidays else 0)
        
        # Add proximity to special events (examples: Black Friday, Christmas, etc.)
        black_friday = pd.to_datetime(f"{df[date_col].dt.year.iloc[0]}-11-26")  # Approximate
        christmas = pd.to_datetime(f"{df[date_col].dt.year.iloc[0]}-12-25")
        
        df['days_to_black_friday'] = df[date_col].apply(lambda x: 
            min(abs((x - black_friday).days), 60))  # Cap at 60 days
        df['days_to_christmas'] = df[date_col].apply(lambda x: 
            min(abs((x - christmas).days), 60))  # Cap at 60 days
        
        # Normalize these proximity features
        df['days_to_black_friday'] = 1 - (df['days_to_black_friday'] / 60)
        df['days_to_christmas'] = 1 - (df['days_to_christmas'] / 60)
        
        logger.info(f"Added time features based on {date_col}")
        return df
    
    def add_price_features(self, df):
        """
        Create price-related features.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price columns
            
        Returns:
            pd.DataFrame: DataFrame with additional price features
        """
        df = df.copy()
        
        # Calculate discount features
        if all(col in df.columns for col in ['price', 'original_price']):
            df['discount_amount'] = df['original_price'] - df['price']
            df['discount_percentage'] = (df['discount_amount'] / df['original_price'] * 100).round(2)
            
            # Flag for psychological pricing
            df['is_psychological_price'] = df['price'].apply(
                lambda x: 1 if str(int(x))[-1] == '9' else 0
            )
            
            # Price positioning relative to product category
            if 'category' in df.columns:
                category_avg = df.groupby('category')['price'].transform('mean')
                category_std = df.groupby('category')['price'].transform('std')
                df['price_position'] = (df['price'] - category_avg) / category_std
        
        logger.info("Added price-related features")
        return df
    
    def encode_categorical_features(self, df, categorical_cols):
        """
        Encode categorical features using appropriate encoding schemes.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            categorical_cols (list): List of categorical columns
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        df = df.copy()
        
        # Filter existing columns only
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # Select encoding method based on cardinality
        for col in categorical_cols:
            if col not in self.encoders:
                unique_count = df[col].nunique()
                
                if unique_count <= 2:  # Binary features
                    self.encoders[col] = {"type": "label", "encoder": ce.BinaryEncoder(cols=[col])}
                elif unique_count <= 15:  # Low cardinality
                    self.encoders[col] = {"type": "onehot", "encoder": OneHotEncoder(sparse=False, handle_unknown='ignore')}
                else:  # High cardinality
                    self.encoders[col] = {"type": "target", "encoder": ce.TargetEncoder(cols=[col])}
                    
        # Apply encodings
        for col, encoder_info in self.encoders.items():
            if col not in df.columns:
                continue
                
            encoder = encoder_info["encoder"]
            encoder_type = encoder_info["type"]
            
            if encoder_type == "onehot":
                # Fit if not already fitted
                if not hasattr(encoder, 'categories_'):
                    encoder.fit(df[[col]])
                
                # Transform
                encoded_array = encoder.transform(df[[col]])
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
                
                # Join with original dataframe
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:
                # Target and other encoders work differently
                if col in df.columns:
                    if 'target' in df.columns:
                        encoded_df = encoder.fit_transform(df[col], df['target'])
                    else:
                        encoded_df = encoder.fit_transform(df[col])
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df
    
    def scale_numerical_features(self, df, numerical_cols):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_cols (list): List of numerical columns
            
        Returns:
            pd.DataFrame: DataFrame with scaled numerical features
        """
        df = df.copy()
        
        # Filter existing columns only
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        # Create a combined scaler for all numerical features
        if 'numerical' not in self.scalers:
            self.scalers['numerical'] = StandardScaler()
            
        # Extract numerical data for scaling
        numerical_data = df[numerical_cols].copy()
        
        # Handle missing values
        numerical_data.fillna(numerical_data.median(), inplace=True)
        
        # Fit and transform
        if not hasattr(self.scalers['numerical'], 'mean_'):
            scaled_data = self.scalers['numerical'].fit_transform(numerical_data)
        else:
            scaled_data = self.scalers['numerical'].transform(numerical_data)
        
        # Replace original columns with scaled versions
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols, index=df.index)
        df[numerical_cols] = scaled_df
        
        logger.info(f"Scaled {len(numerical_cols)} numerical features")
        return df
    
    def add_interaction_features(self, df, feature_pairs):
        """
        Create interaction features between specified pairs of features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_pairs (list): List of feature pairs to create interactions
            
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        df = df.copy()
        
        for pair in feature_pairs:
            if len(pair) != 2:
                continue
                
            feat1, feat2 = pair
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df[interaction_name] = df[feat1] * df[feat2]
                
        logger.info(f"Added {len(feature_pairs)} interaction features")
        return df
    
    def add_lag_features(self, df, time_col, entity_col, value_cols, lags):
        """
        Create lag features for time series data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            time_col (str): Time column name
            entity_col (str): Entity column name (e.g., product_id)
            value_cols (list): Value columns to create lags for
            lags (list): List of lag periods
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        df = df.copy().sort_values([entity_col, time_col])
        
        for col in value_cols:
            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}"
                df[lag_col_name] = df.groupby(entity_col)[col].shift(lag)
                
        logger.info(f"Added lag features for {len(value_cols)} columns with {len(lags)} lags each")
        return df
    
    def add_rolling_features(self, df, time_col, entity_col, value_cols, windows, funcs):
        """
        Create rolling window features for time series data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            time_col (str): Time column name
            entity_col (str): Entity column name (e.g., product_id)
            value_cols (list): Value columns to create rolling features for
            windows (list): List of window sizes
            funcs (list): List of functions to apply (e.g., 'mean', 'std')
            
        Returns:
            pd.DataFrame: DataFrame with rolling window features
        """
        df = df.copy().sort_values([entity_col, time_col])
        
        for col in value_cols:
            for window in windows:
                for func in funcs:
                    roll_col_name = f"{col}_{func}_{window}"
                    
                    if func == 'mean':
                        df[roll_col_name] = df.groupby(entity_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                    elif func == 'std':
                        df[roll_col_name] = df.groupby(entity_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).std()
                        )
                    elif func == 'max':
                        df[roll_col_name] = df.groupby(entity_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).max()
                        )
                    elif func == 'min':
                        df[roll_col_name] = df.groupby(entity_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).min()
                        )
        
        logger.info(f"Added rolling features for {len(value_cols)} columns")
        return df


if __name__ == "__main__":
    # Example usage
    data = {
        'date': pd.date_range(start='2023-01-01', periods=10),
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'D', 'A'],
        'price': [99.99, 149.99, 89.99, 199.99, 159.99, 79.99, 189.99, 139.99, 299.99, 109.99],
        'original_price': [129.99, 179.99, 109.99, 249.99, 189.99, 99.99, 229.99, 159.99, 349.99, 129.99],
        'product_id': [1, 2, 1, 3, 2, 1, 3, 2, 4, 1]
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineering()
    result = fe.fit_transform(
        df, 
        categorical_cols=['category'],
        numerical_cols=['price', 'original_price']
    )
    
    print("Original DataFrame shape:", df.shape)
    print("Transformed DataFrame shape:", result.shape)
    print("New features:", set(result.columns) - set(df.columns))
