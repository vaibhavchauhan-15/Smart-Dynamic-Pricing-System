"""
Data loading module for SmartDynamic pricing system.
This module handles all data ingestion, preprocessing, and database interactions.
"""

import os
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pymongo import MongoClient
import requests
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

class DataLoader:
    """
    DataLoader class for handling all data related operations.
    Supports loading from CSV files, databases (SQL and NoSQL), and APIs.
    """
    
    def __init__(self, config=None):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary for data sources.
        """
        self.config = config or {}
        self.db_engine = None
        self.mongo_client = None
        self._initialize_connections()
        logger.info("DataLoader initialized")
        
    def _initialize_connections(self):
        """Initialize database connections if credentials are provided"""
        # PostgreSQL connection
        if os.getenv("POSTGRES_URI"):
            try:
                self.db_engine = create_engine(os.getenv("POSTGRES_URI"))
                logger.info("PostgreSQL connection established")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
        
        # MongoDB connection
        if os.getenv("MONGO_URI"):
            try:
                self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
                logger.info("MongoDB connection established")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
    
    def load_product_data(self, source_type="csv", source_path=None, query=None):
        """
        Load product data from specified source.
        
        Args:
            source_type (str): Type of data source ('csv', 'sql', 'mongodb', 'api')
            source_path (str): Path to CSV file or table name
            query (str): SQL query if source_type is 'sql'
            
        Returns:
            pd.DataFrame: DataFrame containing product data
        """
        try:
            if source_type == "csv":
                logger.info(f"Loading product data from CSV: {source_path}")
                return pd.read_csv(source_path)
            
            elif source_type == "sql":
                if self.db_engine is None:
                    logger.error("Database connection not established")
                    return None
                
                logger.info(f"Loading product data from SQL with query: {query}")
                return pd.read_sql(query or f"SELECT * FROM {source_path}", self.db_engine)
            
            elif source_type == "mongodb":
                if self.mongo_client is None:
                    logger.error("MongoDB connection not established")
                    return None
                
                db_name, collection = source_path.split('.')
                logger.info(f"Loading product data from MongoDB: {source_path}")
                data = list(self.mongo_client[db_name][collection].find())
                return pd.DataFrame(data)
            
            elif source_type == "api":
                logger.info(f"Loading product data from API: {source_path}")
                response = requests.get(source_path)
                response.raise_for_status()
                return pd.DataFrame(response.json())
            
            else:
                logger.error(f"Unsupported source type: {source_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading product data: {e}")
            return None
    
    def load_transaction_history(self, start_date=None, end_date=None, product_ids=None):
        """
        Load historical transaction data with optional date range and product filters.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            product_ids (list): List of product IDs to filter
            
        Returns:
            pd.DataFrame: DataFrame containing transaction history
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            query = f"""
                SELECT * FROM transactions 
                WHERE transaction_date BETWEEN '{start_date}' AND '{end_date}'
            """
            
            if product_ids:
                product_list = ", ".join([f"'{pid}'" for pid in product_ids])
                query += f" AND product_id IN ({product_list})"
                
            logger.info(f"Loading transaction history from {start_date} to {end_date}")
            return pd.read_sql(query, self.db_engine)
            
        except Exception as e:
            logger.error(f"Error loading transaction history: {e}")
            return None
    
    def load_competitor_prices(self, product_ids=None):
        """
        Load competitor pricing data for specified products.
        
        Args:
            product_ids (list): List of product IDs to get competitor pricing for
            
        Returns:
            pd.DataFrame: DataFrame containing competitor pricing
        """
        try:
            logger.info(f"Loading competitor pricing data for {len(product_ids) if product_ids else 'all'} products")
            
            if self.db_engine:
                query = "SELECT * FROM competitor_prices"
                if product_ids:
                    product_list = ", ".join([f"'{pid}'" for pid in product_ids])
                    query += f" WHERE product_id IN ({product_list})"
                return pd.read_sql(query, self.db_engine)
            else:
                logger.warning("Database connection not available for competitor prices")
                return None
                
        except Exception as e:
            logger.error(f"Error loading competitor prices: {e}")
            return None
    
    def load_weather_data(self, location_ids=None, date=None):
        """
        Load weather data for specified locations and date.
        
        Args:
            location_ids (list): List of location IDs
            date (str): Date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame containing weather data
        """
        try:
            # If we have a weather API key
            if os.getenv("WEATHER_API_KEY"):
                logger.info(f"Loading weather data from API for {date}")
                # This would be implemented with actual API call
                return pd.DataFrame()  # Placeholder
            else:
                logger.info(f"Loading weather data from local storage for {date}")
                # Placeholder for loading from local storage
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            return None
    
    def load_events_data(self, location_ids=None, start_date=None, end_date=None):
        """
        Load events data (holidays, festivals, etc.) for a date range.
        
        Args:
            location_ids (list): List of location IDs
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame containing events data
        """
        try:
            logger.info(f"Loading events data from {start_date} to {end_date}")
            # This would connect to a calendar API or local database
            return pd.DataFrame()  # Placeholder
            
        except Exception as e:
            logger.error(f"Error loading events data: {e}")
            return None
    
    def save_predictions(self, predictions_df, destination="db", table_name="price_predictions"):
        """
        Save price predictions to the specified destination.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame containing predictions
            destination (str): Where to save ('db', 'csv', 'mongodb')
            table_name (str): Table or collection name
            
        Returns:
            bool: Success status
        """
        try:
            if destination == "db" and self.db_engine:
                logger.info(f"Saving predictions to database table: {table_name}")
                predictions_df.to_sql(table_name, self.db_engine, if_exists='append', index=False)
                return True
                
            elif destination == "csv":
                file_path = f"data/{table_name}_{datetime.now().strftime('%Y%m%d')}.csv"
                logger.info(f"Saving predictions to CSV: {file_path}")
                predictions_df.to_csv(file_path, index=False)
                return True
                
            elif destination == "mongodb" and self.mongo_client:
                logger.info(f"Saving predictions to MongoDB collection: {table_name}")
                db_name = os.getenv("MONGO_DB_NAME", "smartdynamic")
                records = predictions_df.to_dict('records')
                self.mongo_client[db_name][table_name].insert_many(records)
                return True
                
            else:
                logger.error(f"Unsupported destination: {destination}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return False
            
    def __del__(self):
        """Clean up database connections on destruction"""
        if hasattr(self, 'db_engine') and self.db_engine:
            self.db_engine.dispose()
            
        if hasattr(self, 'mongo_client') and self.mongo_client:
            self.mongo_client.close()


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load sample CSV data (if available)
    try:
        df = loader.load_product_data(source_type="csv", source_path="data/sample_products.csv")
        print(f"Loaded {len(df)} product records from CSV")
    except:
        print("No sample CSV file available")
