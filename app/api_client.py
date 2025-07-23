"""
API client for SmartDynamic pricing system.
This module handles all interactions with the FastAPI backend.
"""

import requests
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

class ApiClient:
    """
    Client for interacting with SmartDynamic pricing API.
    """
    
    def __init__(self, base_url=None):
        """
        Initialize the API client.
        
        Args:
            base_url (str, optional): Base URL for the API. Defaults to the API_BASE_URL environment variable or localhost.
        """
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        logger.info(f"API Client initialized with base URL: {self.base_url}")
    
    def _make_request(self, method, endpoint, data=None, params=None):
        """
        Make a request to the API.
        
        Args:
            method (str): HTTP method ('get', 'post', 'put', 'delete')
            endpoint (str): API endpoint (without base URL)
            data (dict, optional): Data to send in the request body
            params (dict, optional): Query parameters
            
        Returns:
            dict: Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.lower() == 'get':
                response = requests.get(url, params=params)
            elif method.lower() == 'post':
                response = requests.post(url, json=data, params=params)
            elif method.lower() == 'put':
                response = requests.put(url, json=data, params=params)
            elif method.lower() == 'delete':
                response = requests.delete(url, json=data, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}
    
    def get_products(self, category=None, limit=100):
        """
        Get list of products.
        
        Args:
            category (str, optional): Filter products by category
            limit (int, optional): Maximum number of products to return
            
        Returns:
            list: List of products
        """
        params = {"limit": limit}
        if category:
            params["category"] = category
            
        return self._make_request('get', '/products', params=params)
    
    def get_price_recommendation(self, product_id, context=None):
        """
        Get price recommendation for a product.
        
        Args:
            product_id (str): Product ID
            context (dict, optional): Additional context for pricing
            
        Returns:
            dict: Price recommendation data
        """
        data = {
            "product_id": product_id,
            "context": context or {}
        }
        
        return self._make_request('post', '/price/recommend', data=data)
    
    def get_price_explanation(self, product_id, context=None):
        """
        Get explanation for a price recommendation.
        
        Args:
            product_id (str): Product ID
            context (dict, optional): Additional context for pricing
            
        Returns:
            dict: Price explanation data
        """
        data = {
            "product_id": product_id,
            "context": context or {}
        }
        
        return self._make_request('post', '/price/explain', data=data)
    
    def get_product_insights(self, product_id):
        """
        Get insights for a product.
        
        Args:
            product_id (str): Product ID
            
        Returns:
            dict: Product insights data
        """
        return self._make_request('get', f'/product/insights/{product_id}')
    
    def get_market_analysis(self, category=None):
        """
        Get market analysis data.
        
        Args:
            category (str, optional): Product category
            
        Returns:
            dict: Market analysis data
        """
        params = {}
        if category and category.lower() != "all categories":
            params["category"] = category
            
        return self._make_request('get', '/market/analysis', params=params)
    
    def apply_price_recommendation(self, product_id, price):
        """
        Apply a price recommendation.
        
        Args:
            product_id (str): Product ID
            price (float): Price to apply
            
        Returns:
            dict: Result of the price application
        """
        data = {
            "product_id": product_id,
            "price": price
        }
        
        return self._make_request('post', '/price/apply', data=data)
    
    def get_health(self):
        """
        Check API health.
        
        Returns:
            dict: API health status
        """
        try:
            return self._make_request('get', '/health')
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
