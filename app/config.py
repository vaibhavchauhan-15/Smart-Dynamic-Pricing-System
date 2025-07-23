"""
Configuration for SmartDynamic pricing app.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL", "http://localhost:8000"),
    "timeout": 10,
    "retry_attempts": 3
}

# App configuration
APP_CONFIG = {
    "title": "SmartDynamic Pricing",
    "icon": "💲",
    "layout": "wide",
    "theme": {
        "primaryColor": "#2E86C1",
        "backgroundColor": "#F8F9F9",
        "secondaryBackgroundColor": "#E5E8E8",
        "textColor": "#17202A",
        "font": "sans-serif"
    }
}

# Pricing strategies
PRICING_STRATEGIES = [
    {"id": "demand_based", "name": "Demand Based", "description": "Optimize price based on demand forecasting"},
    {"id": "competitive", "name": "Competitive Pricing", "description": "Price relative to competitor prices"},
    {"id": "mab", "name": "Multi-armed Bandit", "description": "Explore-exploit price points using MAB algorithms"},
    {"id": "rl", "name": "Reinforcement Learning", "description": "Use RL agents to optimize long-term pricing strategy"},
    {"id": "rule_based", "name": "Rule Based", "description": "Apply business rules and constraints to pricing"}
]

# Context factors that influence pricing
CONTEXT_FACTORS = {
    "market_factors": [
        "competitor_discount",
        "special_event",
        "stock_level",
        "season"
    ],
    "customer_segments": [
        "Premium",
        "Value Conscious",
        "Deal Hunters",
        "Loyal Customers"
    ],
    "stock_levels": [
        "Very Low",
        "Low", 
        "Medium", 
        "High", 
        "Excess"
    ]
}

# Demo/sample data configuration
SAMPLE_DATA = {
    "enabled": True,  # Use sample data if API is not available
    "products": [
        {"id": "P001", "name": "Premium Headphones", "category": "Electronics", "base_price": 149.99},
        {"id": "P002", "name": "Organic Coffee Beans", "category": "Grocery", "base_price": 12.99},
        {"id": "P003", "name": "Fitness Tracker", "category": "Electronics", "base_price": 89.99},
        {"id": "P004", "name": "Yoga Mat", "category": "Sports", "base_price": 29.99},
        {"id": "P005", "name": "Wireless Mouse", "category": "Electronics", "base_price": 24.99},
        {"id": "P006", "name": "Protein Powder", "category": "Grocery", "base_price": 39.99},
        {"id": "P007", "name": "Running Shoes", "category": "Sports", "base_price": 119.99},
        {"id": "P008", "name": "Smart Speaker", "category": "Electronics", "base_price": 79.99}
    ]
}
