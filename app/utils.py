"""
Utility functions for SmartDynamic pricing app.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import json
import os

def generate_mock_data(product_id, days=90, base_price=None):
    """
    Generate mock data for demo purposes.
    
    Args:
        product_id (str): Product ID
        days (int): Number of days to generate data for
        base_price (float, optional): Base price for the product
        
    Returns:
        dict: Dictionary of mock data
    """
    # Set random seed based on product_id for consistent results
    seed = sum(ord(c) for c in product_id)
    np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data
    if base_price is None:
        base_price = np.random.uniform(50, 200)
    
    # Add some randomness and trends to prices
    price_trend = np.cumsum(np.random.normal(0, 0.01, size=len(dates)))
    seasonal = 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    
    prices = base_price * (1 + price_trend + seasonal)
    optimal_prices = prices * np.random.uniform(0.95, 1.15, size=len(dates))
    competitor_prices = prices * np.random.uniform(0.9, 1.1, size=len(dates))
    
    # Generate sales volume data (influenced by price)
    base_volume = np.random.randint(30, 100)
    price_elasticity = -1.5
    
    volumes = []
    for i, price in enumerate(prices):
        relative_price = price / base_price
        elasticity_effect = relative_price ** price_elasticity
        
        # Add random noise and day-of-week effect
        dow_effect = 1 + 0.2 * np.sin(2 * np.pi * (i % 7) / 7)
        noise = np.random.normal(1, 0.1)
        
        volume = int(base_volume * elasticity_effect * dow_effect * noise)
        volumes.append(max(0, volume))
    
    # Create price history
    price_history = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "price": float(p),
            "optimal_price": float(op),
            "competitor_price": float(cp)
        }
        for d, p, op, cp in zip(dates, prices, optimal_prices, competitor_prices)
    ]
    
    # Create sales volume history
    sales_volume = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "volume": int(v)
        }
        for d, v in zip(dates, volumes)
    ]
    
    # Create elasticity curve
    price_range = np.linspace(base_price * 0.7, base_price * 1.5, 20)
    demands = base_volume * (price_range / base_price) ** price_elasticity
    revenues = price_range * demands
    
    elasticity_curve = [
        {
            "price": float(p),
            "demand": float(d),
            "revenue": float(r)
        }
        for p, d, r in zip(price_range, demands, revenues)
    ]
    
    # Performance metrics
    performance = {
        "optimal_adherence": float(np.random.uniform(70, 95)),
        "average_margin": float(np.random.uniform(25, 40)),
        "price_change_frequency": float(np.random.uniform(1, 4)),
        "competitor_difference": float(np.random.uniform(-8, 8))
    }
    
    # Return all mock data
    return {
        "price_history": price_history,
        "sales_volume": sales_volume,
        "elasticity_curve": elasticity_curve,
        "price_performance": performance
    }

def generate_mock_recommendation(product_id, context=None):
    """
    Generate mock price recommendation.
    
    Args:
        product_id (str): Product ID
        context (dict, optional): Pricing context
        
    Returns:
        dict: Mock recommendation data
    """
    # Set random seed based on product_id for consistent results
    seed = sum(ord(c) for c in product_id)
    np.random.seed(seed)
    
    # Base price
    base_price = np.random.uniform(50, 200)
    
    # Apply context factors
    adjustment = 1.0
    if context:
        # Competitor discount effect
        competitor_discount = context.get("competitor_discount", 0)
        if competitor_discount > 0:
            adjustment -= 0.3 * competitor_discount / 100
        
        # Special event effect
        if context.get("special_event", False):
            adjustment += 0.05
        
        # Stock level effect
        stock_level = context.get("stock_level", "Medium")
        stock_effect = {
            "Very Low": 0.1,
            "Low": 0.05,
            "Medium": 0,
            "High": -0.03,
            "Excess": -0.1
        }
        adjustment += stock_effect.get(stock_level, 0)
        
        # Customer segment effect
        segments = context.get("target_segments", [])
        if "Premium" in segments:
            adjustment += 0.05
        if "Value Conscious" in segments:
            adjustment -= 0.03
        if "Deal Hunters" in segments:
            adjustment -= 0.08
    
    # Calculate optimal price
    optimal_price = base_price * (1 + adjustment)
    
    # Current price slightly different from base
    current_price = base_price * np.random.uniform(0.95, 1.05)
    
    # Competitor price
    competitor_discount_factor = 1.0
    if context and "competitor_discount" in context:
        competitor_discount_factor = 1.0 - (context["competitor_discount"] / 100)
    competitor_price = base_price * competitor_discount_factor * np.random.uniform(0.95, 1.05)
    
    # Calculate change percentage
    price_change_pct = 100 * (optimal_price - current_price) / current_price
    
    # Estimate impact
    price_elasticity = -1.5
    base_volume = np.random.randint(100, 500)
    
    current_volume = base_volume * (current_price / base_price) ** price_elasticity
    optimal_volume = base_volume * (optimal_price / base_price) ** price_elasticity
    
    current_revenue = current_price * current_volume
    optimal_revenue = optimal_price * optimal_volume
    
    revenue_impact = optimal_revenue - current_revenue
    revenue_impact_pct = 100 * revenue_impact / current_revenue
    
    # Profit margin (assuming cost is 60% of base price)
    cost = 0.6 * base_price
    profit_margin_pct = 100 * (optimal_price - cost) / optimal_price
    
    # Return recommendation
    return {
        "product_id": product_id,
        "optimal_price": float(optimal_price),
        "current_price": float(current_price),
        "competitor_price": float(competitor_price),
        "price_change_pct": float(price_change_pct),
        "revenue_impact": float(revenue_impact),
        "revenue_impact_pct": float(revenue_impact_pct),
        "estimated_volume": int(optimal_volume),
        "profit_margin_pct": float(profit_margin_pct)
    }

def generate_mock_explanation(product_id, context=None):
    """
    Generate mock price explanation.
    
    Args:
        product_id (str): Product ID
        context (dict, optional): Pricing context
        
    Returns:
        dict: Mock explanation data
    """
    # Set random seed based on product_id for consistent results
    seed = sum(ord(c) for c in product_id)
    np.random.seed(seed)
    
    # Select a random strategy
    strategies = ["Demand-based Pricing", "Competitive Pricing", "Reinforcement Learning", 
                 "Multi-armed Bandit", "Rule-based Pricing"]
    strategy = np.random.choice(strategies)
    
    # Generate SHAP factors
    factors = [
        {"feature": "Base Price", "value": float(np.random.uniform(5, 20))},
        {"feature": "Historical Demand", "value": float(np.random.uniform(-8, 15))},
        {"feature": "Competitor Price", "value": float(np.random.uniform(-10, 10))},
        {"feature": "Stock Level", "value": float(np.random.uniform(-5, 5))},
        {"feature": "Day of Week", "value": float(np.random.uniform(-2, 2))},
        {"feature": "Season", "value": float(np.random.uniform(-3, 3))}
    ]
    
    if context:
        if context.get("special_event", False):
            factors.append({"feature": "Special Event", "value": float(np.random.uniform(3, 8))})
        
        segments = context.get("target_segments", [])
        if segments:
            factors.append({"feature": "Customer Segment", "value": float(np.random.uniform(-5, 8))})
    
    # Generate elasticity curve
    base_price = np.random.uniform(50, 200)
    base_demand = np.random.randint(100, 500)
    price_elasticity = -1.5
    
    price_range = np.linspace(base_price * 0.7, base_price * 1.5, 20)
    demands = base_demand * (price_range / base_price) ** price_elasticity
    revenues = price_range * demands
    
    elasticity_curve = [
        {
            "price": float(p),
            "demand": float(d),
            "revenue": float(r)
        }
        for p, d, r in zip(price_range, demands, revenues)
    ]
    
    # Return explanation
    return {
        "product_id": product_id,
        "strategy": strategy,
        "factors": factors,
        "elasticity_curve": elasticity_curve
    }

def format_currency(value):
    """
    Format a value as currency.
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted currency string
    """
    return f"₹{value:,.2f}"

def create_price_history_chart(product_id, history_data):
    """
    Create a price history chart using Plotly.
    
    Args:
        product_id (str): Product ID
        history_data (list): List of price history records
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    df = pd.DataFrame(history_data)
    
    fig = px.line(df, x="date", y=["price", "optimal_price", "competitor_price"], 
                 title=f"Price History for Product {product_id}",
                 labels={"value": "Price", "date": "Date", "variable": "Price Type"},
                 template="plotly_white")
    
    fig.update_layout(legend_title_text="Price Type", 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

def create_demand_elasticity_chart(elasticity_data):
    """
    Create a demand elasticity chart using Plotly.
    
    Args:
        elasticity_data (list): List of elasticity data points
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    df = pd.DataFrame(elasticity_data)
    
    fig = px.line(df, x="price", y="demand", 
                 title="Price Elasticity of Demand",
                 labels={"price": "Price", "demand": "Estimated Demand"},
                 template="plotly_white")
    
    # Add revenue-maximizing point
    max_revenue_idx = df["revenue"].idxmax()
    optimal_price = df.iloc[max_revenue_idx]["price"]
    optimal_demand = df.iloc[max_revenue_idx]["demand"]
    
    fig.add_scatter(x=[optimal_price], 
                   y=[optimal_demand], 
                   mode="markers", 
                   marker=dict(size=12, color="red"),
                   name="Revenue-Maximizing Price")
    
    return fig
