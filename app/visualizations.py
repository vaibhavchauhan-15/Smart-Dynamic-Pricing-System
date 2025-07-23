"""
Visualization module for SmartDynamic pricing app.
This module provides functions for creating charts and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

def price_history_chart(product_id, history_data):
    """
    Create an interactive price history chart.
    
    Args:
        product_id (str): Product ID
        history_data (list): List of price history records
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    df = pd.DataFrame(history_data)
    
    fig = px.line(df, x="date", y=["price", "optimal_price", "competitor_price"], 
                 title=f"Price History for Product {product_id}",
                 labels={"value": "Price", "date": "Date", "variable": "Price Type"},
                 template="plotly_white")
    
    # Customize line styles and colors
    fig.update_traces(
        line=dict(width=3),
        selector=dict(name="price")
    )
    
    # Update layout for better appearance
    fig.update_layout(
        legend_title_text="Price Type", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Date",
            tickformat="%b %d, %Y",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            title="Price (₹)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        hovermode="x unified"
    )
    
    # Rename series to make them more user-friendly
    new_names = {
        "price": "Current Price",
        "optimal_price": "AI-Recommended Price",
        "competitor_price": "Competitor Price"
    }
    
    for i, trace in enumerate(fig.data):
        fig.data[i].name = new_names.get(trace.name, trace.name)
    
    return fig

def demand_elasticity_chart(elasticity_data):
    """
    Create price elasticity curve chart.
    
    Args:
        elasticity_data (list): List of elasticity data points
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    df = pd.DataFrame(elasticity_data)
    
    # Create figure with secondary y-axis for revenue
    fig = go.Figure()
    
    # Add demand curve
    fig.add_trace(
        go.Scatter(
            x=df["price"], 
            y=df["demand"],
            name="Demand",
            line=dict(color="blue", width=3),
            hovertemplate="Price: ₹%{x:.2f}<br>Demand: %{y:.0f}<extra></extra>"
        )
    )
    
    # Add revenue curve
    fig.add_trace(
        go.Scatter(
            x=df["price"],
            y=df["revenue"],
            name="Revenue",
            line=dict(color="green", width=3, dash="dot"),
            yaxis="y2",
            hovertemplate="Price: ₹%{x:.2f}<br>Revenue: ₹%{y:.2f}<extra></extra>"
        )
    )
    
    # Find revenue-maximizing price point
    max_revenue_idx = df["revenue"].idxmax()
    optimal_price = df.iloc[max_revenue_idx]["price"]
    optimal_demand = df.iloc[max_revenue_idx]["demand"]
    optimal_revenue = df.iloc[max_revenue_idx]["revenue"]
    
    # Add marker for optimal price point
    fig.add_trace(
        go.Scatter(
            x=[optimal_price],
            y=[optimal_demand],
            mode="markers",
            marker=dict(size=12, color="red", symbol="star"),
            name=f"Optimal Price (₹{optimal_price:.2f})",
            hovertemplate="Optimal Price: ₹%{x:.2f}<br>Demand: %{y:.0f}<extra></extra>"
        )
    )
    
    # Set titles and labels
    fig.update_layout(
        title="Price Elasticity and Revenue",
        xaxis=dict(
            title="Price (₹)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            title="Estimated Demand (Units)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis2=dict(
            title="Revenue (₹)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def shap_waterfall_chart(shap_values):
    """
    Create SHAP waterfall chart for price explanation.
    
    Args:
        shap_values (list): List of SHAP values with feature names
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    # Sort factors by absolute impact
    factors = sorted(shap_values, key=lambda x: abs(x["value"]), reverse=True)
    
    # Extract feature names and values
    features = [f"{item['feature']} ({item['value']:.2f})" for item in factors]
    values = [item["value"] for item in factors]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.5)))
    
    # Create horizontal bar chart
    colors = ["#2ca02c" if x > 0 else "#d62728" for x in values]
    bars = ax.barh(features, values, color=colors)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.5 if width > 0 else width - 0.5
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f"{width:.2f}",
                va='center', ha='left' if width > 0 else 'right', color='black')
    
    # Formatting
    ax.set_title("Factors Influencing Price Recommendation", fontsize=14)
    ax.set_xlabel("Impact on Price (₹)", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add legend
    import matplotlib.patches as mpatches
    increase_patch = mpatches.Patch(color='#2ca02c', label='Increases Price')
    decrease_patch = mpatches.Patch(color='#d62728', label='Decreases Price')
    ax.legend(handles=[increase_patch, decrease_patch], loc='lower right')
    
    plt.tight_layout()
    return fig

def sales_volume_chart(sales_data, product_id):
    """
    Create sales volume chart.
    
    Args:
        sales_data (list): List of sales volume records
        product_id (str): Product ID
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    df = pd.DataFrame(sales_data)
    
    # Convert to datetime for better formatting
    df["date"] = pd.to_datetime(df["date"])
    
    fig = px.bar(df, x="date", y="volume",
                title=f"Sales Volume for Product {product_id}",
                labels={"volume": "Units Sold", "date": "Date"},
                template="plotly_white")
    
    # Add moving average line
    df["ma7"] = df["volume"].rolling(window=7).mean()
    
    fig.add_scatter(
        x=df["date"], 
        y=df["ma7"],
        mode="lines",
        name="7-Day Moving Avg",
        line=dict(color="red", width=3)
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            title="Date",
            tickformat="%b %d",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            title="Units Sold",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        hovermode="x unified",
        bargap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def price_comparison_chart(competitor_data):
    """
    Create competitor price comparison chart.
    
    Args:
        competitor_data (dict): Dictionary with competitor price data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    df = pd.DataFrame(competitor_data)
    
    fig = px.bar(df, y="competitor", x="avg_price", 
                title="Average Price Comparison",
                labels={"avg_price": "Average Price (₹)", "competitor": "Retailer"},
                template="plotly_white",
                orientation="h")
    
    # Highlight "YourStore"
    fig.update_traces(marker_color=['#2ca02c' if c == "YourStore" else '#1f77b4' 
                                for c in df.competitor])
    
    # Update layout
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis=dict(
            title="Average Price (₹)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        )
    )
    
    return fig

def market_price_index_chart(market_data):
    """
    Create market price index chart.
    
    Args:
        market_data (pd.DataFrame): DataFrame with market data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = px.line(market_data, x="date", y=["market_price_index", "our_price_index"],
                 title="Price Index Comparison",
                 labels={"value": "Price Index", "date": "Date", "variable": "Index Type"},
                 template="plotly_white")
    
    # Rename series
    new_names = {
        "market_price_index": "Market Average",
        "our_price_index": "Your Store"
    }
    
    for i, trace in enumerate(fig.data):
        fig.data[i].name = new_names.get(trace.name, trace.name)
        
        # Make "Your Store" line thicker and highlighted
        if trace.name == "Your Store":
            fig.data[i].line = dict(width=3, color="#2ca02c")
        else:
            fig.data[i].line = dict(width=2)
    
    # Update layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Date",
            tickformat="%b %d",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            title="Price Index (Base 100)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        hovermode="x unified"
    )
    
    return fig
