"""
Streamlit frontend for SmartDynamic pricing system.
This app provides a user interface to interact with the pricing API and visualize pricing recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="SmartDynamic Pricing",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Define helper functions for API calls
def get_price_recommendation(product_id, context=None):
    """Get price recommendation from API."""
    url = f"{API_BASE_URL}/price/recommend"
    data = {
        "product_id": product_id,
        "context": context or {}
    }
    try:
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def get_price_explanation(product_id, context=None):
    """Get explanation for price recommendation."""
    url = f"{API_BASE_URL}/price/explain"
    data = {
        "product_id": product_id,
        "context": context or {}
    }
    try:
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None
        
def get_product_insights(product_id):
    """Get insights about product pricing history and performance."""
    url = f"{API_BASE_URL}/product/insights/{product_id}"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def get_products():
    """Get list of products from API."""
    url = f"{API_BASE_URL}/products"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return []

def display_price_history_chart(product_id, history_data):
    """Display interactive price history chart."""
    df = pd.DataFrame(history_data)
    
    fig = px.line(df, x="date", y=["price", "optimal_price", "competitor_price"], 
                 title=f"Price History for Product {product_id}",
                 labels={"value": "Price", "date": "Date", "variable": "Price Type"},
                 template="plotly_white")
    
    fig.update_layout(legend_title_text="Price Type", 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig, use_container_width=True)

def display_demand_elasticity_chart(elasticity_data):
    """Display price elasticity curve."""
    df = pd.DataFrame(elasticity_data)
    
    fig = px.line(df, x="price", y="demand", 
                 title="Price Elasticity of Demand",
                 labels={"price": "Price", "demand": "Estimated Demand"},
                 template="plotly_white")
    
    fig.add_scatter(x=[df.iloc[df["revenue"].idxmax()]["price"]], 
                   y=[df.iloc[df["revenue"].idxmax()]["demand"]], 
                   mode="markers", 
                   marker=dict(size=12, color="red"),
                   name="Revenue-Maximizing Price")
    
    st.plotly_chart(fig, use_container_width=True)

def display_shap_waterfall(shap_values):
    """Display SHAP waterfall chart for price explanation."""
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort factors by impact
    factors = sorted(shap_values, key=lambda x: abs(x["value"]), reverse=True)
    
    # Extract feature names and values
    features = [f"{item['feature']} ({item['value']:.2f})" for item in factors]
    values = [item["value"] for item in factors]
    
    # Create horizontal bar chart
    colors = ["green" if x > 0 else "red" for x in values]
    ax.barh(features, values, color=colors)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Formatting
    ax.set_title("Factors Influencing Price Recommendation")
    ax.set_xlabel("Impact on Price (₹)")
    
    plt.tight_layout()
    st.pyplot(fig)

# App UI starts here
def main():
    st.title("SmartDynamic Pricing Dashboard 💲")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Price Recommendations", "Product Insights", "Market Analysis", "Settings"])
    
    # Add branding and information
    with st.sidebar.expander("About SmartDynamic"):
        st.write("""
        SmartDynamic is an AI-powered dynamic pricing system that optimizes prices in real-time 
        based on demand patterns, competitor pricing, and other market factors.
        """)
    
    # Price Recommendations page
    if page == "Price Recommendations":
        st.header("Price Recommendations")
        
        # Get products (mock data for now)
        try:
            products = get_products()
        except:
            # Mock data if API is not available
            products = [
                {"id": "P001", "name": "Premium Headphones", "category": "Electronics"},
                {"id": "P002", "name": "Organic Coffee Beans", "category": "Grocery"},
                {"id": "P003", "name": "Fitness Tracker", "category": "Electronics"},
                {"id": "P004", "name": "Yoga Mat", "category": "Sports"},
            ]
        
        # Layout with columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Product selection
            selected_product = st.selectbox(
                "Select Product",
                options=[p["id"] for p in products],
                format_func=lambda x: f"{x} - {next((p['name'] for p in products if p['id'] == x), 'Unknown')}"
            )
            
            # Context inputs
            st.subheader("Pricing Context")
            
            with st.expander("Market Factors", expanded=True):
                competitor_discount = st.slider("Competitor Discount (%)", 0, 50, 0)
                special_event = st.checkbox("Special Event/Holiday")
                stock_level = st.select_slider(
                    "Inventory Stock Level",
                    options=["Very Low", "Low", "Medium", "High", "Excess"]
                )
            
            with st.expander("Customer Segments"):
                target_segment = st.multiselect(
                    "Target Customer Segments",
                    ["Premium", "Value Conscious", "Deal Hunters", "Loyal Customers"],
                    default=["Value Conscious"]
                )
            
            # Build context dictionary for API
            context = {
                "competitor_discount": competitor_discount,
                "special_event": special_event,
                "stock_level": stock_level,
                "target_segments": target_segment,
                "timestamp": datetime.now().isoformat()
            }
            
            # Get recommendation button
            if st.button("Get Price Recommendation", type="primary"):
                with st.spinner("Calculating optimal price..."):
                    recommendation = get_price_recommendation(selected_product, context)
                    
                    if recommendation:
                        st.session_state.recommendation = recommendation
                        st.session_state.selected_product = selected_product
                        st.session_state.context = context
                        
                        # Also fetch explanation
                        explanation = get_price_explanation(selected_product, context)
                        if explanation:
                            st.session_state.explanation = explanation
        
        with col2:
            if "recommendation" in st.session_state and st.session_state.selected_product == selected_product:
                recommendation = st.session_state.recommendation
                
                # Display recommendation card
                st.subheader("Price Recommendation")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric(
                        "Optimal Price", 
                        f"₹{recommendation['optimal_price']:.2f}",
                        f"{recommendation.get('price_change_pct', 0):.1f}%"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        "Current Price", 
                        f"₹{recommendation.get('current_price', 0):.2f}"
                    )
                
                with metric_cols[2]:
                    st.metric(
                        "Competitor Price", 
                        f"₹{recommendation.get('competitor_price', 0):.2f}"
                    )
                
                # Additional metrics
                st.markdown("#### Estimated Impact")
                impact_cols = st.columns(3)
                with impact_cols[0]:
                    st.metric(
                        "Revenue Impact", 
                        f"₹{recommendation.get('revenue_impact', 0):.2f}",
                        f"{recommendation.get('revenue_impact_pct', 0):.1f}%"
                    )
                
                with impact_cols[1]:
                    st.metric(
                        "Est. Sales Volume", 
                        f"{recommendation.get('estimated_volume', 0)}"
                    )
                
                with impact_cols[2]:
                    st.metric(
                        "Profit Margin", 
                        f"{recommendation.get('profit_margin_pct', 0):.1f}%"
                    )
                
                # Price Explanation
                if "explanation" in st.session_state:
                    explanation = st.session_state.explanation
                    
                    st.subheader("Price Recommendation Explanation")
                    
                    # Display strategy used
                    st.markdown(f"**Strategy:** {explanation.get('strategy', 'Optimal Pricing')}")
                    
                    # Display SHAP values visualization
                    if "factors" in explanation:
                        display_shap_waterfall(explanation["factors"])
                    
                    # Show elasticity curve
                    if "elasticity_curve" in explanation:
                        display_demand_elasticity_chart(explanation["elasticity_curve"])
            else:
                # Placeholder when no recommendation is available
                st.info("Select a product and click 'Get Price Recommendation' to see pricing insights")
                
                # Sample visualization as placeholder
                placeholder_data = {
                    "date": pd.date_range(start=datetime.now()-timedelta(days=30), periods=30, freq="D"),
                    "price": np.random.normal(100, 5, 30),
                    "optimal_price": np.random.normal(105, 3, 30),
                    "competitor_price": np.random.normal(102, 7, 30)
                }
                placeholder_df = pd.DataFrame(placeholder_data)
                
                fig = px.line(placeholder_df, x="date", y=["price", "optimal_price", "competitor_price"], 
                             title="Sample Price Tracking (Select a product for actual data)",
                             labels={"value": "Price", "date": "Date", "variable": "Price Type"},
                             template="plotly_white")
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Product Insights page
    elif page == "Product Insights":
        st.header("Product Insights")
        
        # Get products (mock data for now)
        try:
            products = get_products()
        except:
            # Mock data if API is not available
            products = [
                {"id": "P001", "name": "Premium Headphones", "category": "Electronics"},
                {"id": "P002", "name": "Organic Coffee Beans", "category": "Grocery"},
                {"id": "P003", "name": "Fitness Tracker", "category": "Electronics"},
                {"id": "P004", "name": "Yoga Mat", "category": "Sports"},
            ]
        
        # Product selection
        selected_product = st.selectbox(
            "Select Product",
            options=[p["id"] for p in products],
            format_func=lambda x: f"{x} - {next((p['name'] for p in products if p['id'] == x), 'Unknown')}"
        )
        
        # Time range selection
        time_range = st.radio("Time Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date"], horizontal=True)
        
        # Fetch product insights
        insights = get_product_insights(selected_product)
        
        if not insights:
            # Mock data for demonstration
            mock_dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
            insights = {
                "price_history": [
                    {"date": d.strftime("%Y-%m-%d"), 
                     "price": float(np.random.normal(100, 5)), 
                     "optimal_price": float(np.random.normal(105, 3)),
                     "competitor_price": float(np.random.normal(102, 7))}
                    for d in mock_dates
                ],
                "sales_volume": [
                    {"date": d.strftime("%Y-%m-%d"), 
                     "volume": int(np.random.normal(50, 15))}
                    for d in mock_dates
                ],
                "elasticity_curve": [
                    {"price": float(p), 
                     "demand": float(1000 * np.exp(-0.01 * p)),
                     "revenue": float(p * 1000 * np.exp(-0.01 * p))}
                    for p in range(70, 151, 5)
                ],
                "price_performance": {
                    "optimal_adherence": 78.5,
                    "average_margin": 32.4,
                    "price_change_frequency": 2.3,
                    "competitor_difference": -4.2
                }
            }
        
        # Display insights in tabs
        tabs = st.tabs(["Price History", "Sales Volume", "Price Elasticity", "Performance Metrics"])
        
        with tabs[0]:
            # Price history chart
            display_price_history_chart(selected_product, insights["price_history"])
            
            # Show data table with price history
            with st.expander("Price History Data"):
                price_history_df = pd.DataFrame(insights["price_history"])
                st.dataframe(price_history_df)
        
        with tabs[1]:
            # Sales volume chart
            sales_df = pd.DataFrame(insights["sales_volume"])
            fig = px.bar(sales_df, x="date", y="volume",
                        title=f"Sales Volume for Product {selected_product}",
                        labels={"volume": "Units Sold", "date": "Date"},
                        template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        with tabs[2]:
            # Display elasticity curve
            display_demand_elasticity_chart(insights["elasticity_curve"])
            
            # Additional explanation
            st.markdown("""
            **About Price Elasticity:**
            
            This chart shows how demand changes with different price points. The red dot indicates 
            the revenue-maximizing price point based on the current elasticity model.
            """)
            
        with tabs[3]:
            # Performance metrics
            perf = insights["price_performance"]
            
            # Create 2x2 grid of metrics
            metric_cols1 = st.columns(2)
            metric_cols2 = st.columns(2)
            
            with metric_cols1[0]:
                st.metric("Optimal Price Adherence", f"{perf['optimal_adherence']}%")
            
            with metric_cols1[1]:
                st.metric("Average Profit Margin", f"{perf['average_margin']}%")
            
            with metric_cols2[0]:
                st.metric("Price Changes per Week", f"{perf['price_change_frequency']}")
            
            with metric_cols2[1]:
                st.metric("Avg. Competitor Difference", f"{perf['competitor_difference']}%",
                        delta_color="inverse")
    
    # Market Analysis page
    elif page == "Market Analysis":
        st.header("Market Analysis")
        
        # Market trends
        st.subheader("Market Trends")
        
        # Category selection
        category = st.selectbox("Product Category", 
                               ["All Categories", "Electronics", "Grocery", "Sports", "Apparel"])
        
        # Mock market data
        mock_dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        market_data = pd.DataFrame({
            "date": mock_dates,
            "market_price_index": np.cumsum(np.random.normal(0, 0.5, size=60)) + 100,
            "our_price_index": np.cumsum(np.random.normal(0, 0.4, size=60)) + 98,
            "market_volume": np.random.normal(1000, 100, size=60)
        })
        
        # Market price index chart
        fig = px.line(market_data, x="date", y=["market_price_index", "our_price_index"],
                     title=f"Price Index Comparison - {category}",
                     labels={"value": "Price Index", "date": "Date", "variable": "Index Type"},
                     template="plotly_white")
        
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Competitor analysis section
        st.subheader("Competitor Analysis")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            # Competitor price comparison
            competitor_data = {
                "competitor": ["YourStore", "Competitor A", "Competitor B", "Competitor C", "Competitor D"],
                "avg_price": [100, 105, 98, 110, 95],
                "price_changes": [12, 8, 15, 5, 22]
            }
            
            comp_df = pd.DataFrame(competitor_data)
            
            # Horizontal bar chart for price comparison
            fig = px.bar(comp_df, y="competitor", x="avg_price", 
                        title="Average Price Comparison",
                        labels={"avg_price": "Average Price (₹)", "competitor": "Retailer"},
                        template="plotly_white",
                        orientation="h")
            
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            
            # Highlight "YourStore"
            fig.update_traces(marker_color=['#2ca02c' if c == "YourStore" else '#1f77b4' 
                                        for c in comp_df.competitor])
            
            st.plotly_chart(fig, use_container_width=True)
        
        with comp_col2:
            # Price change frequency
            fig = px.bar(comp_df, y="competitor", x="price_changes", 
                        title="Price Change Frequency (Last 30 Days)",
                        labels={"price_changes": "Number of Price Changes", "competitor": "Retailer"},
                        template="plotly_white",
                        orientation="h")
            
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            
            # Highlight "YourStore"
            fig.update_traces(marker_color=['#2ca02c' if c == "YourStore" else '#1f77b4' 
                                        for c in comp_df.competitor])
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Settings page
    elif page == "Settings":
        st.header("Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        
        if st.button("Test API Connection"):
            try:
                response = requests.get(f"{api_url}/health")
                if response.status_code == 200:
                    st.success("API connection successful!")
                else:
                    st.error(f"API returned status code {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
        
        # Pricing Strategy Settings
        st.subheader("Pricing Strategy Settings")
        
        strategy_options = ["Demand Based", "Competitive Pricing", "Multi-armed Bandit", "Reinforcement Learning", "Rule Based"]
        default_strategy = st.selectbox("Default Pricing Strategy", strategy_options)
        
        # Global price adjustment bounds
        st.subheader("Price Adjustment Bounds")
        
        min_pct = st.slider("Minimum Price Adjustment (%)", -50, 0, -15)
        max_pct = st.slider("Maximum Price Adjustment (%)", 0, 50, 15)
        
        # Save settings button
        if st.button("Save Settings"):
            # In a real app, this would save to a config file or database
            st.success("Settings saved successfully!")


# Run the app
if __name__ == "__main__":
    main()
