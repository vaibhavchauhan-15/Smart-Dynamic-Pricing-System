"""
FastAPI backend for SmartDynamic pricing system.
This module implements RESTful API for price recommendations.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import uuid
import json
import os
from datetime import datetime
from loguru import logger

# Import local modules
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.price_optimizer import PriceOptimizer
from src.shap_explainer import ShapExplainer

# Initialize FastAPI app
app = FastAPI(
    title="SmartDynamic Pricing API",
    description="API for AI-powered dynamic pricing recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_loader = None
price_optimizer = None
explainer = None


# Pydantic models for request/response validation
class ProductFeatures(BaseModel):
    """Product features for pricing recommendation."""
    product_id: str = Field(..., description="Unique identifier of the product")
    category: Optional[str] = Field(None, description="Product category")
    brand: Optional[str] = Field(None, description="Product brand")
    cost: Optional[float] = Field(None, description="Product cost")
    current_price: Optional[float] = Field(None, description="Current price of the product")
    min_price: Optional[float] = Field(None, description="Minimum allowed price")
    max_price: Optional[float] = Field(None, description="Maximum allowed price")
    competitor_prices: Optional[Dict[str, float]] = Field(None, description="Competitor prices for the product")
    inventory_level: Optional[int] = Field(None, description="Current inventory level")
    sales_velocity: Optional[float] = Field(None, description="Sales velocity (units sold per day)")
    seasonality_factor: Optional[float] = Field(None, description="Seasonality factor")
    promotion_flag: Optional[bool] = Field(False, description="Whether the product is under promotion")
    days_since_last_price_change: Optional[int] = Field(None, description="Days since last price change")
    additional_features: Optional[Dict[str, Any]] = Field({}, description="Additional features")
    
    @validator('cost', 'current_price', 'min_price', 'max_price')
    def validate_prices(cls, v):
        if v is not None and v < 0:
            raise ValueError("Price values must be non-negative")
        return v


class PricingStrategy(BaseModel):
    """Pricing strategy specification."""
    name: str = Field(..., description="Name of the pricing strategy")
    weight: float = Field(1.0, description="Weight of the strategy in ensemble")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Strategy parameters")


class PricingRequest(BaseModel):
    """Request model for pricing recommendations."""
    features: ProductFeatures
    strategies: Optional[List[PricingStrategy]] = Field(None, description="Strategies to use")
    explanation_required: Optional[bool] = Field(False, description="Whether to include explanation")
    context: Optional[Dict[str, Any]] = Field({}, description="Additional context")


class PricePoint(BaseModel):
    """Price point with expected metrics."""
    price: float = Field(..., description="Recommended price")
    expected_profit: Optional[float] = Field(None, description="Expected profit")
    expected_revenue: Optional[float] = Field(None, description="Expected revenue")
    expected_units_sold: Optional[float] = Field(None, description="Expected units sold")
    confidence: Optional[float] = Field(None, description="Confidence score")


class ExplanationFactor(BaseModel):
    """Factor in price explanation."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")
    direction: str = Field(..., description="Effect direction (+/-)")
    value: Any = Field(..., description="Feature value")


class PricingExplanation(BaseModel):
    """Explanation for price recommendation."""
    explanation_type: str = Field(..., description="Type of explanation (shap, basic)")
    base_value: Optional[float] = Field(None, description="Base price before features")
    top_factors: Optional[List[ExplanationFactor]] = Field(None, description="Top influential factors")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    message: Optional[str] = Field(None, description="Explanation message")


class PricingResponse(BaseModel):
    """Response model for pricing recommendations."""
    request_id: str = Field(..., description="Unique request identifier")
    product_id: str = Field(..., description="Product identifier")
    recommended_price: float = Field(..., description="Recommended price")
    strategy_name: str = Field(..., description="Strategy used for pricing")
    strategy_type: str = Field(..., description="Type of strategy used")
    price_points: Optional[List[PricePoint]] = Field(None, description="Alternative price points")
    explanation: Optional[PricingExplanation] = Field(None, description="Price explanation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of recommendation")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, bool] = Field(..., description="Component status")


class BulkPricingRequest(BaseModel):
    """Request model for bulk pricing."""
    products: List[PricingRequest] = Field(..., description="List of products to price")
    batch_id: Optional[str] = Field(None, description="Batch identifier")


class BulkPricingResponse(BaseModel):
    """Response model for bulk pricing."""
    batch_id: str = Field(..., description="Batch identifier")
    recommendations: List[PricingResponse] = Field(..., description="List of recommendations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of batch")
    failed_products: Optional[List[str]] = Field(None, description="List of failed product IDs")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global data_loader, price_optimizer, explainer
    
    try:
        logger.info("Initializing API components")
        
        # Initialize data loader
        data_loader = DataLoader()
        
        # Initialize price optimizer
        price_optimizer = PriceOptimizer()
        
        # Initialize SHAP explainer
        explainer = ShapExplainer()
        
        logger.info("API components initialized successfully")
        
        # Load models if they exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if os.path.exists(models_dir):
            logger.info("Loading models from disk")
            price_optimizer.load_models(models_dir)
            
        # Load background data for explainer if it exists
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        background_data_path = os.path.join(data_dir, "background_data.csv")
        if os.path.exists(background_data_path):
            logger.info("Loading background data for explainer")
            background_data = pd.read_csv(background_data_path)
            explainer.background_data = background_data
            
    except Exception as e:
        logger.error(f"Error initializing API components: {e}")
        # Continue startup even if components failed to initialize
        # They will be initialized lazily when needed


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "data_loader": data_loader is not None,
            "price_optimizer": price_optimizer is not None,
            "explainer": explainer is not None
        }
    }


@app.post("/price/recommend", response_model=PricingResponse, tags=["Pricing"])
async def recommend_price(request: PricingRequest):
    """
    Get price recommendation for a product.
    """
    global price_optimizer, explainer
    
    try:
        # Ensure price optimizer is initialized
        if price_optimizer is None:
            price_optimizer = PriceOptimizer()
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Extract features as dictionary
        features = request.features.dict()
        product_id = features.pop("product_id")
        
        # Combine all features
        all_features = {**features, **(features.get("additional_features", {}) or {})}
        if "additional_features" in all_features:
            del all_features["additional_features"]
            
        # Get strategies if specified
        strategies = None
        if request.strategies:
            strategies = [s.dict() for s in request.strategies]
            
        # Get price recommendation
        recommendation = price_optimizer.get_price_recommendation(
            product_id, all_features, strategies=strategies
        )
        
        if not recommendation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate price recommendation"
            )
        
        # Generate explanation if required
        explanation = None
        if request.explanation_required:
            if explainer is None:
                explainer = ShapExplainer()
                
            # Try to get SHAP explanation
            shap_explanation = explainer.explain_price_recommendation(
                price_optimizer, product_id, all_features,
                strategy=recommendation["strategy"]
            )
            
            if shap_explanation and "explanation_type" in shap_explanation:
                if shap_explanation["explanation_type"] == "shap":
                    explanation = {
                        "explanation_type": "shap",
                        "base_value": shap_explanation["base_value"],
                        "top_factors": [
                            {
                                "feature": factor["feature"],
                                "importance": factor["importance"],
                                "direction": factor["direction"],
                                "value": factor["value"]
                            } 
                            for factor in shap_explanation["top_factors"]
                        ],
                        "feature_importance": shap_explanation["feature_importance"]
                    }
                else:
                    explanation = {
                        "explanation_type": "basic",
                        "message": shap_explanation.get("message", "Price determined by the selected strategy.")
                    }
            else:
                # Fallback to basic explanation
                explanation = {
                    "explanation_type": "basic",
                    "message": "Price determined by the selected strategy."
                }
                
        # Prepare price points if available
        price_points = None
        if "alternative_prices" in recommendation and recommendation["alternative_prices"]:
            price_points = [
                {
                    "price": price["price"],
                    "expected_profit": price.get("expected_profit"),
                    "expected_revenue": price.get("expected_revenue"),
                    "expected_units_sold": price.get("expected_units_sold"),
                    "confidence": price.get("confidence", 1.0)
                }
                for price in recommendation["alternative_prices"]
            ]
        
        # Prepare response
        response = {
            "request_id": request_id,
            "product_id": product_id,
            "recommended_price": recommendation["price"],
            "strategy_name": recommendation["strategy"],
            "strategy_type": recommendation["strategy_type"],
            "price_points": price_points,
            "explanation": explanation,
            "timestamp": datetime.now()
        }
        
        logger.info(f"Generated price recommendation for product {product_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating price recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate price recommendation: {str(e)}"
        )


@app.post("/price/bulk", response_model=BulkPricingResponse, tags=["Pricing"])
async def bulk_price_recommend(request: BulkPricingRequest):
    """
    Get price recommendations for multiple products.
    """
    try:
        # Generate batch ID if not provided
        batch_id = request.batch_id or str(uuid.uuid4())
        
        recommendations = []
        failed_products = []
        
        # Process each product
        for product_request in request.products:
            try:
                # Call the single product endpoint for each product
                recommendation = await recommend_price(product_request)
                recommendations.append(recommendation)
            except Exception as e:
                # Log the error and add to failed products
                logger.error(f"Error pricing product {product_request.features.product_id}: {e}")
                failed_products.append(product_request.features.product_id)
        
        # Prepare response
        response = {
            "batch_id": batch_id,
            "recommendations": recommendations,
            "timestamp": datetime.now(),
            "failed_products": failed_products if failed_products else None
        }
        
        logger.info(f"Generated bulk pricing recommendations for {len(recommendations)} products")
        return response
        
    except Exception as e:
        logger.error(f"Error generating bulk price recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate bulk price recommendations: {str(e)}"
        )


@app.post("/model/train", status_code=status.HTTP_202_ACCEPTED, tags=["Models"])
async def train_model(
    model_type: str = Body(..., description="Type of model to train"),
    product_categories: Optional[List[str]] = Body(None, description="Product categories to train for"),
    training_parameters: Optional[Dict[str, Any]] = Body({}, description="Training parameters")
):
    """
    Trigger model training.
    """
    try:
        # Ensure price optimizer is initialized
        if price_optimizer is None:
            price_optimizer = PriceOptimizer()
            
        # Validate model type
        valid_model_types = ["demand_forecasting", "reinforcement_learning", "bandit", "rule_based"]
        if model_type not in valid_model_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type. Must be one of: {valid_model_types}"
            )
            
        # Start training in background (would be async in production)
        # For now, just return an acknowledgment
        logger.info(f"Model training triggered for {model_type}")
        
        return {
            "message": f"Model training for {model_type} initiated",
            "model_type": model_type,
            "status": "training_initiated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating model training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate model training: {str(e)}"
        )


@app.post("/model/evaluate", tags=["Models"])
async def evaluate_model(
    model_name: str = Body(..., description="Name of the model to evaluate"),
    evaluation_data: Optional[str] = Body(None, description="Path to evaluation data"),
    evaluation_parameters: Optional[Dict[str, Any]] = Body({}, description="Evaluation parameters")
):
    """
    Evaluate a trained model.
    """
    try:
        # Ensure price optimizer is initialized
        if price_optimizer is None:
            price_optimizer = PriceOptimizer()
            
        # Check if model exists
        if model_name not in price_optimizer.models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
            
        # For now, just return an acknowledgment
        logger.info(f"Model evaluation triggered for {model_name}")
        
        return {
            "message": f"Model evaluation for {model_name} complete",
            "model_name": model_name,
            "metrics": {
                "accuracy": 0.85,  # Placeholder metrics
                "mae": 0.12,
                "rmse": 0.18
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate model: {str(e)}"
        )


@app.get("/explanation/{product_id}", tags=["Explanations"])
async def get_price_explanation(
    product_id: str,
    strategy: Optional[str] = Query(None, description="Pricing strategy to explain")
):
    """
    Get explanation for a product's price recommendation.
    """
    try:
        # Ensure components are initialized
        if price_optimizer is None:
            price_optimizer = PriceOptimizer()
        if explainer is None:
            explainer = ShapExplainer()
        if data_loader is None:
            data_loader = DataLoader()
            
        # Get product features
        features = data_loader.get_product_features(product_id)
        if not features:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product {product_id} not found"
            )
            
        # Get explanation
        explanation = explainer.explain_price_recommendation(
            price_optimizer, product_id, features, strategy=strategy
        )
        
        if not explanation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Explanation for product {product_id} not available"
            )
            
        logger.info(f"Generated explanation for product {product_id}")
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@app.get("/competitors/{product_id}", tags=["Data"])
async def get_competitor_prices(product_id: str):
    """
    Get competitor prices for a product.
    """
    try:
        # Ensure data loader is initialized
        if data_loader is None:
            data_loader = DataLoader()
            
        # Get competitor prices
        competitor_prices = data_loader.get_competitor_prices(product_id)
        if not competitor_prices:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Competitor prices for product {product_id} not found"
            )
            
        logger.info(f"Retrieved competitor prices for product {product_id}")
        return {
            "product_id": product_id,
            "competitor_prices": competitor_prices,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving competitor prices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve competitor prices: {str(e)}"
        )


@app.get("/history/{product_id}", tags=["Data"])
async def get_price_history(
    product_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get price history for a product.
    """
    try:
        # Ensure data loader is initialized
        if data_loader is None:
            data_loader = DataLoader()
            
        # Get price history
        price_history = data_loader.get_price_history(product_id, start_date, end_date)
        if price_history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Price history for product {product_id} not found"
            )
            
        logger.info(f"Retrieved price history for product {product_id}")
        return {
            "product_id": product_id,
            "price_history": price_history,
            "start_date": start_date,
            "end_date": end_date,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving price history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve price history: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run FastAPI with uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
