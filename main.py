
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import json
import os

app = FastAPI(
    title="Customer Repeat Purchase Predictor",
    description="ML API with Web Interface for predicting customer repeat purchases",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
model = None
model_metadata = None

@app.on_event("startup")
async def load_model():
    global model, model_metadata
    
    try:
        with open("models/rf_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        with open("models/model_metadata.json", 'r') as f:
            model_metadata = json.load(f)
        
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

class CustomerFeatures(BaseModel):
    """Input schema for customer features"""
    # Core numerical features
    avg_order_value: float
    purchase_frequency: int
    total_lifetime_value: float
    customer_tenure_days: int
    recency_score: float
    spending_velocity: float
    order_value_coefficient_variation: float = 0.0
    last_order_month: int
    last_order_day_of_week: int
    is_weekend: int = 0
    is_holiday_season: int = 0
    preferred_category_encoded: int = 1
    preferred_payment_type_encoded: int = 1
    preferred_delivery_location_encoded: int = 1
    
    # Category dummy features (one-hot encoded)
    category_groceries: int = 0
    category_electronics: int = 0
    category_clothing: int = 0
    category_home_appliances: int = 0
    category_stationery: int = 0
    
    # Payment dummy features (one-hot encoded)
    payment_credit_card: int = 0
    payment_digital_wallet: int = 0
    payment_bank_transfer: int = 0
    payment_cash: int = 0
    
    # Location dummy features (one-hot encoded)
    location_lalitpur: int = 0
    location_biratnagar: int = 0
    location_bhaktapur: int = 0

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model/features")
async def get_model_features():
    """Get the expected feature list for the model"""
    if model_metadata and "features" in model_metadata:
        return {
            "expected_features": model_metadata["features"],
            "feature_count": len(model_metadata["features"])
        }
    else:
        # Return the feature order we expect based on our implementation
        expected_features = [
            "avg_order_value", "purchase_frequency", "total_lifetime_value",
            "customer_tenure_days", "recency_score", "spending_velocity",
            "order_value_coefficient_variation", "last_order_month", 
            "last_order_day_of_week", "is_weekend", "is_holiday_season",
            "preferred_category_encoded", "preferred_payment_type_encoded",
            "preferred_delivery_location_encoded",
            "category_groceries", "category_electronics", "category_clothing",
            "category_home_appliances", "category_stationery",
            "payment_credit_card", "payment_digital_wallet", 
            "payment_bank_transfer", "payment_cash",
            "location_lalitpur", "location_biratnagar", "location_bhaktapur"
        ]
        return {
            "expected_features": expected_features,
            "feature_count": len(expected_features),
            "note": "Features inferred from model implementation"
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "metadata_loaded": model_metadata is not None
    }

def create_full_feature_vector(feature_dict):
    """Create complete feature vector with all dummy variables"""
    
    # Start with numerical features
    full_features = {
        "avg_order_value": feature_dict.get("avg_order_value", 0),
        "purchase_frequency": feature_dict.get("purchase_frequency", 0),
        "total_lifetime_value": feature_dict.get("total_lifetime_value", 0),
        "customer_tenure_days": feature_dict.get("customer_tenure_days", 0),
        "recency_score": feature_dict.get("recency_score", 0),
        "spending_velocity": feature_dict.get("spending_velocity", 0),
        "order_value_coefficient_variation": feature_dict.get("order_value_coefficient_variation", 0),
        "last_order_month": feature_dict.get("last_order_month", 1),
        "last_order_day_of_week": feature_dict.get("last_order_day_of_week", 1),
        "is_weekend": feature_dict.get("is_weekend", 0),
        "is_holiday_season": feature_dict.get("is_holiday_season", 0),
        "preferred_category_encoded": feature_dict.get("preferred_category_encoded", 1),
        "preferred_payment_type_encoded": feature_dict.get("preferred_payment_type_encoded", 1),
        "preferred_delivery_location_encoded": feature_dict.get("preferred_delivery_location_encoded", 1)
    }
    
    # Add all category dummy variables
    categories = ["groceries", "electronics", "clothing", "home_appliances", "stationery"]
    for cat in categories:
        full_features[f"category_{cat}"] = feature_dict.get(f"category_{cat}", 0)
    
    # Add all payment dummy variables  
    payments = ["credit_card", "digital_wallet", "bank_transfer", "cash"]
    for payment in payments:
        full_features[f"payment_{payment}"] = feature_dict.get(f"payment_{payment}", 0)
    
    # Add all location dummy variables
    locations = ["lalitpur", "biratnagar", "bhaktapur"]
    for location in locations:
        full_features[f"location_{location}"] = feature_dict.get(f"location_{location}", 0)
    
    return full_features

@app.post("/predict")
async def predict_api(features: CustomerFeatures, customer_id: str = "unknown"):
    """API endpoint for predictions"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to feature dict
        feature_dict = features.dict()
        
        # Create complete feature vector with all dummy variables
        full_features = create_full_feature_vector(feature_dict)
        
        # Create DataFrame
        input_df = pd.DataFrame([full_features])
        
        # Ensure feature order matches training (if model_metadata available)
        if model_metadata and "features" in model_metadata:
            expected_features = model_metadata["features"]
            # Reorder columns to match training order
            input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        if probability >= 0.7:
            confidence = "high"
        elif probability >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "customer_id": customer_id,
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "confidence": confidence,
            "model_version": "1.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    customer_id: str = Form(...),
    avg_order_value: float = Form(...),
    purchase_frequency: int = Form(...),
    total_lifetime_value: float = Form(...),
    customer_tenure_days: int = Form(...),
    recency_score: float = Form(...),
    spending_velocity: float = Form(...),
    last_order_month: int = Form(...),
    last_order_day_of_week: int = Form(...),
    is_weekend: int = Form(0),
    is_holiday_season: int = Form(0),
    preferred_category: str = Form("groceries"),
    preferred_payment: str = Form("credit_card"),
    preferred_location: str = Form("lalitpur")
):
    """Handle form submission and return results page"""
    
    if model is None:
        return templates.TemplateResponse("error.html", {
            "request": request, 
            "error": "Model not loaded"
        })
    
    try:
        # Map categorical values to encoded numbers
        category_mapping = {
            "groceries": 1, "electronics": 2, "clothing": 3, 
            "home_appliances": 4, "stationery": 5
        }
        payment_mapping = {
            "credit_card": 1, "digital_wallet": 2, 
            "bank_transfer": 3, "cash": 4
        }
        location_mapping = {
            "lalitpur": 1, "biratnagar": 2, "bhaktapur": 3
        }
        
        # Create base feature dictionary
        feature_dict = {
            "avg_order_value": avg_order_value,
            "purchase_frequency": purchase_frequency,
            "total_lifetime_value": total_lifetime_value,
            "customer_tenure_days": customer_tenure_days,
            "recency_score": recency_score,
            "spending_velocity": spending_velocity,
            "order_value_coefficient_variation": 0.0,
            "last_order_month": last_order_month,
            "last_order_day_of_week": last_order_day_of_week,
            "is_weekend": is_weekend,
            "is_holiday_season": is_holiday_season,
            "preferred_category_encoded": category_mapping.get(preferred_category, 1),
            "preferred_payment_type_encoded": payment_mapping.get(preferred_payment, 1),
            "preferred_delivery_location_encoded": location_mapping.get(preferred_location, 1)
        }
        
        # Add one-hot encoded features for categories
        categories = ["groceries", "electronics", "clothing", "home_appliances", "stationery"]
        for cat in categories:
            feature_dict[f"category_{cat}"] = 1 if preferred_category == cat else 0
        
        # Add one-hot encoded features for payment types
        payments = ["credit_card", "digital_wallet", "bank_transfer", "cash"]
        for payment in payments:
            feature_dict[f"payment_{payment}"] = 1 if preferred_payment == payment else 0
        
        # Add one-hot encoded features for locations
        locations = ["lalitpur", "biratnagar", "bhaktapur"]
        for location in locations:
            feature_dict[f"location_{location}"] = 1 if preferred_location == location else 0
        
        # Create complete feature vector
        full_features = create_full_feature_vector(feature_dict)
        
        # Create DataFrame
        input_df = pd.DataFrame([full_features])
        
        # Ensure feature order matches training
        if model_metadata and "features" in model_metadata:
            expected_features = model_metadata["features"]
            input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        if probability >= 0.7:
            confidence = "High"
            confidence_color = "success"
        elif probability >= 0.4:
            confidence = "Medium"
            confidence_color = "warning"
        else:
            confidence = "Low"
            confidence_color = "danger"
        
        result = {
            "customer_id": customer_id,
            "prediction": int(prediction),
            "prediction_text": "Will Repeat Purchase" if prediction == 1 else "Will Not Repeat Purchase",
            "probability": round(float(probability), 4),
            "probability_percent": round(float(probability) * 100, 1),
            "confidence": confidence,
            "confidence_color": confidence_color,
            "input_data": {
                "avg_order_value": avg_order_value,
                "purchase_frequency": purchase_frequency,
                "total_lifetime_value": total_lifetime_value,
                "customer_tenure_days": customer_tenure_days,
                "preferred_category": preferred_category.replace("_", " ").title(),
                "preferred_payment": preferred_payment.replace("_", " ").title(),
                "preferred_location": preferred_location.title()
            }
        }
        
        return templates.TemplateResponse("result.html", {
            "request": request, 
            "result": result
        })
        
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request, 
            "error": f"Prediction error: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)