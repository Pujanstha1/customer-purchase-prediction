import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        print("✅ Health Check: API is running")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Health Check Failed: {response.status_code}")

def test_prediction():
    """Test prediction endpoint"""
    test_data = {
        "avg_order_value": 25000.50,
        "purchase_frequency": 3,
        "total_lifetime_value": 75000.0,
        "customer_tenure_days": 90,
        "recency_score": 0.015,
        "spending_velocity": 833.33,
        "last_order_month": 7,
        "last_order_day_of_week": 3,
        "is_weekend": 0,
        "is_holiday_season": 0,
        "preferred_category_encoded": 1,
        "preferred_payment_type_encoded": 2,
        "preferred_delivery_location_encoded": 1
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        params={"customer_id": "TEST_CUSTOMER_001"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction Test: Success")
        print(f"Customer: {result['customer_id']}")
        print(f"Prediction: {result['prediction']} ({'Repeat Purchase' if result['prediction'] == 1 else 'No Repeat Purchase'})")
        print(f"Probability: {result['probability']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"❌ Prediction Test Failed: {response.status_code}")
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("🧪 Testing ML API...")
    test_health()
    print()
    test_prediction()
    print("\n🎉 Testing complete!")