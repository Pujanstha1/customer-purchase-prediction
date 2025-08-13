# Customer Repeat Purchase Predictor

This project predicts if a customer will make another purchase within the next 30 days.  
It’s a complete machine learning workflow — from cleaning and preparing the data, training and testing models, to making predictions available through an easy-to-use API.

---

## Table of Contents
- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Results](#results)
- [Getting Started](#getting-started)
- [Using the API](#using-the-api)
- [Business Insights](#business-insights)
- [Future Improvements](#future-improvements)

---

## Project Overview

### Goal
Help businesses:
- Identify loyal customers who are likely to buy again.
- Focus marketing on the right people.
- Reduce customer loss by acting at the right time.

### Data
- **What’s included**: Customer transactions and purchase history.
- **Features**: Demographics, buying patterns, timing, and preferences.
- **Target**: 1 = will buy again, 0 = won’t.
- **Time window**: Predict activity in the next 30 days.

### Highlights
- **Scalable data handling** using PySpark.
- **Tested multiple models** for best results.
- **Measured performance** with F1-score and ROC-AUC.
- **Deployed** using FastAPI with a simple web form.
- **Visualized** results for insights.

---

## How It Works

### 1. Data Preparation
We start with raw details:
- ID, order date, amount, category, payment type, location, days since last purchase.

Then create new features:
- Average order value
- Purchase frequency
- Total lifetime value
- Days since first purchase
- Spending rate
- Patterns by month/day/weekend/holiday

Categories like product type and payment method are encoded so the models can understand them.

### 2. Model Training
We used:
1. **Logistic Regression** — simple and explainable.
2. **Random Forest** — great for feature importance.
3. **XGBoost** — strong boosting algorithm.

Data was split 80% for training, 20% for testing.  
We used 5-fold cross-validation, handled imbalanced classes, and scaled features only when needed.

### 3. Evaluation
We checked:
- **F1-score** for balanced accuracy.
- **ROC-AUC** for probability ranking.
- Confusion matrices to see correct/incorrect predictions.
- Feature importance to understand what drives purchases.

### 4. Deployment
The best model is packaged with:
- **FastAPI** backend for predictions.
- **Bootstrap** web form for easy use.
- **Docker** for running anywhere.
- Endpoints for single or bulk predictions.

---

## Results

### Model Comparison

| Model | F1 | ROC-AUC | Status |
|-------|----|--------|--------|
| **Random Forest** | **0.6286** | 0.5085 | **Chosen** |
| Logistic Regression | 0.5484 | **0.5455** | Runner-up |
| XGBoost | 0.5312 | 0.5390 | Baseline |

**Why Random Forest?**  
It gave the best F1-score, finding more repeat customers without too many false positives.

**Random Forest Highlights**:
- Found **71%** of repeat customers.
- Precision: **56%**.
- Missed 9 repeat customers and wrongly targeted 10 non-repeat customers — acceptable for the business goal.

**Top 5 Important Features**:
1. Average order value  
2. Purchase frequency  
3. Total lifetime value  
4. Recency score (how recent the last purchase was)  
5. Customer tenure (how long they’ve been a customer)  

---

## Getting Started

### Requirements
- Docker & Docker Compose (recommended)
- Python 3.9+
- Git

### Quick Start with Docker
```bash
git clone <repository-url>
cd customer-repeat-purchase-predictor
docker-compose up --build
```
Then open:
- Web app: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### Local Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Using the API

**Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{ "avg_order_value": 25000.50, "purchase_frequency": 3, ... }'
```

**Batch Prediction (Python)**
```python
import requests
response = requests.post("http://localhost:8000/predict/batch", json=batch_data)
print(response.json())
```

Or simply use the web form at `http://localhost:8000`.

---

## Business Insights

- High spenders (>₹30,000 avg order value) return more often.
- Customers with 3+ past purchases are more loyal.
- Groceries have the highest repeat rate; electronics the lowest.
- Weekend and holiday purchases are linked to more repeats.

**Action Plans**:
- High probability customers: reward programs and premium offers.
- Medium probability: targeted campaigns.
- Low probability: strong win-back incentives.

---

## Future Improvements

### Model
- Try neural networks and time series models.
- Add features like customer similarity and seasonal trends.
- Enable real-time learning and A/B testing.

### Tech
- Deploy on Kubernetes for scaling.
- Add live monitoring with dashboards.
- Set up automated retraining.

---

**Version**: 1.0  
**Last Updated**: August 2025 

