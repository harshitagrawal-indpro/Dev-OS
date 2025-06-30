#!/usr/bin/env python3
"""
AI DevLab OS Demo Backend - Real Training with Showcase Data
"""

import os
import io
import json
import uuid
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Core FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Check and import ML libraries with fallbacks
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
    HAS_NUMPY = True
except ImportError:
    print("Warning: pandas/numpy not installed. Using demo mode.")
    HAS_PANDAS = False
    HAS_NUMPY = False

try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not installed. Using mock training.")
    HAS_SKLEARN = False

# Initialize FastAPI app
app = FastAPI(
    title="AI DevLab OS Demo API",
    description="Real-time AI model training demonstration platform",
    version="2.0.0",
    docs_url="/docs"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
STORAGE_DIR = Path("demo_storage")
MODELS_DIR = STORAGE_DIR / "models"
for directory in [STORAGE_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# In-memory storage
datasets_store = {}
models_store = {}
training_jobs_store = {}

# Pydantic models
class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str
    model_type: str = "RandomForest"
    test_size: float = 0.2

class PredictionRequest(BaseModel):
    model_id: str
    data: Dict[str, Any]

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

def create_demo_datasets():
    """Create realistic demo datasets for showcase"""
    demo_datasets = {}
    
    if HAS_SKLEARN and HAS_PANDAS:
        # 1. Customer Purchase Prediction (Classification)
        np.random.seed(42)
        X_customers, y_customers = make_classification(
            n_samples=1000,
            n_features=8,
            n_informative=6,
            n_redundant=2,
            n_clusters_per_class=1,
            class_sep=1.2,
            random_state=42
        )
        
        customer_features = [
            'age', 'income', 'spending_score', 'loyalty_years',
            'website_visits', 'email_opens', 'social_engagement', 'support_calls'
        ]
        
        customer_df = pd.DataFrame(X_customers, columns=customer_features)
        customer_df['age'] = (customer_df['age'] * 20 + 35).round().astype(int)
        customer_df['income'] = (customer_df['income'] * 25000 + 50000).round().astype(int)
        customer_df['spending_score'] = (customer_df['spending_score'] * 30 + 50).round(1)
        customer_df['loyalty_years'] = (customer_df['loyalty_years'] * 3 + 2).round(1)
        customer_df['website_visits'] = (customer_df['website_visits'] * 50 + 20).round().astype(int)
        customer_df['email_opens'] = (customer_df['email_opens'] * 15 + 5).round().astype(int)
        customer_df['social_engagement'] = (customer_df['social_engagement'] * 100 + 50).round().astype(int)
        customer_df['support_calls'] = (customer_df['support_calls'] * 3 + 1).round().astype(int)
        customer_df['will_purchase'] = y_customers
        
        customer_id = "customer_purchase_prediction"
        demo_datasets[customer_id] = {
            'dataset_id': customer_id,
            'filename': 'customer_purchase_prediction.csv',
            'shape': customer_df.shape,
            'columns': customer_df.columns.tolist(),
            'preview': customer_df.head(10).to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in customer_df.dtypes.items()},
            'dataframe': customer_df,
            'description': '🛍️ Customer Purchase Prediction - E-commerce behavioral data',
            'task_type': 'classification',
            'target': 'will_purchase',
            'features': customer_features
        }

        # 2. House Price Prediction (Regression)
        X_houses, y_houses = make_regression(
            n_samples=800,
            n_features=6,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        
        house_features = [
            'size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'location_score', 'garage_size'
        ]
        
        house_df = pd.DataFrame(X_houses, columns=house_features)
        house_df['size_sqft'] = (house_df['size_sqft'] * 800 + 1500).round().astype(int)
        house_df['bedrooms'] = (house_df['bedrooms'] * 2 + 3).round().astype(int)
        house_df['bathrooms'] = (house_df['bathrooms'] * 1.5 + 2).round(1)
        house_df['age_years'] = (abs(house_df['age_years']) * 15 + 5).round().astype(int)
        house_df['location_score'] = (house_df['location_score'] * 3 + 7).round(1)
        house_df['garage_size'] = (house_df['garage_size'] * 1 + 2).round().astype(int)
        house_df['price'] = (y_houses * 100000 + 300000).round().astype(int)
        
        house_id = "house_price_prediction"
        demo_datasets[house_id] = {
            'dataset_id': house_id,
            'filename': 'house_price_prediction.csv',
            'shape': house_df.shape,
            'columns': house_df.columns.tolist(),
            'preview': house_df.head(10).to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in house_df.dtypes.items()},
            'dataframe': house_df,
            'description': '🏠 House Price Prediction - Real estate market data',
            'task_type': 'regression',
            'target': 'price',
            'features': house_features
        }

        # 3. Employee Performance Classification
        X_emp, y_emp = make_classification(
            n_samples=600,
            n_features=7,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            class_sep=1.0,
            random_state=42
        )
        
        emp_features = [
            'experience_years', 'education_level', 'training_hours', 'projects_completed',
            'team_size', 'overtime_hours', 'satisfaction_score'
        ]
        
        emp_df = pd.DataFrame(X_emp, columns=emp_features)
        emp_df['experience_years'] = (emp_df['experience_years'] * 5 + 3).round(1)
        emp_df['education_level'] = (emp_df['education_level'] * 2 + 3).round().astype(int)
        emp_df['training_hours'] = (emp_df['training_hours'] * 50 + 20).round().astype(int)
        emp_df['projects_completed'] = (emp_df['projects_completed'] * 10 + 5).round().astype(int)
        emp_df['team_size'] = (emp_df['team_size'] * 5 + 5).round().astype(int)
        emp_df['overtime_hours'] = (abs(emp_df['overtime_hours']) * 20 + 5).round().astype(int)
        emp_df['satisfaction_score'] = (emp_df['satisfaction_score'] * 2 + 7).round(1)
        emp_df['performance_rating'] = y_emp  # 0: Low, 1: Medium, 2: High
        
        emp_id = "employee_performance"
        demo_datasets[emp_id] = {
            'dataset_id': emp_id,
            'filename': 'employee_performance.csv',
            'shape': emp_df.shape,
            'columns': emp_df.columns.tolist(),
            'preview': emp_df.head(10).to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in emp_df.dtypes.items()},
            'dataframe': emp_df,
            'description': '👥 Employee Performance Classification - HR analytics data',
            'task_type': 'classification',
            'target': 'performance_rating',
            'features': emp_features
        }

    return demo_datasets

# Create demo data on startup
@app.on_event("startup")
async def startup_event():
    global datasets_store
    datasets_store.update(create_demo_datasets())
    print(f"🚀 Demo backend started with {len(datasets_store)} datasets")
    print("📊 Available datasets:")
    for dataset_id, info in datasets_store.items():
        print(f"   - {info['description']}")

# API Routes
@app.get("/")
async def root():
    return {
        "message": "AI DevLab OS Demo Backend",
        "version": "2.0.0",
        "status": "running",
        "demo_datasets": len(datasets_store),
        "features": {
            "real_time_training": True,
            "live_progress": True,
            "multiple_algorithms": True,
            "instant_predictions": True
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "demo_mode": True,
        "ml_libraries": {
            "pandas": HAS_PANDAS,
            "sklearn": HAS_SKLEARN,
            "numpy": HAS_NUMPY
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/datasets")
async def get_datasets():
    datasets = []
    for dataset_id, info in datasets_store.items():
        dataset_info = {k: v for k, v in info.items() if k != 'dataframe'}
        datasets.append(dataset_info)
    return {"datasets": datasets}

@app.get("/api/dataset/{dataset_id}")
async def get_dataset_details(dataset_id: str):
    if dataset_id not in datasets_store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    info = datasets_store[dataset_id]
    return {k: v for k, v in info.items() if k != 'dataframe'}

@app.post("/api/train-model")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    if request.dataset_id not in datasets_store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    job_id = str(uuid.uuid4())
    dataset_info = datasets_store[request.dataset_id]
    
    job_info = {
        'job_id': job_id,
        'status': 'training',
        'progress': 0,
        'accuracy': None,
        'model_name': f"{request.model_type}_{dataset_info['description'].split(' ')[0]}",
        'dataset_name': dataset_info['filename'],
        'created_at': datetime.now().isoformat(),
        'algorithm': request.model_type,
        'task_type': request.task_type,
        'logs': []
    }
    
    training_jobs_store[job_id] = job_info
    
    # Start real training
    background_tasks.add_task(real_time_training, job_id, request)
    
    await manager.broadcast(json.dumps({
        'type': 'training_started',
        'data': job_info
    }))
    
    return {"job_id": job_id, "status": "training", "message": "Real-time training started!"}

async def real_time_training(job_id: str, request: TrainingRequest):
    """Real machine learning training with live progress updates"""
    try:
        job = training_jobs_store[job_id]
        dataset_info = datasets_store[request.dataset_id]
        
        # Phase 1: Data Loading
        await update_progress(job_id, 10, "Loading dataset...")
        await asyncio.sleep(1)
        
        if not HAS_SKLEARN or not HAS_PANDAS:
            # Fallback to mock training
            await mock_training_progress(job_id)
            return
        
        # Phase 2: Data Preparation
        await update_progress(job_id, 25, "Preparing data...")
        df = dataset_info['dataframe'].copy()
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42
        )
        await asyncio.sleep(1)
        
        # Phase 3: Model Selection
        await update_progress(job_id, 40, f"Initializing {request.model_type}...")
        
        if request.task_type == 'classification':
            if request.model_type == 'RandomForest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif request.model_type == 'LogisticRegression':
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif request.model_type == 'SVM':
                model = SVC(random_state=42, probability=True)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            if request.model_type == 'RandomForest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif request.model_type == 'LinearRegression':
                model = LinearRegression()
            elif request.model_type == 'SVR':
                model = SVR()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        await asyncio.sleep(1)
        
        # Phase 4: Training
        await update_progress(job_id, 60, "Training model...")
        await asyncio.sleep(2)  # Simulate training time
        
        model.fit(X_train, y_train)
        
        # Phase 5: Evaluation
        await update_progress(job_id, 85, "Evaluating model...")
        predictions = model.predict(X_test)
        
        if request.task_type == 'classification':
            accuracy = accuracy_score(y_test, predictions)
            metric_name = "Accuracy"
        else:
            accuracy = r2_score(y_test, predictions)
            metric_name = "R² Score"
        
        await asyncio.sleep(1)
        
        # Phase 6: Save Model
        await update_progress(job_id, 95, "Saving model...")
        model_id = str(uuid.uuid4())
        model_path = MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        
        # Store model info
        model_info = {
            'model_id': model_id,
            'model_name': job['model_name'],
            'accuracy': accuracy,
            'metric_name': metric_name,
            'task_type': request.task_type,
            'algorithm': request.model_type,
            'target_column': request.target_column,
            'feature_columns': X.columns.tolist(),
            'dataset_id': request.dataset_id,
            'created_at': datetime.now().isoformat(),
            'status': 'completed'
        }
        models_store[model_id] = model_info
        
        # Final update
        job['status'] = 'completed'
        job['accuracy'] = accuracy
        job['model_id'] = model_id
        job['logs'].append(f"Training completed! {metric_name}: {accuracy:.3f}")
        
        await update_progress(job_id, 100, f"Complete! {metric_name}: {accuracy:.1%}")
        
        await manager.broadcast(json.dumps({
            'type': 'training_completed',
            'data': {
                'job_id': job_id,
                'accuracy': accuracy,
                'model_id': model_id,
                'metric_name': metric_name
            }
        }))
        
    except Exception as e:
        job = training_jobs_store.get(job_id, {})
        job['status'] = 'failed'
        job['error'] = str(e)
        await manager.broadcast(json.dumps({
            'type': 'training_failed',
            'data': {'job_id': job_id, 'error': str(e)}
        }))

async def mock_training_progress(job_id: str):
    """Fallback mock training for demo"""
    for progress in [10, 25, 40, 60, 85, 100]:
        await update_progress(job_id, progress, f"Mock training: {progress}%")
        await asyncio.sleep(1.5)
    
    job = training_jobs_store[job_id]
    job['status'] = 'completed'
    job['accuracy'] = 0.94
    
    await manager.broadcast(json.dumps({
        'type': 'training_completed',
        'data': {'job_id': job_id, 'accuracy': 0.94}
    }))

async def update_progress(job_id: str, progress: int, message: str):
    """Update training progress and broadcast to clients"""
    if job_id in training_jobs_store:
        job = training_jobs_store[job_id]
        job['progress'] = progress
        job['logs'].append(message)
        
        await manager.broadcast(json.dumps({
            'type': 'training_progress',
            'data': {
                'job_id': job_id,
                'progress': progress,
                'message': message
            }
        }))

@app.get("/api/training-jobs")
async def get_training_jobs():
    return {"jobs": list(training_jobs_store.values())}

@app.get("/api/models")
async def get_models():
    return {"models": list(models_store.values())}

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    if request.model_id not in models_store:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models_store[request.model_id]
    model_path = MODELS_DIR / f"{request.model_id}.joblib"
    
    try:
        if model_path.exists() and HAS_SKLEARN:
            model = joblib.load(model_path)
            feature_values = [request.data.get(col, 0) for col in model_info['feature_columns']]
            prediction = model.predict([feature_values])[0]
            
            confidence = None
            if hasattr(model, 'predict_proba') and model_info['task_type'] == 'classification':
                probabilities = model.predict_proba([feature_values])[0]
                confidence = float(max(probabilities))
        else:
            # Mock prediction
            prediction = np.random.choice([0, 1]) if model_info['task_type'] == 'classification' else np.random.uniform(100, 500)
            confidence = 0.92
        
        return {
            "prediction": float(prediction),
            "confidence": confidence,
            "model_name": model_info['model_name'],
            "algorithm": model_info['algorithm'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/demo/sample-prediction/{dataset_id}")
async def get_sample_prediction_data(dataset_id: str):
    """Get sample data for testing predictions"""
    if dataset_id not in datasets_store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_info = datasets_store[dataset_id]
    features = dataset_info.get('features', [])
    
    if HAS_PANDAS and 'dataframe' in dataset_info:
        # Get a random sample from the dataset
        df = dataset_info['dataframe']
        sample_row = df[features].sample(1).iloc[0]
        sample_data = sample_row.to_dict()
    else:
        # Generate mock sample data
        sample_data = {feature: round(np.random.uniform(1, 100), 2) for feature in features}
    
    return {
        "sample_data": sample_data,
        "features": features,
        "dataset_name": dataset_info['filename']
    }

if __name__ == "__main__":
    print("🚀 Starting AI DevLab OS Demo Backend...")
    print("📊 Features:")
    print("   ✅ Real-time ML training")
    print("   ✅ Live progress updates via WebSocket")
    print("   ✅ Multiple algorithms (RandomForest, LogisticRegression, SVM)")
    print("   ✅ Instant predictions")
    print("   ✅ Demo datasets with realistic data")
    print("")
    print("🌐 Access points:")
    print("   - API: http://localhost:8000")
    print("   - Docs: http://localhost:8000/docs")
    print("   - WebSocket: ws://localhost:8000/ws")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)