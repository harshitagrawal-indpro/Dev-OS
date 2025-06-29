#!/usr/bin/env python3
"""
AI DevLab OS Backend Server
Complete FastAPI backend for AI development platform
"""

import os
import io
import json
import uuid
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import traceback
import base64

# Core FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Check and import optional ML libraries with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas not installed. Dataset functionality will be limited.")
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("Warning: numpy not installed. Numerical operations will be limited.")
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Plotting features will be disabled.")
    HAS_PLOTTING = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib
    HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not installed. ML training features will be disabled.")
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not installed. XGBoost algorithms will be unavailable.")
    HAS_XGBOOST = False

# Initialize FastAPI app
app = FastAPI(
    title="AI DevLab OS API",
    description="Complete AI development platform backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create storage directories
STORAGE_DIR = Path("storage")
MODELS_DIR = STORAGE_DIR / "models"
DATASETS_DIR = STORAGE_DIR / "datasets"
TEMP_DIR = STORAGE_DIR / "temp"

for directory in [STORAGE_DIR, MODELS_DIR, DATASETS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# In-memory storage (replace with database in production)
datasets_store = {}
models_store = {}
training_jobs_store = {}

# Pydantic models for API
class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    shape: tuple
    columns: List[str]
    preview: List[Dict[str, Any]]
    dtypes: Dict[str, str]

class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str
    test_size: float = 0.2
    algorithms: List[str] = ["RandomForest", "XGBoost", "LogisticRegression"]
    cv_folds: int = 5
    max_time: int = 3600

class PredictionRequest(BaseModel):
    model_id: str
    data: Dict[str, Any]

class CodeExecutionRequest(BaseModel):
    code: str
    dataset_id: Optional[str] = None

# Helper functions
def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling numpy types"""
    if HAS_NUMPY and isinstance(obj, np.integer):
        return int(obj)
    elif HAS_NUMPY and isinstance(obj, np.floating):
        return float(obj)
    elif HAS_NUMPY and isinstance(obj, np.ndarray):
        return obj.tolist()
    elif HAS_PANDAS and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj

def create_sample_data(rows=1000):
    """Create sample dataset when pandas is not available"""
    if not HAS_PANDAS or not HAS_NUMPY:
        return {
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'tenure': [12, 24, 36, 48, 60],
            'satisfaction_score': [7.5, 8.0, 6.5, 9.0, 7.0],
            'churn': [0, 0, 1, 0, 1]
        }
    
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, rows),
        'income': np.random.normal(50000, 20000, rows),
        'tenure': np.random.randint(1, 60, rows),
        'satisfaction_score': np.random.uniform(1, 10, rows),
        'monthly_charges': np.random.uniform(30, 120, rows)
    }
    
    # Create target based on logic
    data['churn'] = (
        (data['satisfaction_score'] < 5) | 
        (data['monthly_charges'] > 100) | 
        (data['tenure'] < 12)
    ).astype(int)
    
    return data

# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI DevLab OS Backend API",
        "version": "1.0.0",
        "status": "running",
        "features": {
            "pandas": HAS_PANDAS,
            "sklearn": HAS_SKLEARN,
            "xgboost": HAS_XGBOOST,
            "plotting": HAS_PLOTTING
        },
        "endpoints": {
            "datasets": "/api/datasets",
            "models": "/api/models",
            "training": "/api/train-model",
            "prediction": "/api/predict",
            "docs": "/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "pandas": HAS_PANDAS,
            "scikit-learn": HAS_SKLEARN,
            "xgboost": HAS_XGBOOST,
            "matplotlib": HAS_PLOTTING
        }
    }

@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and analyze a dataset"""
    try:
        if not HAS_PANDAS:
            raise HTTPException(
                status_code=501, 
                detail="Dataset upload requires pandas. Please install: pip install pandas"
            )
        
        # Read file content
        content = await file.read()
        dataset_id = str(uuid.uuid4())
        
        # Parse file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        # Store dataset info
        dataset_info = {
            'dataset_id': dataset_id,
            'filename': file.filename,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'preview': df.head(10).to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'dataframe': df  # Store actual dataframe
        }
        
        datasets_store[dataset_id] = dataset_info
        
        # Return response without dataframe (not JSON serializable)
        response_data = {k: v for k, v in dataset_info.items() if k != 'dataframe'}
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@app.get("/api/datasets")
async def get_datasets():
    """Get list of uploaded datasets"""
    datasets = []
    for dataset_id, info in datasets_store.items():
        dataset_summary = {k: v for k, v in info.items() if k != 'dataframe'}
        datasets.append(dataset_summary)
    
    return {"datasets": datasets}

@app.get("/api/dataset/{dataset_id}/eda")
async def perform_eda(dataset_id: str):
    """Perform Exploratory Data Analysis on dataset"""
    try:
        if dataset_id not in datasets_store:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if not HAS_PANDAS:
            return {
                "statistics": {"message": "EDA requires pandas"},
                "charts": [],
                "insights": ["Install pandas for full EDA functionality"],
                "data_quality_score": 0
            }
        
        df = datasets_store[dataset_id]['dataframe']
        
        # Basic statistics
        statistics = {
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
        }
        
        charts = []
        insights = []
        
        # Generate some insights
        missing_pct = (df.isnull().sum() / len(df) * 100).max()
        if missing_pct > 20:
            insights.append(f"High missing data detected: {missing_pct:.1f}% in some columns")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Dataset contains {len(numeric_cols)} numeric features for modeling")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Found {len(categorical_cols)} categorical features that may need encoding")
        
        # Calculate data quality score
        quality_score = max(0, 100 - missing_pct - (len(categorical_cols) * 5))
        
        return {
            "statistics": statistics,
            "charts": charts,
            "insights": insights,
            "data_quality_score": quality_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing EDA: {str(e)}")

@app.post("/api/train-model")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train machine learning model"""
    try:
        if not HAS_SKLEARN:
            raise HTTPException(
                status_code=501,
                detail="Model training requires scikit-learn. Please install: pip install scikit-learn"
            )
        
        if request.dataset_id not in datasets_store:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        job_id = str(uuid.uuid4())
        
        # Create training job
        job_info = {
            'job_id': job_id,
            'status': 'training',
            'progress': 0,
            'accuracy': None,
            'model_name': f"Model_{job_id[:8]}",
            'dataset_name': datasets_store[request.dataset_id]['filename'],
            'created_at': datetime.now().isoformat(),
            'request': request.dict()
        }
        
        training_jobs_store[job_id] = job_info
        
        # Start training in background
        background_tasks.add_task(train_model_background, job_id, request)
        
        return {"job_id": job_id, "status": "training", "message": "Training started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

async def train_model_background(job_id: str, request: TrainingRequest):
    """Background task for model training"""
    try:
        if not HAS_PANDAS or not HAS_SKLEARN:
            training_jobs_store[job_id]['status'] = 'failed'
            training_jobs_store[job_id]['error'] = 'Missing required dependencies'
            return
        
        # Get dataset
        df = datasets_store[request.dataset_id]['dataframe']
        
        # Update progress
        training_jobs_store[job_id]['progress'] = 25
        
        # Prepare data
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]
        
        # Handle categorical variables (simple approach)
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42
        )
        
        training_jobs_store[job_id]['progress'] = 50
        
        # Train models
        best_model = None
        best_score = 0
        best_algorithm = None
        
        algorithms = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42) 
                          if request.task_type == 'classification' 
                          else RandomForestRegressor(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000) 
                                if request.task_type == 'classification' 
                                else LinearRegression()
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST and 'XGBoost' in request.algorithms:
            algorithms['XGBoost'] = xgb.XGBClassifier(random_state=42) \
                                  if request.task_type == 'classification' \
                                  else xgb.XGBRegressor(random_state=42)
        
        for i, (name, model) in enumerate(algorithms.items()):
            if name in request.algorithms:
                try:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    
                    if request.task_type == 'classification':
                        score = accuracy_score(y_test, predictions)
                    else:
                        score = 1 - (mean_squared_error(y_test, predictions) / y_test.var())
                        score = max(0, score)  # Ensure non-negative
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_algorithm = name
                        
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            # Update progress
            training_jobs_store[job_id]['progress'] = 50 + (i + 1) * 30 // len(algorithms)
        
        if best_model is None:
            training_jobs_store[job_id]['status'] = 'failed'
            training_jobs_store[job_id]['error'] = 'No models could be trained successfully'
            return
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(best_model, model_path)
        
        # Store model info
        model_info = {
            'model_id': model_id,
            'model_name': f"{best_algorithm}_{request.target_column}",
            'score': best_score,
            'task_type': request.task_type,
            'target_column': request.target_column,
            'algorithm': best_algorithm,
            'created_at': datetime.now().isoformat(),
            'dataset_id': request.dataset_id,
            'feature_columns': X.columns.tolist()
        }
        
        models_store[model_id] = model_info
        
        # Update job status
        training_jobs_store[job_id]['status'] = 'completed'
        training_jobs_store[job_id]['progress'] = 100
        training_jobs_store[job_id]['accuracy'] = best_score
        training_jobs_store[job_id]['model_id'] = model_id
        
    except Exception as e:
        training_jobs_store[job_id]['status'] = 'failed'
        training_jobs_store[job_id]['error'] = str(e)
        print(f"Training error: {e}")
        traceback.print_exc()

@app.get("/api/training-jobs")
async def get_training_jobs():
    """Get list of training jobs"""
    jobs = []
    for job_id, job_info in training_jobs_store.items():
        job_data = {k: v for k, v in job_info.items() if k != 'request'}
        jobs.append(job_data)
    
    return {"jobs": jobs}

@app.get("/api/models")
async def get_models():
    """Get list of trained models"""
    models = list(models_store.values())
    return {"models": models}

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Make prediction using trained model"""
    try:
        if not HAS_SKLEARN:
            raise HTTPException(
                status_code=501,
                detail="Prediction requires scikit-learn. Please install: pip install scikit-learn"
            )
        
        if request.model_id not in models_store:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = models_store[request.model_id]
        model_path = MODELS_DIR / f"{request.model_id}.joblib"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Load model
        model = joblib.load(model_path)
        
        # Prepare input data
        if HAS_PANDAS:
            input_df = pd.DataFrame([request.data])
            
            # Ensure all required columns are present
            for col in model_info['feature_columns']:
                if col not in input_df.columns:
                    input_df[col] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            input_df = input_df[model_info['feature_columns']]
            
            # Handle categorical variables (simple approach)
            for col in input_df.select_dtypes(include=['object']).columns:
                input_df[col] = pd.Categorical(input_df[col]).codes
            
            prediction = model.predict(input_df)
        else:
            # Fallback without pandas
            feature_values = [request.data.get(col, 0) for col in model_info['feature_columns']]
            prediction = model.predict([feature_values])
        
        # Get prediction probability if available
        try:
            if hasattr(model, 'predict_proba') and model_info['task_type'] == 'classification':
                probabilities = model.predict_proba(input_df if HAS_PANDAS else [feature_values])
                probability = float(max(probabilities[0]))
            else:
                probability = None
        except:
            probability = None
        
        result = {
            "prediction": safe_json_serialize(prediction[0]),
            "model_id": request.model_id,
            "model_name": model_info['model_name'],
            "task_type": model_info['task_type'],
            "confidence": probability,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/execute-code")
async def execute_code(request: CodeExecutionRequest):
    """Execute Python code in a safe environment"""
    try:
        # Create a temporary file for the code
        temp_file = TEMP_DIR / f"code_{uuid.uuid4()}.py"
        
        # Prepare code with dataset loading if requested
        code_to_execute = request.code
        
        if request.dataset_id and request.dataset_id in datasets_store:
            if HAS_PANDAS:
                # Add dataset loading code at the beginning
                dataset_info = datasets_store[request.dataset_id]
                df_code = f"""
# Auto-loaded dataset: {dataset_info['filename']}
import pandas as pd
import numpy as np

# Load the dataset
df = pd.DataFrame({repr(dataset_info['dataframe'].to_dict('list'))})
print(f"Dataset loaded: {dataset_info['filename']}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

"""
                code_to_execute = df_code + code_to_execute
        
        # Write code to temporary file
        with open(temp_file, 'w') as f:
            f.write(code_to_execute)
        
        # Execute the code using subprocess for safety
        try:
            result = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=str(TEMP_DIR)
            )
            
            output = result.stdout
            error = result.stderr
            return_code = result.returncode
            
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": "Code execution timed out (30 seconds limit)",
                "return_code": -1
            }
        
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
        
        return {
            "output": output,
            "error": error,
            "return_code": return_code
        }
        
    except Exception as e:
        return {
            "output": "",
            "error": f"Execution error: {str(e)}",
            "return_code": -1
        }

@app.delete("/api/model/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    try:
        if model_id not in models_store:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Delete model file
        model_path = MODELS_DIR / f"{model_id}.joblib"
        if model_path.exists():
            model_path.unlink()
        
        # Remove from store
        del models_store[model_id]
        
        return {"message": "Model deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

@app.get("/api/dataset/{dataset_id}")
async def get_dataset_details(dataset_id: str):
    """Get detailed information about a specific dataset"""
    if dataset_id not in datasets_store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_info = datasets_store[dataset_id]
    response_data = {k: v for k, v in dataset_info.items() if k != 'dataframe'}
    
    return response_data

# Create some sample data on startup if no datasets exist
@app.on_event("startup")
async def startup_event():
    """Initialize sample data on startup"""
    if not datasets_store and HAS_PANDAS:
        try:
            # Create sample dataset
            sample_data = create_sample_data(1000)
            df = pd.DataFrame(sample_data)
            
            dataset_id = "sample_dataset"
            dataset_info = {
                'dataset_id': dataset_id,
                'filename': 'sample_customer_data.csv',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'preview': df.head(10).to_dict('records'),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'dataframe': df
            }
            
            datasets_store[dataset_id] = dataset_info
            print("✓ Sample dataset created")
            
        except Exception as e:
            print(f"Warning: Could not create sample dataset: {e}")
    
    print("🚀 AI DevLab OS Backend started successfully!")
    print(f"📊 Available features:")
    print(f"   - Pandas: {'✓' if HAS_PANDAS else '✗'}")
    print(f"   - Scikit-learn: {'✓' if HAS_SKLEARN else '✗'}")
    print(f"   - XGBoost: {'✓' if HAS_XGBOOST else '✗'}")
    print(f"   - Plotting: {'✓' if HAS_PLOTTING else '✗'}")
    print()
    print("📝 API Documentation: http://localhost:8000/docs")
    print("🔗 API Base URL: http://localhost:8000")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Additional utility endpoints
@app.get("/api/system-info")
async def get_system_info():
    """Get system information and capabilities"""
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "dependencies": {
            "pandas": HAS_PANDAS,
            "numpy": HAS_NUMPY,
            "scikit-learn": HAS_SKLEARN,
            "xgboost": HAS_XGBOOST,
            "matplotlib": HAS_PLOTTING,
        },
        "storage": {
            "datasets": len(datasets_store),
            "models": len(models_store),
            "training_jobs": len(training_jobs_store)
        },
        "directories": {
            "storage": str(STORAGE_DIR),
            "models": str(MODELS_DIR),
            "datasets": str(DATASETS_DIR),
            "temp": str(TEMP_DIR)
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting AI DevLab OS Backend Server...")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )