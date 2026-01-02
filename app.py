import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import mlflow
import dagshub
import shutil
from networksecurity.pipeline.training_pipeline import TrainingPipeline

# ============================================================
# Configuration
# ============================================================

APP_TITLE = "Network Security - Phishing Detection"
APP_DESCRIPTION = "ML-powered phishing URL detection system"
APP_VERSION = "1.0.0"

# Project Links
GITHUB_URL = "https://github.com/Inder-26/NetworkSecurity"
DAGSHUB_URL = "https://dagshub.com/Inder-26/NetworkSecurity"
LINKEDIN_URL = "https://linkedin.com/in/inderjeet"  # Update with your actual LinkedIn

# Model Paths
MODEL_PATH = "final_model/model.pkl"
PREPROCESSOR_PATH = "final_model/preprocessor.pkl"

# Initialize DagsHub
if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
    try:
        dagshub.init(repo_owner="Inder-26", repo_name="NetworkSecurity", mlflow=True)
    except Exception as e:
        print(f"⚠️ Error initializing DagsHub: {e}")
else:
    print("⚠️ DagsHub credentials not found. Skipping initialization.")

# Feature Columns (30 features)
FEATURE_COLUMNS = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report"
]

# Feature descriptions for UI
FEATURE_INFO = {
    "having_IP_Address": {"group": "URL Structure", "desc": "IP address in URL"},
    "URL_Length": {"group": "URL Structure", "desc": "Length of URL"},
    "Shortining_Service": {"group": "URL Structure", "desc": "URL shortening service used"},
    "having_At_Symbol": {"group": "URL Structure", "desc": "@ symbol in URL"},
    "double_slash_redirecting": {"group": "URL Structure", "desc": "// redirecting"},
    "Prefix_Suffix": {"group": "URL Structure", "desc": "Prefix/suffix in domain"},
    "having_Sub_Domain": {"group": "URL Structure", "desc": "Subdomain presence"},
    "SSLfinal_State": {"group": "Security", "desc": "SSL certificate state"},
    "Domain_registeration_length": {"group": "Domain", "desc": "Domain registration length"},
    "Favicon": {"group": "Security", "desc": "Favicon loaded from external"},
    "port": {"group": "Security", "desc": "Non-standard port"},
    "HTTPS_token": {"group": "Security", "desc": "HTTPS token in domain"},
    "Request_URL": {"group": "Page Content", "desc": "External request URLs"},
    "URL_of_Anchor": {"group": "Page Content", "desc": "Anchor URL analysis"},
    "Links_in_tags": {"group": "Page Content", "desc": "Links in meta/script tags"},
    "SFH": {"group": "Page Content", "desc": "Server Form Handler"},
    "Submitting_to_email": {"group": "Page Content", "desc": "Form submits to email"},
    "Abnormal_URL": {"group": "URL Structure", "desc": "Abnormal URL pattern"},
    "Redirect": {"group": "Page Behavior", "desc": "Redirect count"},
    "on_mouseover": {"group": "Page Behavior", "desc": "onMouseOver events"},
    "RightClick": {"group": "Page Behavior", "desc": "Right-click disabled"},
    "popUpWidnow": {"group": "Page Behavior", "desc": "Pop-up windows"},
    "Iframe": {"group": "Page Behavior", "desc": "Iframe usage"},
    "age_of_domain": {"group": "Domain", "desc": "Domain age"},
    "DNSRecord": {"group": "Domain", "desc": "DNS record exists"},
    "web_traffic": {"group": "Domain", "desc": "Web traffic ranking"},
    "Page_Rank": {"group": "Domain", "desc": "Google PageRank"},
    "Google_Index": {"group": "Domain", "desc": "Indexed by Google"},
    "Links_pointing_to_page": {"group": "Domain", "desc": "External links count"},
    "Statistical_report": {"group": "Security", "desc": "In statistical reports"}
}

# Actual Model Metrics from DagsHub experiments
MODEL_METRICS = {
    "model_name": "Random Forest",
    "accuracy": 0.0,
    "f1_score": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "training_date": "N/A"
}

# All model comparison data
ALL_MODELS = []

def get_latest_metrics():
    """Fetch latest metrics from DagsHub/MLflow"""
    global MODEL_METRICS, ALL_MODELS
    try:
        # Search for all successful runs
        runs = mlflow.search_runs()
        
        if runs.empty:
            print("⚠️ No MLflow runs found.")
            return

        # Process all runs for the table
        all_models_data = []
        best_f1 = -1
        best_run = None

        for _, run in runs.iterrows():
            # Extract metrics
            accuracy = run.get("metrics.test_accuracy", 0) # Fallback if specific metric missing
            f1 = run.get("metrics.test_f1", 0)
            precision = run.get("metrics.test_precision", 0)
            recall = run.get("metrics.test_recall", 0)
            
            # Extract tags/params
            model_name = run.get("tags.model_name", run.get("params.model_name", "Unknown Model"))
            
            # Check if this is the best model so far
            if f1 > best_f1:
                best_f1 = f1
                best_run = {
                     "model_name": model_name,
                    "accuracy": round(accuracy, 4),
                    "f1_score": round(f1, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "training_date": run.get("start_time", datetime.now()).strftime("%Y-%m-%d") if isinstance(run.get("start_time"), datetime) else str(run.get("start_time"))[:10]
                }

            all_models_data.append({
                "name": model_name,
                "accuracy": round(accuracy, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "best": False # Will set best later
            })
        
        # Update global variables
        if best_run:
            MODEL_METRICS = best_run
            # Mark the best model in the list
            for m in all_models_data:
                if m["name"] == best_run["model_name"] and m["f1"] == best_run["f1_score"]:
                    m["best"] = True
            
            ALL_MODELS = sorted(all_models_data, key=lambda x: x['f1'], reverse=True)
            print(f"✅ Metrics updated from MLflow. Best model: {best_run['model_name']}")

    except Exception as e:
        print(f"❌ Error fetching MLflow metrics: {e}")

# Fetch initial metrics
get_latest_metrics()

# ============================================================
# Initialize FastAPI App
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files and Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================
# Load Models
# ============================================================

def load_models():
    """Load the trained model and preprocessor"""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("✅ Models loaded successfully!")
            return model, preprocessor
        else:
            print("⚠️ Model files not found. Running in demo mode.")
            return None, None
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

model, preprocessor = load_models()

# ============================================================
# Utility Functions
# ============================================================

def generate_sample_data(n_samples: int = 5) -> pd.DataFrame:
    """Generate sample input data for testing"""
    np.random.seed(42)
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = np.random.choice([-1, 0, 1], size=n_samples)
    return pd.DataFrame(data)

def predict_single(features: dict) -> dict:
    """Make prediction for a single sample"""
    if model is None or preprocessor is None:
        # Demo mode - return random prediction
        prediction = np.random.choice([0, 1])
        confidence = np.random.uniform(0.7, 0.99)
    else:
        df = pd.DataFrame([features])
        X_transformed = preprocessor.transform(df)
        prediction = model.predict(X_transformed)[0]
        
        try:
            proba = model.predict_proba(X_transformed)[0]
            confidence = float(max(proba))
        except:
            confidence = 0.95
    
    return {
        "prediction": int(prediction),
        "label": "Phishing" if prediction == 0 else "Legitimate",
        "confidence": round(confidence * 100, 2),
        "is_threat": prediction == 0
    }

def predict_batch(df: pd.DataFrame) -> tuple:
    """Make predictions for batch data"""
    X = df[FEATURE_COLUMNS]
    
    if model is None or preprocessor is None:
        # Demo mode
        predictions = np.random.choice([0, 1], size=len(df))
        confidence = np.random.uniform(0.7, 0.99, size=len(df))
    else:
        X_transformed = preprocessor.transform(X)
        predictions = model.predict(X_transformed)
        
        try:
            probabilities = model.predict_proba(X_transformed)
            confidence = np.max(probabilities, axis=1)
        except:
            confidence = np.ones(len(predictions)) * 0.95
    
    return predictions, confidence

# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload interface"""
    # Refresh metrics on page load
    get_latest_metrics()
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": APP_TITLE,
            "github_url": GITHUB_URL,
            "dagshub_url": DAGSHUB_URL,
            "linkedin_url": LINKEDIN_URL,
            "metrics": MODEL_METRICS,
            "all_models": ALL_MODELS,
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "feature_info": FEATURE_INFO
        }
    )


@app.get("/manual", response_class=HTMLResponse)
async def manual_input(request: Request):
    """Manual input form page"""
    return templates.TemplateResponse(
        "single_predict.html",
        {
            "request": request,
            "title": "Manual Prediction",
            "features": FEATURE_COLUMNS,
            "feature_info": FEATURE_INFO,
            "github_url": GITHUB_URL,
            "dagshub_url": DAGSHUB_URL,
            "linkedin_url": LINKEDIN_URL
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "version": APP_VERSION
    }


@app.get("/api/features")
async def get_features():
    """Return list of required features"""
    return {
        "features": FEATURE_COLUMNS,
        "count": len(FEATURE_COLUMNS),
        "info": FEATURE_INFO
    }


@app.get("/api/metrics")
async def get_metrics():
    """Return model performance metrics"""
    return {
        "best_model": MODEL_METRICS,
        "all_models": ALL_MODELS
    }


@app.get("/download/sample")
async def download_sample():
    """Download sample CSV file"""
    sample_path = "valid_data/test.csv"
    
    if not os.path.exists(sample_path):
        # Fallback if specific file not found
        return JSONResponse(status_code=404, content={"message": "Sample file not found at valid_data/test.csv"})
    
    return FileResponse(
        sample_path,
        media_type="text/csv",
        filename="sample_phishing_data.csv"
    )


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Handle file upload and return predictions"""
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )
    
    try:
        # Read uploaded file
        df = pd.read_csv(file.file)
        original_df = df.copy()
        
        # Validate columns
        missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {', '.join(sorted(missing_cols))}"
            )
        
        # Make predictions
        predictions, confidence = predict_batch(df)
        
        # Map predictions to labels
        prediction_labels = ["⚠️ Phishing" if p == 0 else "🔒 Legitimate" for p in predictions]
        
        # Create results dataframe
        results_df = original_df.copy()
        results_df["Prediction"] = predictions
        results_df["Label"] = prediction_labels
        results_df["Confidence"] = (confidence * 100).round(2)
        
        # Save results
        os.makedirs("prediction_output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"predictions_{timestamp}.csv"
        output_path = f"prediction_output/{output_filename}"
        results_df.to_csv(output_path, index=False)
        
        # Calculate summary
        total = len(predictions)
        # 0 is Phishing (mapped from -1), 1 is Legitimate
        phishing_count = int(np.sum(predictions == 0))
        legitimate_count = total - phishing_count
        
        # Prepare display columns (show first 5 features + results)
        display_columns = FEATURE_COLUMNS[:5] + ["Prediction", "Label", "Confidence"]
        
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "title": "Prediction Results",
                "results": results_df.to_dict(orient="records"),
                "summary": {
                    "total": total,
                    "phishing": phishing_count,
                    "legitimate": legitimate_count,
                    "phishing_percent": round(phishing_count / total * 100, 1) if total > 0 else 0
                },
                "output_filename": output_filename,
                "columns": display_columns,
                "github_url": GITHUB_URL,
                "dagshub_url": DAGSHUB_URL,
                "linkedin_url": LINKEDIN_URL
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/api/predict/single")
async def predict_single_api(request: Request):
    """Handle single prediction from form (API)"""
    try:
        form_data = await request.form()
        
        # Build feature dictionary
        features = {}
        for col in FEATURE_COLUMNS:
            value = form_data.get(col, "0")
            try:
                features[col] = int(value) if value else 0
            except ValueError:
                features[col] = 0
        
        # Make prediction
        result = predict_single(features)

        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, 'item'):  # Numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # Numpy arrays
                return obj.tolist()
            return obj
        
        # Ensure result is JSON serializable
        clean_result = convert_numpy(result)
        
        return JSONResponse(clean_result)
        
    except Exception as e:
        print(f"Prediction Error: {e}") # Log error to console
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/single", response_class=HTMLResponse)
async def predict_single_form(request: Request):
    """Handle single prediction from form (HTML response)"""
    try:
        form_data = await request.form()
        
        # Build feature dictionary
        features = {}
        for col in FEATURE_COLUMNS:
            value = form_data.get(col, "0")
            try:
                features[col] = int(value) if value else 0
            except ValueError:
                features[col] = 0
        
        # Make prediction
        result = predict_single(features)
        
        return templates.TemplateResponse(
            "single_predict.html",
            {
                "request": request,
                "title": "Prediction Result",
                "features": FEATURE_COLUMNS,
                "feature_info": FEATURE_INFO,
                "result": result,
                "input_values": features,
                "github_url": GITHUB_URL,
                "dagshub_url": DAGSHUB_URL,
                "linkedin_url": LINKEDIN_URL
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/results/{filename}")
async def download_results(filename: str):
    """Download prediction results"""
    file_path = f"prediction_output/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="text/csv",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/api/reload-model")
async def reload_model():
    """Force reload of the model and metrics"""
    global model, preprocessor
    try:
        model, preprocessor = load_models()
        get_latest_metrics()
        
        status = "success" if model is not None else "failed"
        return {
            "status": status,
            "message": "Model reload attempted",
            "model_loaded": model is not None,
            "metrics_updated": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model():
    """Trigger model training"""
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        
        # Reload model and metrics
        global model, preprocessor
        model, preprocessor = load_models()
        get_latest_metrics()
        
        return {
            "status": "success",
            "message": "Training completed successfully"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Training Error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": APP_TITLE,
            "error": "Page not found",
            "github_url": GITHUB_URL,
            "dagshub_url": DAGSHUB_URL,
            "linkedin_url": LINKEDIN_URL,
            "metrics": MODEL_METRICS,
            "all_models": ALL_MODELS,
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "feature_info": FEATURE_INFO
        },
        status_code=404
    )


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )