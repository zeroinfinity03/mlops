## End-to-End MLOps Project demonstrating a complete machine learning pipeline using modern tools and best practices for model development, deployment, and monitoring.

### This is not the readme created by cookiecutter, this is my guide i always paste inside readme.md




# End-to-End MLOps Project Guide

## 1. Project Setup with Cookiecutter

**Install cookiecutter data science project using uv:**
(Check if uv is installed: `pip install uv`)

```bash
uvx --from cookiecutter-data-science ccds
```

**Configuration choices:**
- python_version_number => 3.11
- Select dataset_storage => 1(none)
- Select environment_manager => 4(uv)
- Select dependency_file => 2(pyproject.toml)
- Select pydata_packages => 2(basic)
- Select testing_framework => 1(pytest)
- Select linting_and_formatting => 1(ruff)
- Select open_source_license => 1(No license file)
- Select docs => mkdocs
- Select include_code_scaffold => 2(We don't want any boilerplate code for now)

---

## 2. Initialize Dependencies

```bash
cd projectname_you_just_chose
# We won't do => uv init => because cookiecutter already installed pyproject.toml file since we chose uv earlier
```

```bash
# Install core ML/data libraries
uv add pandas polars matplotlib seaborn scipy pydantic pingouin statsmodels scikit-learn mlflow dvc fastapi requests numpy torch tensorflow prophet transformers jupyter uvicorn python-multipart joblib docker boto3 dagshub
```

```bash
# Sync lockfile
uv sync
```

---

## 3. Setup Git Repository

```bash
git init
git add .
git commit -m "project structure ready"
git branch -M main
git remote add origin https://github.com/zeroinfinity03/mlops.git
git push -u origin main
```

**Note:** If the folders are empty, you won't see them in GitHub for now.

---

## 4. Setup MLFlow on DagHub

1. Go to: https://dagshub.com/dashboard
2. Create -> New Repo -> Connect a repo -> (GitHub) Connect -> Select your repo -> Connect
3. Copy experiment tracking URL and code snippet (Also try: Go To MLFlow UI)

**Security Note:** Paste the credentials from DagHub to `.env` - we will use it later for tracking.

---

## 5. Create DVC Pipeline Scripts

Create these numbered scripts in `mlops/mlops/` folder:

```
mlops/mlops/
├── 1.data_ingest.py      ← Download/load raw data
├── 2.data_preprocess.py  ← Clean and prepare data  
├── 3.FE.py               ← Feature engineering
├── 4.feature_selection.py ← Select best features
├── 5.split&train.py      ← Split data and train model
├── 6.evaluate.py         ← Evaluate model performance
├── 7.register.py         ← Register model to MLflow
```

**What each script does:**
- **Data Ingestion:** Load raw data from source
- **Preprocessing:** Clean, handle missing values, outliers
- **Feature Engineering:** Create new features, transformations
- **Feature Selection:** Select most important features
- **Train:** Split data, train model with best parameters
- **Evaluate:** Test model performance, generate metrics
- **Register:** Save model to MLflow registry

---

## 6. Experiment with MLflow (Find Best Model)

Before creating the DVC pipeline, experiment to find the best model:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set tracking URI from .env
mlflow.set_tracking_uri("your-dagshub-uri")

# Experiment with different models
with mlflow.start_run():
    # Try different models and hyperparameters
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log everything
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    # Save model
    mlflow.sklearn.log_model(model, "model")
```

**Goal:** Run multiple experiments, compare results in MLflow UI, find the best model and parameters.

---

## 7. Create DVC Pipeline Configuration

Once you know your best parameters from MLflow experiments:

### Create `params.yaml` in project root:

```yaml
data_ingestion:
  source_url: "your-data-source"
  raw_data_path: "data/raw/dataset.csv"
  
preprocessing:
  test_size: 0.2
  random_state: 42
  handle_missing: "drop"

feature_engineering:
  scaling_method: "standard"
  
model:
  model_type: "RandomForest"
  n_estimators: 100
  max_depth: 10
  random_state: 42
  
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
```

### Create `dvc.yaml` in project root:

```yaml
stages:
  data_ingestion:
    cmd: uv run python -m mlops.1.data_ingest
    params:
    - data_ingestion
    outs:
    - data/raw/dataset.csv
    
  preprocessing:
    cmd: uv run python -m mlops.2.data_preprocess
    deps:
    - data/raw/dataset.csv
    params:
    - preprocessing
    outs:
    - data/interim/cleaned_data.csv
    
  feature_engineering:
    cmd: uv run python -m mlops.3.FE
    deps:
    - data/interim/cleaned_data.csv
    params:
    - feature_engineering
    outs:
    - data/processed/features.csv
    
  feature_selection:
    cmd: uv run python -m mlops.4.feature_selection
    deps:
    - data/processed/features.csv
    outs:
    - data/processed/selected_features.csv
    
  train:
    cmd: uv run python -m mlops.5.split&train
    deps:
    - data/processed/selected_features.csv
    params:
    - model
    - preprocessing.test_size
    - preprocessing.random_state
    outs:
    - models/model.pkl
    - data/processed/X_train.csv
    - data/processed/X_test.csv
    - data/processed/y_train.csv
    - data/processed/y_test.csv
    
  evaluate:
    cmd: uv run python -m mlops.6.evaluate
    deps:
    - models/model.pkl
    - data/processed/X_test.csv
    - data/processed/y_test.csv
    params:
    - evaluation
    metrics:
    - reports/metrics.json
    
  register:
    cmd: uv run python -m mlops.7.register
    deps:
    - models/model.pkl
    - reports/metrics.json
```

### Run the DVC pipeline:

```bash
# Initialize DVC
dvc init

# Run the entire pipeline
dvc repro

# If you change params.yaml, just run:
dvc repro
```

### ✅ Why DVC Pipeline?

| Goal                            | DVC Handles? |
| ------------------------------- | ------------ |
| Auto-run only changed steps     | ✅ Yes        |
| Version control for data/models | ✅ Yes        |
| Reproducibility                 | ✅ Yes        |
| Avoid retraining needlessly     | ✅ Yes        |
| Re-run stages only if needed    | ✅ Yes        |

### ✅ What if new data comes in (after 6 months)?

```bash
# If same structure (just new rows):
# ✅ Update CSV → dvc repro

# If structure changed (new cols, formats):
# ❗ Update cleaning or FE script → dvc repro
```

| Situation                      | What You Do                          |
| ------------------------------ | ------------------------------------ |
| Just new rows                  | ✅ Replace data → `dvc repro`         |
| New column structure or format | ❗ Edit script → `dvc repro`          |
| Only parameter change          | ✅ Update `params.yaml` → `dvc repro` |

### ✅ DVC's Strength in One Line:

🧠 You write logic ONCE → ⚙️ DVC reruns when inputs change

---

## 8. Testing: Model Readiness Before Deployment

We do **2 stages of testing** — in this order:

### ✅ 1. Model Loading Test (Most Important)

```text
→ Check if model loads from registry
→ Check if it gives any prediction at all
→ If it fails here, skip further testing
```

### ✅ 2. Performance Testing (Choose ONE)

```text
→ Option A: Check if accuracy ≥ required threshold (e.g., 85%)
→ Option B: Compare v18 with v17 → if better, promote
```

| Type             | What it checks                    |
| ---------------- | --------------------------------- |
| Load Test        | Can we load model from registry?  |
| Prediction Test  | Does it give valid predictions?   |
| Performance Test | Either threshold or version-based |

### Create `tests/test_model.py`:

```python
import joblib
import pandas as pd
import pytest
import json

def test_model_loading():
    """Test if model loads from registry - MOST IMPORTANT"""
    try:
        model = joblib.load("models/model.pkl")
        assert model is not None
        print("✅ Model loads successfully")
    except Exception as e:
        pytest.fail(f"❌ Model loading failed: {e}")

def test_model_prediction():
    """Test if it gives any prediction at all"""
    model = joblib.load("models/model.pkl")
    # Load sample data
    X_test = pd.read_csv("data/processed/X_test.csv")
    
    predictions = model.predict(X_test.head(5))
    assert len(predictions) == 5
    assert predictions is not None
    print("✅ Model gives valid predictions")

def test_model_performance_threshold():
    """Option A: Check if accuracy ≥ required threshold"""
    with open("reports/metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Set your threshold
    ACCURACY_THRESHOLD = 0.85
    
    assert metrics["accuracy"] >= ACCURACY_THRESHOLD, f"❌ Accuracy {metrics['accuracy']} below threshold {ACCURACY_THRESHOLD}"
    print(f"✅ Model meets accuracy threshold: {metrics['accuracy']}")

def test_model_version_comparison():
    """Option B: Compare current model with previous version"""
    # Load current model metrics
    with open("reports/metrics.json", "r") as f:
        current_metrics = json.load(f)
    
    # Load previous model metrics (if exists)
    try:
        with open("reports/previous_metrics.json", "r") as f:
            previous_metrics = json.load(f)
        
        # Compare performance
        assert current_metrics["accuracy"] >= previous_metrics["accuracy"], f"❌ New model worse than previous: {current_metrics['accuracy']} < {previous_metrics['accuracy']}"
        print(f"✅ New model better than previous: {current_metrics['accuracy']} >= {previous_metrics['accuracy']}")
    except FileNotFoundError:
        print("⚠️ No previous model to compare with")
```

### Run tests:

```bash
uv run pytest tests/test_model.py -v
```

**Testing Strategy:**
- If **Load Test** fails → Stop here, fix the model loading issue
- If **Prediction Test** fails → Fix model inference logic  
- If **Performance Test** fails → Either retrain or use previous model version

---

## 9. Production Setup

### Backend API
Create `production/backend/main.py` for serving the model:

```
mlops/
├── production/
│   ├── backend/
│   │   ├── main.py          ← FastAPI app
│   │   ├── models.py        ← Pydantic models
│   │   └── utils.py         ← Helper functions
│   └── frontend/            ← Frontend code (React/Streamlit/HTML)
│       ├── app.py           ← Streamlit app (if using Streamlit)
│       ├── index.html       ← HTML frontend (if using vanilla JS)
│       ├── src/             ← React components (if using React)
│       └── static/          ← CSS, JS, images
```

### Backend API (`production/backend/main.py`):

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI(title="ML Model API")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="production/frontend/static"), name="static")

# Load model at startup
model = joblib.load("models/model.pkl")

class PredictionRequest(BaseModel):
    # Define your input features here
    feature1: float
    feature2: float
    feature3: str

class PredictionResponse(BaseModel):
    prediction: float
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Convert to DataFrame
    input_data = pd.DataFrame([request.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()
    
    return PredictionResponse(
        prediction=prediction,
        probability=probability
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Serve frontend
@app.get("/")
async def serve_frontend():
    with open("production/frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())
```

### Frontend Options:

**Option 1: Simple HTML/JS (`production/frontend/index.html`):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>ML Model Prediction</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>ML Model Prediction</h1>
        <form id="predictionForm">
            <input type="number" id="feature1" placeholder="Feature 1" required>
            <input type="number" id="feature2" placeholder="Feature 2" required>
            <input type="text" id="feature3" placeholder="Feature 3" required>
            <button type="submit">Predict</button>
        </form>
        <div id="result"></div>
    </div>
    <script src="/static/app.js"></script>
</body>
</html>
```

**Option 2: Streamlit App (`production/frontend/app.py`):**
```python
import streamlit as st
import requests
import json

st.title("ML Model Prediction")

# Input form
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2") 
feature3 = st.text_input("Feature 3")

if st.button("Predict"):
    # Call backend API
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Probability: {result['probability']:.2f}")
    else:
        st.error("Prediction failed")
```

### Run the applications:

```bash
# Run backend API
uv run uvicorn production.backend.main:app --reload

# If using Streamlit frontend (in separate terminal)
uv run streamlit run production/frontend/app.py
```

---

## 10. Deployment Strategy

### Create `deploy/` folder structure:

```
mlops/
├── deploy/
│   ├── Dockerfile
│   ├── start.sh
│   ├── docker-compose.yml
│   └── aws/
│       ├── ecr_push.sh
│       └── ec2_deploy.sh
```

### Create `deploy/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv
RUN uv sync --frozen

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "production.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### AWS Deployment Process:

**Step 1: Dockerize API → Store in Amazon ECR**
```bash
# Build and tag image
docker build -t mlops-api .
docker tag mlops-api:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/mlops-api:latest

# Push to ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/mlops-api:latest
```

**Step 2: Deploy to EC2 instances**
```bash
# Launch EC2 instance, then pull image from ECR
docker pull 123456789.dkr.ecr.us-east-1.amazonaws.com/mlops-api:latest
docker run -p 8000:8000 mlops-api:latest
```

**Step 3: Production Best Practices**

### ⚠️ Important Notes:

1. **Don't deploy on just one EC2 instance** → Use 2-3 instances for redundancy
2. **Use Load Balancer** → Frontend hits Load Balancer, which routes to least loaded EC2
3. **Use Auto Scaling Group** → AWS adjusts server count based on traffic
4. **Zero-downtime deployment** → Use Rolling Updates or Blue-Green deployment

### Deployment Strategies:

| Strategy | How it works | Downtime |
|----------|-------------|----------|
| **Rolling Update** | Update 1-2 servers at a time | ✅ Zero |
| **Blue-Green** | Maintain old & new versions, switch after testing | ✅ Zero |
| **Manual** | Update all servers at once | ❌ High risk |

### Use AWS CodeDeploy for automation:

```yaml
# deploy/aws/codedeploy.yml
version: 0.0
os: linux
files:
  - source: /
    destination: /var/www/mlops-api
hooks:
  BeforeInstall:
    - location: scripts/install_dependencies.sh
  ApplicationStart:
    - location: scripts/start_server.sh
  ApplicationStop:
    - location: scripts/stop_server.sh
```

---

## 11. CI/CD with GitHub Actions

Create `.github/workflows/ci-cd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install uv
      run: pip install uv
    
    - name: Install dependencies
      run: uv sync
    
    - name: Run tests
      run: uv run pytest tests/ -v
    
    - name: Run DVC pipeline
      run: |
        uv run dvc repro
    
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t mlops-api .
    
    - name: Deploy to production
      run: |
        # Add your deployment commands here
        echo "Deploying to production..."
```

---

## Summary

This guide provides a complete MLOps workflow:

1. **Setup** → Project structure with cookiecutter
2. **Dependencies** → Modern Python package management with uv
3. **Version Control** → Git setup
4. **Experiment Tracking** → MLflow + DagHub integration
5. **Pipeline Scripts** → Modular ML pipeline components
6. **Experimentation** → Find best model with MLflow
7. **Automation** → DVC pipeline for reproducibility
8. **Testing** → Model validation before deployment
9. **Production** → FastAPI for model serving
10. **Deployment** → Docker containerization
11. **CI/CD** → Automated testing and deployment

Each step builds on the previous one, creating a professional MLOps workflow that's scalable and maintainable.