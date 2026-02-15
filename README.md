# Karachi Air Quality Forecasting System

An end-to-end production-ready MLOps pipeline that collects, stores, forecasts, and visualizes Air Quality Index (AQI) data for Karachi, Pakistan.

This project implements automated data ingestion, cloud storage, model training with hyperparameter tuning, MLflow model registry integration, CI/CD automation, and an interactive forecasting dashboard.

---
## 🚀 Live Demo

🔗 https://aqi-forecast-7krbn3hkzzwdfweq3m4ghy.streamlit.app/


## Table of Contents

- Project Overview
- System Architecture
- Data Engineering
- Machine Learning Pipeline
- Model Registry (MLflow + DagsHub)
- Dashboard
- Automation (CI/CD)
- Installation
- Environment Variables
- Project Structure
- Future Improvements
- Author

---

## Project Overview

This system performs:

- Historical AQI data backfilling
- Real-time hourly AQI data ingestion
- Automated daily model retraining
- Multi-step forecasting (3-day ahead prediction)
- Best model selection based on evaluation metrics
- Automatic model registration to MLflow
- Interactive Streamlit dashboard visualization

The entire workflow is fully automated using GitHub Actions.

---

## System Architecture

```
Air Quality API
        ↓
ETL Pipeline (Hourly)
        ↓
MongoDB Atlas
        ↓
Model Training (Daily)
        ↓
MLflow Model Registry
        ↓
Streamlit Dashboard
```

---

## Data Engineering

### Historical Backfill
File: `backfill_data.py`

- Fetches historical hourly AQI data from Open-Meteo API
- Transforms into structured format
- Bulk inserts into MongoDB Atlas

### Real-Time ETL
File: `etl.py`

Runs every hour via GitHub Actions.

Steps:
1. Extract data from API Ninjas
2. Transform into structured pollutant dataset
3. Load into MongoDB Atlas

Database:
- Database: `aqi_data`
- Collection: `karachi_aqi_etl`

Stored Fields:
- time
- co
- no2
- o3
- pm10
- pm2_5
- so2
- aqi

---

## Exploratory Data Analysis

Notebook: `EDA.ipynb`

Includes:
- Time-series AQI visualization
- PM10 and PM2.5 analysis
- Distribution plots
- Correlation heatmap
- Scatter analysis
- Boxen plots

Libraries:
- pandas
- seaborn
- matplotlib
- numpy

---

## Machine Learning Pipeline

File: `model_training.py`

### Feature Engineering

Time-Based Features:
- hour
- day
- dayofweek
- month

Derived Feature:
- aqi_change_rate

Multi-step Targets:
- aqi_t+1 (24 hours ahead)
- aqi_t+2 (48 hours ahead)
- aqi_t+3 (72 hours ahead)

### Models Trained

All models wrapped with `MultiOutputRegressor`:

- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regressor (SVR)

### Hyperparameter Tuning

- GridSearchCV
- 3-Fold Cross Validation

### Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

Best model selected based on highest R² score.

---

## Model Registry (MLflow + DagsHub)

The best-performing model is:

- Logged to MLflow
- Registered under name:

```
AQI_Predictor_Model
```

Tracking Server:
- Hosted on DagsHub

Model training runs automatically every day at midnight UTC.

Workflow File:
```
model_training.yaml
```

---

## Dashboard

File: `app.py`

Built with Streamlit and Plotly.

Features:

- 3-Day AQI Forecast Cards
- AQI Category Classification:
  - Good
  - Moderate
  - Unhealthy for Sensitive Groups
  - Unhealthy
  - Very Unhealthy
  - Hazardous
- Historical vs Forecast Comparison Chart
- Forecast Confidence Band
- Interactive Range Slider
- Model Version Display
- Automatic Data Caching

Run locally:

```
streamlit run app.py
```

---

## Automation (CI/CD)

### Hourly ETL Workflow

File:
```
etl_pipeline.yaml
```

Schedule:
```
0 * * * *  (Every hour)
```

### Daily Model Training Workflow

File:
```
model_training.yaml
```

Schedule:
```
0 0 * * *  (Daily at midnight UTC)
```

---

## Installation

Clone the repository:

```
git clone <repository-url>
cd <project-folder>
```

Install dependencies:

```
pip install -r requirements.txt
```

Run dashboard:

```
streamlit run app.py
```

---

## Environment Variables

The following secrets must be configured:

- MONGO_URI
- _API_NINJA_KEY_
- DAGSHUB_REPO_TOKEN

Never commit credentials directly to the repository.

---

## Requirements

```
pandas==2.3.3
requests==2.32.5
pyarrow==18.1.0
pymongo==4.15.5
mlflow>=2.9.0
dagshub>=0.3.8
scikit-learn>=1.3.0
xgboost>=2.0.0
numpy>=1.24.0
streamlit==1.29.0
plotly==5.16.1
```

---

## Project Structure

```
AQI-Predictor-Application/
│
├── .github/
│   └── workflows/
│       ├── etl_pipeline.yaml
│       └── model_training_pipeline.yaml
│
├── EDA/
│   └── EDA.ipynb
│
├── .gitignore
├── aqi_forecast_app.py
├── backfill_data.py
├── etl.py
├── model_training.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Add LSTM / Deep Learning models
- Implement SHAP feature importance
- Add anomaly detection for extreme AQI spikes
- Deploy dashboard to cloud (AWS / Streamlit Cloud)
- Add alert system for hazardous AQI levels

---

## Author

Hasnain Hissam
Data Science & Machine Learning Engineer

---

## License

This project is developed for 10Pearls Shine Internship and research purposes.

