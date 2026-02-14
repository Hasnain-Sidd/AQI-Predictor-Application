import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.graph_objects as go
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import os

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Karachi Air Quality Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Professional Dashboard Styling
# ----------------------------
st.markdown("""
<style>

/* Global background */
.main {
    background-color: #F8FAFC;
}

/* Main Title */
h1 {
    color: #0F172A;
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #475569;
    margin-bottom: 30px;
}

/* Section Headings */
h2 {
    color: #1E293B;
    font-weight: 600;
    margin-top: 35px;
    margin-bottom: 15px;
}

h3 {
    color: #334155;
    font-weight: 600;
}

/* Metric Cards */
.stMetric {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    text-align: center;
}

.stMetric label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #64748B !important;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 34px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

.stMetric [data-testid="stMetricDelta"] {
    font-size: 13px !important;
    color: #94A3B8 !important;
}

/* Status Card */
.status-card {
    background: #FFFFFF;
    padding: 35px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin: 30px 0;
    text-align: center;
}

/* Info Boxes */
.info-box {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

/* AQI Parameter Boxes */
.aqi-parameter {
    background: #1E293B;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    color: #FFFFFF;
    font-size: 15px;
    line-height: 1.6;
}

/* Plot Container */
.stPlotlyChart {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 8px rgba(0,0,0,0.04);
}

/* Footer */
.footer-box {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load MLflow model
# ----------------------------
os.environ['MLFLOW_TRACKING_USERNAME'] = st.secrets.get("DAGSHUB_USERNAME", "hasnainhissam56")
os.environ['MLFLOW_TRACKING_PASSWORD'] = st.secrets.get("DAGSHUB_TOKEN", "")

mlflow.set_tracking_uri(
    "https://dagshub.com/hasnainhissam56/AQI_Predictor_Models.mlflow"
)

@st.cache_resource(ttl=21600)
def load_latest_model():
    client_ml = MlflowClient()
    model_name = "AQI_Predictor_Model"
    latest_versions = client_ml.get_latest_versions(model_name, stages=["None"])
    latest_version = latest_versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, latest_version

model, model_version = load_latest_model()

# ----------------------------
# Load Data from MongoDB
# ----------------------------
@st.cache_data(ttl=300)
def load_data():
    client = MongoClient(
        "mongodb+srv://hasnainhissam56_db_user:Hasnain1234#@cluster0.idmsuf2.mongodb.net/?appName=Cluster0"
    )
    db = client["aqi_data"]
    collection = db["karachi_aqi_etl"]
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

df = load_data()

df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(by='time')

df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['dayofweek'] = df['time'].dt.dayofweek
df['month'] = df['time'].dt.month
df['aqi_change_rate'] = df['aqi'].diff()

features = ['co','no2','o3','pm10','pm2_5','so2',
            'hour','day','dayofweek','month','aqi_change_rate']

df.dropna(inplace=True)

latest_input = df[features].iloc[-1:].values
prediction = model.predict(latest_input)[0]

day1, day2, day3 = map(lambda x: round(x), prediction)

today = datetime.now().date()
d1 = today + timedelta(days=1)
d2 = today + timedelta(days=2)
d3 = today + timedelta(days=3)

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1>Karachi Air Quality Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time Air Quality Index Forecasting Dashboard</p>", unsafe_allow_html=True)

# ----------------------------
# 3-Day Forecast
# ----------------------------
st.markdown("<h2>3-Day AQI Forecast</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label=f"{d1.strftime('%A, %b %d')}", value=f"{day1}", delta="AQI")

with col2:
    st.metric(label=f"{d2.strftime('%A, %b %d')}", value=f"{day2}", delta="AQI")

with col3:
    st.metric(label=f"{d3.strftime('%A, %b %d')}", value=f"{day3}", delta="AQI")

# ----------------------------
# AQI Status Function
# ----------------------------
def aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#16A34A"
    elif aqi <= 100:
        return "Moderate", "#CA8A04"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#EA580C"
    elif aqi <= 200:
        return "Unhealthy", "#DC2626"
    elif aqi <= 300:
        return "Very Unhealthy", "#7C3AED"
    else:
        return "Hazardous", "#7F1D1D"

status, color = aqi_status(day1)

# ----------------------------
# Current Status Card
# ----------------------------
st.markdown(f"""
    <div class='status-card'>
        <h2>Current Air Quality Status</h2>
        <h1 style='color: {color}; font-size: 42px; margin: 15px 0;'>{status}</h1>
        <p style='color: #475569; font-size: 18px; font-weight: 600;'>
            AQI Value: {day1}
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# AQI Information
# ----------------------------
st.markdown("<h2>US AQI Calculation Parameters</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
    <h3>How is AQI Calculated?</h3>
    <p>
        The Air Quality Index (AQI) is calculated based on the concentration 
        of major air pollutants using the US EPA standard formula. 
        The AQI value is determined by the highest concentration 
        of any individual pollutant.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='aqi-parameter'>
        <strong>Pollutants Measured:</strong><br>
        • PM2.5<br>
        • PM10<br>
        • O₃<br>
        • NO₂<br>
        • SO₂<br>
        • CO
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='aqi-parameter'>
        <strong>AQI Scale (US EPA):</strong><br>
        • 0–50: Good<br>
        • 51–100: Moderate<br>
        • 101–150: Unhealthy for Sensitive Groups<br>
        • 151–200: Unhealthy<br>
        • 201–300: Very Unhealthy<br>
        • 301+: Hazardous
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# AQI Trend Plot
# ----------------------------
st.markdown("<h2>AQI Trend Analysis</h2>", unsafe_allow_html=True)

history = df[['time','aqi']].tail(100)
future_dates = [d1, d2, d3]
future_aqi = [day1, day2, day3]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history['time'],
    y=history['aqi'],
    mode='lines',
    name='Historical AQI',
    line=dict(color='#1E40AF', width=3)
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_aqi,
    mode='lines+markers',
    name='Predicted AQI',
    line=dict(color='#DC2626', width=3, dash='dash')
))

fig.update_layout(
    template="plotly_white",
    xaxis_title="Date & Time",
    yaxis_title="AQI Value",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(f"""
<div class='footer-box'>
    <p>Data updated in real-time from Karachi monitoring stations</p>
    <p style='font-size: 13px; color: #64748B;'>
        Powered by Machine Learning (Model v{model_version}) | US EPA AQI Standards
    </p>
    <p style='font-size: 12px; color: #94A3B8;'>
        Model retrained daily • Data refreshed every 5 minutes
    </p>
</div>
""", unsafe_allow_html=True)
