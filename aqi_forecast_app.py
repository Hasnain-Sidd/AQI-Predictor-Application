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
    page_title="Karachi AQI Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .stMetric label {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    h2 {
        color: #ffffff;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    h3 {
        color: #ffffff;
        font-weight: 600;
    }
    .status-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .info-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .aqi-parameter {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        font-weight: 600;
    }
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load MLflow model (always fetch latest)
# ----------------------------
import os

# Set DagHub credentials from Streamlit secrets
os.environ['MLFLOW_TRACKING_USERNAME'] = st.secrets.get("DAGSHUB_USERNAME", "hasnainhissam56")
os.environ['MLFLOW_TRACKING_PASSWORD'] = st.secrets.get("DAGSHUB_TOKEN", "")

mlflow.set_tracking_uri(
    "https://dagshub.com/hasnainhissam56/AQI_Predictor_Models.mlflow"
)

@st.cache_resource(ttl=21600)  # Cache model for 6 hours, reloads 4 times daily to catch new versions
def load_latest_model():
    """Load the latest version of the model from MLflow"""
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

# ----------------------------
# Prepare latest input for prediction
# ----------------------------
latest_input = df[features].iloc[-1:].values

# ----------------------------
# Prediction using MLflow model
# ----------------------------
prediction = model.predict(latest_input)[0]

day1, day2, day3 = map(lambda x: round(x), prediction)

today = datetime.now().date()
d1 = today + timedelta(days=1)
d2 = today + timedelta(days=2)
d3 = today + timedelta(days=3)

# ----------------------------
# Header Section
# ----------------------------
st.markdown("<h1>🌍 Karachi AQI Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px; margin-bottom: 30px;'>Real-time Air Quality Index Forecasting for Karachi, Pakistan</p>", unsafe_allow_html=True)

# ----------------------------
# Display 3-Day AQI Forecast
# ----------------------------
st.markdown("<h2 style='text-align: center; margin-top: 30px;'>📅 3-Day AQI Forecast</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"🌤️ {d1.strftime('%A, %b %d')}",
        value=f"{day1}",
        delta="AQI"
    )
with col2:
    st.metric(
        label=f"🌤️ {d2.strftime('%A, %b %d')}",
        value=f"{day2}",
        delta="AQI"
    )
with col3:
    st.metric(
        label=f"🌤️ {d3.strftime('%A, %b %d')}",
        value=f"{day3}",
        delta="AQI"
    )

# ----------------------------
# AQI Status Function
# ----------------------------
def aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#00e400", "😊"
    elif aqi <= 100:
        return "Moderate", "#ffff00", "😐"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "😷"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000", "😨"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97", "😰"
    else:
        return "Hazardous", "#7e0023", "☠️"

status, color, emoji = aqi_status(day1)

# ----------------------------
# Current Status Card
# ----------------------------
st.markdown(f"""
    <div class='status-card'>
        <h2 style='text-align: center; margin-bottom: 20px;'>Current Air Quality Status</h2>
        <div style='text-align: center; font-size: 72px;'>{emoji}</div>
        <h1 style='color: {color}; text-align: center; font-size: 48px; margin: 20px 0;'>{status}</h1>
        <p style='text-align: center; color: white; font-size: 24px;'>AQI: {day1}</p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# US AQI Calculation Parameters
# ----------------------------
st.markdown("<h2 style='margin-top: 40px;'>📊 US AQI Calculation Parameters</h2>", unsafe_allow_html=True)

st.markdown("""
    <div class='info-box'>
        <h3 style='color: #667eea; margin-bottom: 15px;'>How is AQI Calculated?</h3>
        <p style='color: #333; font-size: 16px; line-height: 1.6;'>
            The Air Quality Index (AQI) is calculated based on the concentration of major air pollutants 
            using the US EPA standard formula. The AQI value is determined by the highest concentration 
            of any individual pollutant.
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class='aqi-parameter'>
            <strong>🔬 Pollutants Measured:</strong><br>
            • PM2.5 (Fine Particles)<br>
            • PM10 (Coarse Particles)<br>
            • O₃ (Ozone)<br>
            • NO₂ (Nitrogen Dioxide)<br>
            • SO₂ (Sulfur Dioxide)<br>
            • CO (Carbon Monoxide)
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='aqi-parameter'>
            <strong>📈 AQI Scale (US EPA):</strong><br>
            • 0-50: Good (Green)<br>
            • 51-100: Moderate (Yellow)<br>
            • 101-150: Unhealthy for Sensitive (Orange)<br>
            • 151-200: Unhealthy (Red)<br>
            • 201-300: Very Unhealthy (Purple)<br>
            • 301+: Hazardous (Maroon)
        </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Plot Historical and Forecast AQI
# ----------------------------
st.markdown("<h2 style='margin-top: 40px;'>📈 AQI Trend Analysis</h2>", unsafe_allow_html=True)

history = df[['time','aqi']].tail(100)
future_dates = [d1, d2, d3]
future_aqi = [day1, day2, day3]

fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=history['time'],
    y=history['aqi'],
    mode='lines',
    name='Historical AQI',
    line=dict(color='#667eea', width=3),
    fill='tozeroy',
    fillcolor='rgba(102, 126, 234, 0.2)'
))

# Forecast data
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_aqi,
    mode='lines+markers',
    name='Predicted AQI',
    line=dict(color='#f5576c', width=4, dash='dash'),
    marker=dict(size=12, color='#f5576c', symbol='diamond')
))

# Add AQI threshold lines
fig.add_hline(y=50, line_dash="dot", line_color="green", opacity=0.5, 
              annotation_text="Good", annotation_position="right")
fig.add_hline(y=100, line_dash="dot", line_color="yellow", opacity=0.5, 
              annotation_text="Moderate", annotation_position="right")
fig.add_hline(y=150, line_dash="dot", line_color="orange", opacity=0.5, 
              annotation_text="Unhealthy for Sensitive", annotation_position="right")
fig.add_hline(y=200, line_dash="dot", line_color="red", opacity=0.5, 
              annotation_text="Unhealthy", annotation_position="right")

fig.update_layout(
    title={
        'text': "Air Quality Index: Historical Data & 3-Day Forecast",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': 'white', 'family': 'Arial Black'}
    },
    xaxis_title="Date & Time",
    yaxis_title="AQI Value",
    template="plotly_dark",
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=12),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(f"""
    <div style='text-align: center; margin-top: 50px; padding: 20px; color: white; opacity: 0.8;'>
        <p>Data updated in real-time from Karachi air quality monitoring stations</p>
        <p style='font-size: 12px;'>Powered by Machine Learning (Model v{model_version}) | US EPA AQI Standards</p>
        <p style='font-size: 10px; margin-top: 10px;'>Model trained daily • Data refreshed every 5 minutes</p>
    </div>
""", unsafe_allow_html=True)