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
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    .stMetric {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #2196F3;
    }
    .stMetric label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #1976D2 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #0D47A1 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 14px !important;
        color: #424242 !important;
    }
    h1 {
        color: #0D47A1;
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    h2 {
        color: #1565C0;
        font-weight: 600;
        margin-top: 30px;
    }
    h3 {
        color: #1976D2;
        font-weight: 600;
    }
    .status-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        border: 3px solid #2196F3;
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.2);
        margin: 20px 0;
    }
    .info-box {
        background: white;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #42A5F5;
    }
    .aqi-parameter {
        background: linear-gradient(135deg, #42A5F5 0%, #1E88E5 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stPlotlyChart {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    p {
        color: #424242;
    }
    .subtitle {
        color: #1976D2;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
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
st.markdown("<p class='subtitle'>Real-time Air Quality Index Forecasting for Karachi, Pakistan</p>", unsafe_allow_html=True)

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
        <h2 style='text-align: center; margin-bottom: 20px; color: #1565C0;'>Current Air Quality Status</h2>
        <div style='text-align: center; font-size: 72px;'>{emoji}</div>
        <h1 style='color: {color}; text-align: center; font-size: 48px; margin: 20px 0;'>{status}</h1>
        <p style='text-align: center; color: #424242; font-size: 24px; font-weight: 600;'>AQI: {day1}</p>
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
    line=dict(color='#1976D2', width=3),
    fill='tozeroy',
    fillcolor='rgba(25, 118, 210, 0.1)'
))

# Forecast data
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_aqi,
    mode='lines+markers',
    name='Predicted AQI',
    line=dict(color='#F57C00', width=4, dash='dash'),
    marker=dict(size=12, color='#F57C00', symbol='diamond')
))

# Add AQI threshold lines
fig.add_hline(y=50, line_dash="dot", line_color="#4CAF50", opacity=0.6, 
              annotation_text="Good", annotation_position="right")
fig.add_hline(y=100, line_dash="dot", line_color="#FFEB3B", opacity=0.6, 
              annotation_text="Moderate", annotation_position="right")
fig.add_hline(y=150, line_dash="dot", line_color="#FF9800", opacity=0.6, 
              annotation_text="Unhealthy for Sensitive", annotation_position="right")
fig.add_hline(y=200, line_dash="dot", line_color="#F44336", opacity=0.6, 
              annotation_text="Unhealthy", annotation_position="right")

fig.update_layout(
    title={
        'text': "Air Quality Index: Historical Data & 3-Day Forecast",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': '#0D47A1', 'family': 'Arial'}
    },
    xaxis_title="Date & Time",
    yaxis_title="AQI Value",
    template="plotly_white",
    hovermode='x unified',
    plot_bgcolor='#F5F5F5',
    paper_bgcolor='white',
    font=dict(color='#424242', size=12),
    xaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
    yaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(f"""
    <div style='text-align: center; margin-top: 50px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
        <p style='color: #424242; font-weight: 500;'>Data updated in real-time from Karachi air quality monitoring stations</p>
        <p style='font-size: 12px; color: #757575;'>Powered by Machine Learning (Model v{model_version}) | US EPA AQI Standards</p>
        <p style='font-size: 10px; margin-top: 10px; color: #9E9E9E;'>Model trained daily • Data refreshed every 5 minutes</p>
    </div>
""", unsafe_allow_html=True)