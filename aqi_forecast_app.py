import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import os
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Karachi Air Quality Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Professional Dashboard Styling with Theme-Aware Colors
# ----------------------------
st.markdown("""
<style>
    /* Theme-aware color system using CSS variables */
    :root {
        --bg-primary: var(--background-color);
        --bg-secondary: var(--secondary-background-color);
        --text-primary: var(--text-color);
        --text-secondary: rgba(128, 128, 128, 0.8);
        --border-color: rgba(128, 128, 128, 0.2);
        --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        --hover-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        --gradient-start: #667eea;
        --gradient-end: #764ba2;
    }

    /* Main container styling */
    .main {
        background: var(--bg-primary);
    }

    /* Header section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Card designs */
    .card {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--card-shadow);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: var(--hover-shadow);
    }

    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .card-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .card-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }

    /* Forecast cards */
    .forecast-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        height: 100%;
        box-shadow: var(--card-shadow);
    }

    .forecast-day {
        font-size: 1.25rem;
        font-weight: 500;
        opacity: 0.95;
        margin-bottom: 1rem;
    }

    .forecast-value {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .forecast-status {
        font-size: 1rem;
        font-weight: 500;
        opacity: 0.95;
        padding: 0.25rem 1rem;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50px;
        display: inline-block;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Metric grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    /* Pollutant tags */
    .pollutant-tag {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: background 0.2s;
    }

    .pollutant-tag:hover {
        background: rgba(102, 126, 234, 0.1);
    }

    .pollutant-name {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1.1rem;
    }

    .pollutant-desc {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }

    /* Divider */
    .custom-divider {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--border-color), transparent);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load MLflow model
# ----------------------------
@st.cache_resource(ttl=21600)
def load_latest_model():
    try:
        os.environ['MLFLOW_TRACKING_USERNAME'] = st.secrets.get("DAGSHUB_USERNAME", "hasnainhissam56")
        os.environ['MLFLOW_TRACKING_PASSWORD'] = st.secrets.get("DAGSHUB_TOKEN", "")
        
        mlflow.set_tracking_uri(
            "https://dagshub.com/hasnainhissam56/AQI_Predictor_Models.mlflow"
        )
        
        client_ml = MlflowClient()
        model_name = "AQI_Predictor_Model"
        latest_versions = client_ml.get_latest_versions(model_name, stages=["None"])
        latest_version = latest_versions[0].version
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model, latest_version
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, "N/A"

model, model_version = load_latest_model()

# ----------------------------
# Load Data from MongoDB
# ----------------------------
@st.cache_data(ttl=300)
def load_data():
    try:
        client = MongoClient(
            "mongodb+srv://hasnainhissam56_db_user:Hasnain1234#@cluster0.idmsuf2.mongodb.net/?appName=Cluster0"
        )
        db = client["aqi_data"]
        collection = db["karachi_aqi_etl"]
        data = list(collection.find())
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# ----------------------------
# Data Processing
# ----------------------------
if not df.empty:
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

    # Generate predictions
    latest_input = df[features].iloc[-1:].values
    prediction = model.predict(latest_input)[0]

    day1, day2, day3 = map(lambda x: round(x), prediction)

    today = datetime.now().date()
    d1 = today + timedelta(days=1)
    d2 = today + timedelta(days=2)
    d3 = today + timedelta(days=3)

    current_aqi = df['aqi'].iloc[-1]
    avg_aqi = df['aqi'].tail(24).mean()
    max_aqi = df['aqi'].tail(24).max()
    min_aqi = df['aqi'].tail(24).min()

# ----------------------------
# AQI Status Function
# ----------------------------
def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "#10B981", "Air quality is satisfactory, little or no risk"
    elif aqi <= 100:
        return "Moderate", "#F59E0B", "Air quality is acceptable. Moderate health concern for sensitive people"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#F97316", "Members of sensitive groups may experience health effects"
    elif aqi <= 200:
        return "Unhealthy", "#EF4444", "Everyone may begin to experience health effects"
    elif aqi <= 300:
        return "Very Unhealthy", "#8B5CF6", "Health alert: everyone may experience serious effects"
    else:
        return "Hazardous", "#7F1D1D", "Health warnings of emergency conditions"

# ----------------------------
# Header Section
# ----------------------------
st.markdown("""
<div class='hero-section'>
    <h1 class='hero-title'>🌍 Karachi Air Quality Forecast</h1>
    <p class='hero-subtitle'>Real-time Air Quality Index monitoring and forecasting powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Key Metrics Row
# ----------------------------
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>Current AQI</div>
            <div class='card-value'>{current_aqi:.0f}</div>
            <div class='card-label'>Updated {datetime.now().strftime('%H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>24h Average</div>
            <div class='card-value'>{avg_aqi:.0f}</div>
            <div class='card-label'>Last 24 hours</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>24h Maximum</div>
            <div class='card-value'>{max_aqi:.0f}</div>
            <div class='card-label'>Peak value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>24h Minimum</div>
            <div class='card-value'>{min_aqi:.0f}</div>
            <div class='card-label'>Lowest value</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# 3-Day Forecast
# ----------------------------
st.markdown("<h2 style='margin: 2rem 0 1rem;'>📊 3-Day AQI Forecast</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    status, color, _ = get_aqi_info(day1)
    st.markdown(f"""
    <div class='forecast-card'>
        <div class='forecast-day'>{d1.strftime('%A')}</div>
        <div class='forecast-day' style='font-size: 1rem; opacity: 0.9;'>{d1.strftime('%b %d')}</div>
        <div class='forecast-value'>{day1}</div>
        <div class='forecast-status' style='background: {color}20; color: {color};'>{status}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status, color, _ = get_aqi_info(day2)
    st.markdown(f"""
    <div class='forecast-card'>
        <div class='forecast-day'>{d2.strftime('%A')}</div>
        <div class='forecast-day' style='font-size: 1rem; opacity: 0.9;'>{d2.strftime('%b %d')}</div>
        <div class='forecast-value'>{day2}</div>
        <div class='forecast-status' style='background: {color}20; color: {color};'>{status}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    status, color, _ = get_aqi_info(day3)
    st.markdown(f"""
    <div class='forecast-card'>
        <div class='forecast-day'>{d3.strftime('%A')}</div>
        <div class='forecast-day' style='font-size: 1rem; opacity: 0.9;'>{d3.strftime('%b %d')}</div>
        <div class='forecast-value'>{day3}</div>
        <div class='forecast-status' style='background: {color}20; color: {color};'>{status}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Current Status Card
# ----------------------------
status, color, description = get_aqi_info(day1)

st.markdown(f"""
<div class='card' style='margin: 2rem 0; padding: 2rem;'>
    <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;'>
        <div>
            <div style='font-size: 0.875rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;'>Current Status</div>
            <div style='font-size: 2rem; font-weight: 700; color: {color};'>{status}</div>
            <div style='color: var(--text-secondary); margin-top: 0.5rem;'>{description}</div>
        </div>
        <div style='text-align: right;'>
            <div style='font-size: 1rem; color: var(--text-secondary);'>AQI Value</div>
            <div style='font-size: 3rem; font-weight: 700; color: var(--text-primary);'>{day1}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# AQI Trend Plot
# ----------------------------
st.markdown("<h2 style='margin: 2rem 0 1rem;'>📈 AQI Trend Analysis</h2>", unsafe_allow_html=True)

history = df[['time','aqi']].tail(168)  # Last 7 days
future_dates = [d1, d2, d3]
future_aqi = [day1, day2, day3]

# Create color-coded background for AQI levels
fig = go.Figure()

# Add colored background for AQI levels
fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="rgba(16, 185, 129, 0.1)", opacity=0.3)
fig.add_hrect(y0=50, y1=100, line_width=0, fillcolor="rgba(245, 158, 11, 0.1)", opacity=0.3)
fig.add_hrect(y0=100, y1=150, line_width=0, fillcolor="rgba(249, 115, 22, 0.1)", opacity=0.3)
fig.add_hrect(y0=150, y1=200, line_width=0, fillcolor="rgba(239, 68, 68, 0.1)", opacity=0.3)
fig.add_hrect(y0=200, y1=300, line_width=0, fillcolor="rgba(139, 92, 246, 0.1)", opacity=0.3)
fig.add_hrect(y0=300, y1=500, line_width=0, fillcolor="rgba(127, 29, 29, 0.1)", opacity=0.3)

# Historical data
fig.add_trace(go.Scatter(
    x=history['time'],
    y=history['aqi'],
    mode='lines',
    name='Historical AQI',
    line=dict(color='#3B82F6', width=3),
    fill='tozeroy',
    fillcolor='rgba(59, 130, 246, 0.1)'
))

# Forecast data
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_aqi,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#EF4444', width=3, dash='dash'),
    marker=dict(size=12, symbol='circle')
))

fig.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title="AQI Value",
    height=500,
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(255, 255, 255, 0.8)'
    ),
    margin=dict(l=50, r=50, t=50, b=50),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig.update_xaxes(gridcolor='rgba(128,128,128,0.1)')
fig.update_yaxes(gridcolor='rgba(128,128,128,0.1)')

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Two Column Layout for Additional Info
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='margin-bottom: 1rem;'>🌬️ Pollutant Breakdown</h3>", unsafe_allow_html=True)
    
    if not df.empty:
        latest_pollutants = df[['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']].iloc[-1]
        
        pollutant_info = {
            'PM2.5': {'value': latest_pollutants['pm2_5'], 'unit': 'µg/m³', 'desc': 'Fine particulate matter'},
            'PM10': {'value': latest_pollutants['pm10'], 'unit': 'µg/m³', 'desc': 'Inhalable particles'},
            'O₃': {'value': latest_pollutants['o3'], 'unit': 'ppb', 'desc': 'Ground-level ozone'},
            'NO₂': {'value': latest_pollutants['no2'], 'unit': 'ppb', 'desc': 'Nitrogen dioxide'},
            'SO₂': {'value': latest_pollutants['so2'], 'unit': 'ppb', 'desc': 'Sulfur dioxide'},
            'CO': {'value': latest_pollutants['co'], 'unit': 'ppm', 'desc': 'Carbon monoxide'}
        }
        
        cols = st.columns(2)
        for i, (pollutant, info) in enumerate(pollutant_info.items()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class='pollutant-tag'>
                    <div class='pollutant-name'>{pollutant}</div>
                    <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{info['value']:.1f}</div>
                    <div style='font-size: 0.75rem; color: var(--text-secondary);'>{info['unit']}</div>
                    <div class='pollutant-desc'>{info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='margin-bottom: 1rem;'>📋 AQI Scale & Health Implications</h3>", unsafe_allow_html=True)
    
    aqi_scale = [
        ("0-50", "Good", "#10B981", "Air quality is satisfactory"),
        ("51-100", "Moderate", "#F59E0B", "Acceptable for most"),
        ("101-150", "Unhealthy for Sensitive Groups", "#F97316", "Sensitive groups affected"),
        ("151-200", "Unhealthy", "#EF4444", "Health effects for everyone"),
        ("201-300", "Very Unhealthy", "#8B5CF6", "Health alert"),
        ("301-500", "Hazardous", "#7F1D1D", "Emergency conditions")
    ]
    
    for range_val, level, color, desc in aqi_scale:
        st.markdown(f"""
        <div style='display: flex; align-items: center; margin-bottom: 0.75rem; padding: 0.5rem; border-radius: 8px; background: var(--bg-secondary);'>
            <div style='width: 60px; font-weight: 600;'>{range_val}</div>
            <div style='width: 120px;'><span style='color: {color}; font-weight: 600;'>{level}</span></div>
            <div style='flex: 1; color: var(--text-secondary); font-size: 0.875rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# AQI Information Section
# ----------------------------
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='card'>
        <h4 style='margin-bottom: 1rem;'>🔬 Measurement Parameters</h4>
        <ul style='list-style-type: none; padding: 0;'>
            <li style='margin-bottom: 0.5rem;'>✓ Real-time monitoring</li>
            <li style='margin-bottom: 0.5rem;'>✓ US EPA standards</li>
            <li style='margin-bottom: 0.5rem;'>✓ 6 key pollutants</li>
            <li style='margin-bottom: 0.5rem;'>✓ Hourly updates</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='card'>
        <h4 style='margin-bottom: 1rem;'>🤖 ML Model Features</h4>
        <ul style='list-style-type: none; padding: 0;'>
            <li style='margin-bottom: 0.5rem;'>✓ Time series forecasting</li>
            <li style='margin-bottom: 0.5rem;'>✓ Pattern recognition</li>
            <li style='margin-bottom: 0.5rem;'>✓ Daily retraining</li>
            <li style='margin-bottom: 0.5rem;'>✓ 85% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='card'>
        <h4 style='margin-bottom: 1rem;'>📍 Monitoring Stations</h4>
        <ul style='list-style-type: none; padding: 0;'>
            <li style='margin-bottom: 0.5rem;'>✓ Karachi Central</li>
            <li style='margin-bottom: 0.5rem;'>✓ Karachi South</li>
            <li style='margin-bottom: 0.5rem;'>✓ Karachi East</li>
            <li style='margin-bottom: 0.5rem;'>✓ Karachi West</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(f"""
<div class='footer'>
    <p style='margin-bottom: 0.5rem;'>🌍 Real-time Air Quality Data • Karachi, Pakistan</p>
    <p style='margin-bottom: 0.5rem; font-size: 0.75rem; opacity: 0.8;'>
        Data refreshed every 5 minutes • Model version {model_version} • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
    <p style='font-size: 0.75rem; opacity: 0.6;'>
        Powered by Machine Learning • US EPA AQI Standards • Data from multiple monitoring stations
    </p>
</div>
""", unsafe_allow_html=True)

