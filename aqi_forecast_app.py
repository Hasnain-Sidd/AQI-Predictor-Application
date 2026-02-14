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
# Simple, Clean Styling
# ----------------------------
st.markdown("""
<style>
    /* Clean, minimal styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
    }
    
    .forecast-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .forecast-card {
        flex: 1;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .forecast-day {
        font-size: 1.2rem;
        font-weight: 500;
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    
    .forecast-date {
        font-size: 0.9rem;
        color: rgba(128, 128, 128, 0.8);
        margin-bottom: 1rem;
    }
    
    .forecast-value {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 0.5rem 0;
    }
    
    .aqi-status {
        display: inline-block;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 500;
        color: var(--text-color);
        margin: 2rem 0 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: rgba(128, 128, 128, 0.6);
        font-size: 0.8rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        margin-top: 3rem;
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

# ----------------------------
# AQI Status Function
# ----------------------------
def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#10B981"
    elif aqi <= 100:
        return "Moderate", "#F59E0B"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#F97316"
    elif aqi <= 200:
        return "Unhealthy", "#EF4444"
    elif aqi <= 300:
        return "Very Unhealthy", "#8B5CF6"
    else:
        return "Hazardous", "#7F1D1D"

# ----------------------------
# Load model and data
# ----------------------------
model, model_version = load_latest_model()
df = load_data()

# ----------------------------
# Main App
# ----------------------------
st.markdown('<h1 class="main-title">Karachi Air Quality Forecast</h1>', unsafe_allow_html=True)

if not df.empty and model is not None:
    # Data Processing
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

    # ----------------------------
    # 3-Day Forecast Cards
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        status, color = get_aqi_status(day1)
        st.markdown(f"""
        <div class='forecast-card'>
            <div class='forecast-day'>{d1.strftime('%A')}</div>
            <div class='forecast-date'>{d1.strftime('%B %d, %Y')}</div>
            <div class='forecast-value'>{day1}</div>
            <div class='aqi-status' style='background-color: {color}20; color: {color};'>{status}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        status, color = get_aqi_status(day2)
        st.markdown(f"""
        <div class='forecast-card'>
            <div class='forecast-day'>{d2.strftime('%A')}</div>
            <div class='forecast-date'>{d2.strftime('%B %d, %Y')}</div>
            <div class='forecast-value'>{day2}</div>
            <div class='aqi-status' style='background-color: {color}20; color: {color};'>{status}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status, color = get_aqi_status(day3)
        st.markdown(f"""
        <div class='forecast-card'>
            <div class='forecast-day'>{d3.strftime('%A')}</div>
            <div class='forecast-date'>{d3.strftime('%B %d, %Y')}</div>
            <div class='forecast-value'>{day3}</div>
            <div class='aqi-status' style='background-color: {color}20; color: {color};'>{status}</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------
    # AQI Chart - Actual vs Forecast
    # ----------------------------
    st.markdown('<h2 class="section-title">AQI Trend: Actual vs Forecast</h2>', unsafe_allow_html=True)

    # Get last 30 days of actual data
    history = df[['time', 'aqi']].tail(30)
    
    # Create future dates for forecast
    future_dates = [d1, d2, d3]
    future_aqi = [day1, day2, day3]

    # Create the chart
    fig = go.Figure()

    # Actual AQI line
    fig.add_trace(go.Scatter(
        x=history['time'],
        y=history['aqi'],
        mode='lines',
        name='Actual AQI',
        line=dict(color='#3B82F6', width=2)
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_aqi,
        mode='lines+markers',
        name='Forecast AQI',
        line=dict(color='#EF4444', width=2, dash='dash'),
        marker=dict(size=8)
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI Value",
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(gridcolor='rgba(128,128,128,0.1)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.1)')

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Simple Footer
    # ----------------------------
    st.markdown(f"""
    <div class='footer'>
        Karachi Air Quality Forecast • Updated every 5 minutes • Model v{model_version}
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Unable to load data. Please check your connection and try again.")