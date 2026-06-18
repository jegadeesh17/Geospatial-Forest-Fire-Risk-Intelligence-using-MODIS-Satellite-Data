import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import sys
import altair as alt

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing import preprocess_data
from feature_engineering import create_features
from visualization import create_fire_map, plot_daily_trend, plot_forecast
from forecasting import train_prophet_model, generate_forecast, evaluate_forecast
from insights import generate_insights

st.set_page_config(page_title="Forest Fire Intelligence", page_icon="🔥", layout="wide")

# Custom CSS for Professional Layout
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-1wivap2 {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI-Powered Geospatial Wildfire Analytics & Forecasting Platform")
st.markdown("An environmental intelligence system combining ML classification, geospatial tracking, and time-series forecasting.")
st.write("") # Force UI reload
st.divider()

@st.cache_data
def load_data():
    # Force cache invalidation to load region_name
    data_path = os.path.join(os.path.dirname(__file__), "../data/modis_data10%.csv")
    if not os.path.exists(data_path):
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    df = create_features(df)
    return df

@st.cache_resource
def get_forecast(_df):
    _cache_buster = 2  # Force cache invalidation
    model, ts_data = train_prophet_model(_df)
    forecast = generate_forecast(model, periods=6, freq='ME')
    metrics = evaluate_forecast(model, _df)
    return forecast, metrics

with st.spinner("Loading Environmental Intelligence Data..."):
    df = load_data()

if df.empty:
    st.error("Data source missing. Please ensure `modis_data10%.csv` is in the `data/` directory.")
    st.stop()

# ================================
# SIDEBAR FILTERS
# ================================
st.sidebar.header("🌍 Platform Controls")
selected_year = st.sidebar.selectbox("Filter Year (Historical)", sorted(df['year'].unique(), reverse=True))
months_avail = sorted(df[df['year'] == selected_year]['month'].unique())
selected_month = st.sidebar.selectbox("Filter Month", months_avail)

filtered_df = df[(df['year'] == selected_year) & (df['month'] == selected_month)]

# ================================
# KPI METRICS SECTION
# ================================
st.subheader("📊 Executive Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"<div class='metric-card'><h4>Total Fire Events</h4><h2>{len(filtered_df):,}</h2></div>", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"<div class='metric-card'><h4>High-Risk Hotspots</h4><h2>{len(filtered_df[filtered_df['high_risk'] == 1]):,}</h2></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div class='metric-card'><h4>Avg FRP</h4><h2>{filtered_df['frp'].mean():.1f} MW</h2></div>", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"<div class='metric-card'><h4>Affected Regions</h4><h2>{filtered_df['region_cluster'].nunique()}</h2></div>", unsafe_allow_html=True)

st.write("") # Spacer

# ================================
# GEOSPATIAL ANALYSIS
# ================================
st.subheader("🗺️ High-Resolution Geospatial Wildfire HeatMap")
st.caption("Visualizing thermal anomalies based on MODIS satellite telemetry. Severity color-coded.")
col_map, col_state = st.columns([2, 1])

with col_map:
    with st.spinner("Rendering mapping engine..."):
        fire_map = create_fire_map(filtered_df, center=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom=5, sample_size=1500)
        st_folium(fire_map, width="stretch", height=500, returned_objects=[])

with col_state:
    st.subheader("Regional Risk Distribution")
    state_counts = filtered_df['region_name'].value_counts().head(5).reset_index()
    state_counts.columns = ['Region', 'Fire Count']
    
    chart = alt.Chart(state_counts).mark_bar().encode(
        x=alt.X('Region', sort='-y', title=None),
        y=alt.Y('Fire Count', title=None),
        tooltip=['Region', 'Fire Count']
    ).properties(height=400)
    
    st.altair_chart(chart, width="stretch")
    st.caption("Top 5 highest risk regions (pseudo-state clusters) for the selected period.")

st.divider()

# ================================
# FORECASTING PIPELINE
# ================================
st.subheader("🔮 6-Month Wildfire Predictive Forecasting")
st.markdown("Time-series projection utilizing the **Prophet** forecasting engine trained on historical thermal telemetry.")

with st.spinner("Training predictive models..."):
    forecast, eval_metrics = get_forecast(df)

col_chart, col_insights = st.columns([2, 1])

with col_chart:
    st.pyplot(plot_forecast(forecast))
    
    with st.expander("Model Evaluation Metrics"):
        st.write(f"**Mean Absolute Error (MAE):** {eval_metrics['MAE']:.2f} events/month")
        st.write(f"**Root Mean Squared Error (RMSE):** {eval_metrics['RMSE']:.2f}")
        st.write(f"**MAPE:** {eval_metrics['MAPE']:.2f}%")

with col_insights:
    st.subheader("🧠 AI Risk Insights")
    insights = generate_insights(df, forecast)
    for insight in insights:
        st.info(insight)

st.divider()
st.subheader("📈 Historical Temporal Trends")
st.pyplot(plot_daily_trend(filtered_df))
