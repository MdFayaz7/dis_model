import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from models.hazard_models import MultiHazardPredictor
from data.data_processor import SatelliteDataProcessor
from data.mock_data_generator import MockDataGenerator
from visualization.maps import create_hazard_map
from utils.alerts import AlertSystem

# Configure page
st.set_page_config(
    page_title="Multi-Hazard Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = SatelliteDataProcessor()
if 'predictor' not in st.session_state:
    st.session_state.predictor = MultiHazardPredictor()
if 'mock_generator' not in st.session_state:
    st.session_state.mock_generator = MockDataGenerator()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()

def main():
    # Header
    st.title("🌍 Multi-Hazard Prediction System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard Overview", "Data Processing", "Drought Prediction", 
         "Flood Prediction", "Earthquake Prediction", "Alert Management"]
    )
    
    # Main content based on selected page
    if page == "Dashboard Overview":
        dashboard_overview()
    elif page == "Data Processing":
        data_processing_page()
    elif page == "Drought Prediction":
        drought_prediction_page()
    elif page == "Flood Prediction":
        flood_prediction_page()
    elif page == "Earthquake Prediction":
        earthquake_prediction_page()
    elif page == "Alert Management":
        alert_management_page()

def dashboard_overview():
    st.header("🎯 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Monitoring Regions",
            value="127",
            delta="5"
        )
    
    with col2:
        st.metric(
            label="Current Alerts",
            value="8",
            delta="-2"
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value="94.2%",
            delta="1.3%"
        )
    
    with col4:
        st.metric(
            label="Data Sources",
            value="12",
            delta="1"
        )
    
    # Real-time map
    st.subheader("🗺️ Global Hazard Overview")
    
    # Generate mock global data for demonstration
    regions_data = st.session_state.mock_generator.generate_global_hazard_data()
    
    # Create map
    hazard_map = create_hazard_map(regions_data)
    st.components.v1.html(hazard_map._repr_html_(), height=500)
    
    # Recent predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Recent Predictions")
        predictions_df = st.session_state.mock_generator.generate_recent_predictions()
        st.dataframe(predictions_df, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Active Alerts")
        alerts_df = st.session_state.alert_system.get_active_alerts()
        st.dataframe(alerts_df, use_container_width=True)

def data_processing_page():
    st.header("📡 Satellite Data Processing")
    
    # Data source selection
    st.subheader("Data Sources Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Optical Satellite Data**")
        sentinel2_enabled = st.checkbox("Sentinel-2", value=True)
        landsat_enabled = st.checkbox("Landsat 8/9", value=True)
        modis_enabled = st.checkbox("MODIS", value=True)
    
    with col2:
        st.markdown("**SAR & Geophysical Data**")
        sentinel1_enabled = st.checkbox("Sentinel-1 SAR", value=True)
        seismic_enabled = st.checkbox("USGS Seismic", value=True)
        weather_enabled = st.checkbox("Weather Stations", value=True)
    
    # Processing parameters
    st.subheader("Processing Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spatial_resolution = st.selectbox(
            "Spatial Resolution (m)",
            [10, 30, 100, 250, 500]
        )
    
    with col2:
        temporal_window = st.selectbox(
            "Temporal Window (days)",
            [7, 14, 30, 60, 90]
        )
    
    with col3:
        region_size = st.selectbox(
            "Region Size (km²)",
            [100, 500, 1000, 5000, 10000]
        )
    
    # Mock data processing demonstration
    if st.button("Process Sample Data", type="primary"):
        with st.spinner("Processing satellite data..."):
            # Simulate data processing
            processed_data = st.session_state.data_processor.process_mock_data(
                spatial_resolution, temporal_window, region_size
            )
            
            st.success("Data processing completed!")
            
            # Display processing results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("NDVI Time Series")
                fig = px.line(
                    processed_data['ndvi_timeseries'],
                    x='date', y='ndvi',
                    title="Normalized Difference Vegetation Index"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Soil Moisture")
                fig = px.line(
                    processed_data['soil_moisture'],
                    x='date', y='moisture',
                    title="Soil Moisture Content (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature extraction results
            st.subheader("Extracted Features")
            features_df = processed_data['features']
            st.dataframe(features_df, use_container_width=True)

def drought_prediction_page():
    from pages.drought_prediction import render_drought_page
    render_drought_page()

def flood_prediction_page():
    from pages.flood_prediction import render_flood_page
    render_flood_page()

def earthquake_prediction_page():
    from pages.earthquake_prediction import render_earthquake_page
    render_earthquake_page()

def alert_management_page():
    st.header("⚠️ Alert Management System")
    
    # Alert configuration
    st.subheader("Alert Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Drought Alerts**")
        drought_low = st.slider("Low Risk", 0.0, 1.0, 0.3, key="drought_low")
        drought_high = st.slider("High Risk", 0.0, 1.0, 0.7, key="drought_high")
    
    with col2:
        st.markdown("**Flood Alerts**")
        flood_low = st.slider("Low Risk", 0.0, 1.0, 0.4, key="flood_low")
        flood_high = st.slider("High Risk", 0.0, 1.0, 0.8, key="flood_high")
    
    with col3:
        st.markdown("**Earthquake Alerts**")
        earthquake_low = st.slider("Low Risk", 0.0, 1.0, 0.2, key="earthquake_low")
        earthquake_high = st.slider("High Risk", 0.0, 1.0, 0.6, key="earthquake_high")
    
    # Alert history
    st.subheader("Alert History")
    
    # Generate alert history data
    alert_history = st.session_state.alert_system.get_alert_history()
    
    # Alert timeline
    fig = px.timeline(
        alert_history,
        x_start="start_time",
        x_end="end_time",
        y="region",
        color="hazard_type",
        title="Alert Timeline"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Alert frequency by type
        alert_counts = alert_history['hazard_type'].value_counts()
        fig = px.pie(
            values=alert_counts.values,
            names=alert_counts.index,
            title="Alert Distribution by Hazard Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Alert severity distribution
        severity_counts = alert_history['severity'].value_counts()
        fig = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            title="Alert Distribution by Severity"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
