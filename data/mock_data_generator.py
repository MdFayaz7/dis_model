import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

class MockDataGenerator:
    """
    Generates realistic mock satellite and geophysical data for demonstration purposes.
    """
    
    def __init__(self):
        self.regions = [
            {'name': 'California Central Valley', 'lat': 36.7783, 'lon': -119.4179, 'type': 'agricultural'},
            {'name': 'Amazon Basin', 'lat': -3.4653, 'lon': -62.2159, 'type': 'forest'},
            {'name': 'Sahel Region', 'lat': 15.0000, 'lon': 0.0000, 'type': 'semi-arid'},
            {'name': 'Ganges Delta', 'lat': 23.6345, 'lon': 90.2934, 'type': 'delta'},
            {'name': 'Ring of Fire - Japan', 'lat': 35.6762, 'lon': 139.6503, 'type': 'seismic'},
            {'name': 'Australian Outback', 'lat': -25.2744, 'lon': 133.7751, 'type': 'arid'},
            {'name': 'European Plains', 'lat': 52.5200, 'lon': 13.4050, 'type': 'temperate'},
            {'name': 'East African Rift', 'lat': -1.2921, 'lon': 36.8219, 'type': 'seismic'}
        ]
        
        self.hazard_types = ['drought', 'flood', 'earthquake']
        self.severity_levels = ['Low', 'Medium', 'High', 'Critical']
    
    def generate_ndvi_data(self, days: int = 365, lat: float = 0.0, 
                          region_type: str = 'temperate') -> pd.DataFrame:
        """
        Generate realistic NDVI time series data.
        """
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        
        # Base NDVI values by region type
        base_ndvi = {
            'forest': 0.8,
            'agricultural': 0.6,
            'semi-arid': 0.3,
            'arid': 0.15,
            'delta': 0.7,
            'temperate': 0.5,
            'seismic': 0.6
        }
        
        base_value = base_ndvi.get(region_type, 0.5)
        
        # Seasonal variation
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal_component = 0.2 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/2)
        
        # Add some random variation and trends
        trend = np.linspace(0, 0.05 * np.random.randn(), days)
        noise = np.random.normal(0, 0.03, days)
        
        # Apply drought stress randomly
        drought_events = np.random.choice([0, 1], size=days, p=[0.95, 0.05])
        drought_stress = drought_events * np.random.uniform(-0.3, -0.1, days)
        
        ndvi_values = base_value + seasonal_component + trend + noise + drought_stress
        ndvi_values = np.clip(ndvi_values, 0, 1)
        
        return pd.DataFrame({
            'date': dates,
            'ndvi': ndvi_values,
            'region_type': region_type
        })
    
    def generate_soil_moisture_data(self, days: int = 365, lat: float = 0.0,
                                  region_type: str = 'temperate') -> pd.DataFrame:
        """
        Generate realistic soil moisture time series data.
        """
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        
        # Base soil moisture by region type
        base_moisture = {
            'forest': 0.6,
            'agricultural': 0.4,
            'semi-arid': 0.2,
            'arid': 0.1,
            'delta': 0.8,
            'temperate': 0.5,
            'seismic': 0.4
        }
        
        base_value = base_moisture.get(region_type, 0.4)
        
        # Seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal_component = 0.15 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Precipitation events
        precip_events = np.random.poisson(0.3, days)  # Average 0.3 events per day
        precip_effect = np.convolve(precip_events, np.exp(-np.arange(10)/3), mode='same')[:days]
        precip_effect = precip_effect / np.max(precip_effect) * 0.2
        
        # Random variation
        noise = np.random.normal(0, 0.02, days)
        
        moisture_values = base_value + seasonal_component + precip_effect + noise
        moisture_values = np.clip(moisture_values, 0, 1)
        
        return pd.DataFrame({
            'date': dates,
            'soil_moisture': moisture_values,
            'region_type': region_type
        })
    
    def generate_seismic_data(self, days: int = 365, lat: float = 0.0,
                            is_seismic_region: bool = False) -> List[Dict]:
        """
        Generate realistic seismic activity data.
        """
        events = []
        
        # Base earthquake frequency (events per day)
        base_frequency = 0.5 if is_seismic_region else 0.1
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            
            # Determine if earthquake occurs
            if np.random.random() < base_frequency:
                # Generate magnitude following Gutenberg-Richter law
                magnitude = self._generate_earthquake_magnitude()
                
                # Generate location near the region
                lat_offset = np.random.normal(0, 0.5)
                lon_offset = np.random.normal(0, 0.5)
                
                events.append({
                    'date': date,
                    'latitude': lat + lat_offset,
                    'longitude': lat + lon_offset,  # Using lat as approximate longitude
                    'magnitude': magnitude,
                    'depth': np.random.uniform(5, 200),  # km
                    'type': 'earthquake'
                })
        
        return events
    
    def generate_weather_data(self, days: int = 365, lat: float = 0.0,
                            region_type: str = 'temperate') -> pd.DataFrame:
        """
        Generate realistic weather data.
        """
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        
        # Base temperature by region and season
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # Temperature patterns by region
        temp_patterns = {
            'forest': {'base': 25, 'seasonal_amp': 10, 'daily_var': 3},
            'agricultural': {'base': 20, 'seasonal_amp': 15, 'daily_var': 4},
            'semi-arid': {'base': 30, 'seasonal_amp': 12, 'daily_var': 6},
            'arid': {'base': 35, 'seasonal_amp': 15, 'daily_var': 8},
            'delta': {'base': 28, 'seasonal_amp': 8, 'daily_var': 2},
            'temperate': {'base': 15, 'seasonal_amp': 20, 'daily_var': 4},
            'seismic': {'base': 22, 'seasonal_amp': 12, 'daily_var': 3}
        }
        
        pattern = temp_patterns.get(region_type, temp_patterns['temperate'])
        
        # Seasonal temperature variation
        seasonal_temp = pattern['seasonal_amp'] * np.sin(2 * np.pi * day_of_year / 365.25)
        base_temp = pattern['base'] + seasonal_temp
        
        # Daily temperature variation
        daily_variation = np.random.normal(0, pattern['daily_var'], days)
        temperature = base_temp + daily_variation
        
        # Precipitation patterns
        # Wet season probability
        wet_season_prob = 0.5 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi)
        precip_occurs = np.random.random(days) < (wet_season_prob * 0.3)
        
        precipitation = np.zeros(days)
        precipitation[precip_occurs] = np.random.exponential(10, np.sum(precip_occurs))
        
        # Humidity patterns
        humidity_base = 60 if region_type != 'arid' else 30
        humidity_seasonal = 20 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/4)
        humidity_daily = np.random.normal(0, 5, days)
        humidity = humidity_base + humidity_seasonal + humidity_daily
        humidity = np.clip(humidity, 0, 100)
        
        return pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'precipitation': precipitation,
            'humidity': humidity,
            'region_type': region_type
        })
    
    def generate_global_hazard_data(self) -> List[Dict]:
        """
        Generate global hazard monitoring data for the overview map.
        """
        global_data = []
        
        for region in self.regions:
            # Generate current hazard levels for each region
            for hazard_type in self.hazard_types:
                # Determine risk level based on region type and hazard type
                risk_weights = self._get_risk_weights(region['type'], hazard_type)
                risk_level = np.random.choice(
                    self.severity_levels,
                    p=risk_weights
                )
                
                # Generate specific risk metrics
                risk_score = self._severity_to_score(risk_level)
                confidence = np.random.uniform(0.7, 0.95)
                
                global_data.append({
                    'region': region['name'],
                    'latitude': region['lat'],
                    'longitude': region['lon'],
                    'region_type': region['type'],
                    'hazard_type': hazard_type,
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'confidence': confidence,
                    'last_updated': datetime.now() - timedelta(hours=np.random.randint(0, 24))
                })
        
        return global_data
    
    def generate_recent_predictions(self, num_predictions: int = 10) -> pd.DataFrame:
        """
        Generate recent prediction data for the dashboard.
        """
        predictions = []
        
        for i in range(num_predictions):
            region = np.random.choice(self.regions)
            hazard_type = np.random.choice(self.hazard_types)
            
            risk_weights = self._get_risk_weights(region['type'], hazard_type)
            risk_level = np.random.choice(self.severity_levels, p=risk_weights)
            
            predictions.append({
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 48)),
                'region': region['name'],
                'hazard_type': hazard_type.title(),
                'risk_level': risk_level,
                'confidence': f"{np.random.uniform(75, 95):.1f}%",
                'model_version': f"v{np.random.randint(1, 5)}.{np.random.randint(0, 10)}"
            })
        
        df = pd.DataFrame(predictions)
        return df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    def _generate_earthquake_magnitude(self) -> float:
        """
        Generate earthquake magnitude following Gutenberg-Richter distribution.
        """
        # Simplified Gutenberg-Richter: N = 10^(a - b*M)
        # Most earthquakes are small magnitude
        rand = np.random.random()
        if rand < 0.7:
            return np.random.uniform(1.0, 3.0)
        elif rand < 0.9:
            return np.random.uniform(3.0, 5.0)
        elif rand < 0.98:
            return np.random.uniform(5.0, 7.0)
        else:
            return np.random.uniform(7.0, 9.0)
    
    def _get_risk_weights(self, region_type: str, hazard_type: str) -> List[float]:
        """
        Get probability weights for risk levels based on region and hazard type.
        """
        # Risk probability matrices
        risk_matrices = {
            'drought': {
                'forest': [0.7, 0.2, 0.08, 0.02],  # Low, Medium, High, Critical
                'agricultural': [0.5, 0.3, 0.15, 0.05],
                'semi-arid': [0.3, 0.4, 0.25, 0.05],
                'arid': [0.2, 0.3, 0.35, 0.15],
                'delta': [0.8, 0.15, 0.04, 0.01],
                'temperate': [0.6, 0.25, 0.12, 0.03],
                'seismic': [0.6, 0.25, 0.12, 0.03]
            },
            'flood': {
                'forest': [0.8, 0.15, 0.04, 0.01],
                'agricultural': [0.6, 0.25, 0.12, 0.03],
                'semi-arid': [0.85, 0.12, 0.02, 0.01],
                'arid': [0.9, 0.08, 0.015, 0.005],
                'delta': [0.3, 0.4, 0.25, 0.05],
                'temperate': [0.7, 0.2, 0.08, 0.02],
                'seismic': [0.7, 0.2, 0.08, 0.02]
            },
            'earthquake': {
                'forest': [0.9, 0.08, 0.015, 0.005],
                'agricultural': [0.85, 0.12, 0.025, 0.005],
                'semi-arid': [0.9, 0.08, 0.015, 0.005],
                'arid': [0.9, 0.08, 0.015, 0.005],
                'delta': [0.8, 0.15, 0.04, 0.01],
                'temperate': [0.9, 0.08, 0.015, 0.005],
                'seismic': [0.4, 0.35, 0.2, 0.05]
            }
        }
        
        return risk_matrices.get(hazard_type, {}).get(region_type, [0.7, 0.2, 0.08, 0.02])
    
    def _severity_to_score(self, severity: str) -> float:
        """
        Convert severity level to numerical score.
        """
        score_map = {
            'Low': np.random.uniform(0.0, 0.3),
            'Medium': np.random.uniform(0.3, 0.6),
            'High': np.random.uniform(0.6, 0.8),
            'Critical': np.random.uniform(0.8, 1.0)
        }
        return score_map.get(severity, 0.5)
