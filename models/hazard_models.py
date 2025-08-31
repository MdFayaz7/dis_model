import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
from datetime import datetime
import logging

class MultiHazardPredictor:
    """
    Multi-hazard prediction model combining drought, flood, and earthquake prediction.
    Uses ensemble methods and neural networks for robust predictions.
    """
    
    def __init__(self):
        self.drought_model = None
        self.flood_model = None
        self.earthquake_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Model configurations
        self.drought_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        self.flood_config = {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 500,
            'random_state': 42
        }
        
        self.earthquake_config = {
            'n_estimators': 150,
            'max_depth': 15,
            'random_state': 42
        }
    
    def prepare_features(self, data):
        """
        Prepare feature matrix from satellite and geophysical data.
        """
        features = []
        feature_names = []
        
        # NDVI features
        if 'ndvi' in data:
            features.extend([
                np.mean(data['ndvi']),
                np.std(data['ndvi']),
                np.min(data['ndvi']),
                np.max(data['ndvi'])
            ])
            feature_names.extend(['ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max'])
        
        # Soil moisture features
        if 'soil_moisture' in data:
            features.extend([
                np.mean(data['soil_moisture']),
                np.std(data['soil_moisture']),
                np.percentile(data['soil_moisture'], 25),
                np.percentile(data['soil_moisture'], 75)
            ])
            feature_names.extend(['moisture_mean', 'moisture_std', 'moisture_q25', 'moisture_q75'])
        
        # Temperature features
        if 'temperature' in data:
            features.extend([
                np.mean(data['temperature']),
                np.max(data['temperature']),
                np.std(data['temperature'])
            ])
            feature_names.extend(['temp_mean', 'temp_max', 'temp_std'])
        
        # Precipitation features
        if 'precipitation' in data:
            features.extend([
                np.sum(data['precipitation']),
                np.mean(data['precipitation']),
                np.max(data['precipitation'])
            ])
            feature_names.extend(['precip_sum', 'precip_mean', 'precip_max'])
        
        # Elevation and topographic features
        if 'elevation' in data:
            features.extend([
                np.mean(data['elevation']),
                np.std(data['elevation'])
            ])
            feature_names.extend(['elevation_mean', 'elevation_std'])
        
        # Seismic activity features
        if 'seismic_activity' in data:
            features.extend([
                np.mean(data['seismic_activity']),
                np.max(data['seismic_activity']),
                len(data['seismic_activity'])
            ])
            feature_names.extend(['seismic_mean', 'seismic_max', 'seismic_count'])
        
        return np.array(features), feature_names
    
    def train_drought_model(self, training_data, labels):
        """
        Train drought prediction model using Random Forest.
        """
        self.drought_model = RandomForestClassifier(**self.drought_config)
        
        # Prepare features
        X = []
        for data_point in training_data:
            features, _ = self.prepare_features(data_point)
            X.append(features)
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.drought_model.fit(X_scaled, labels)
        
        return self.drought_model
    
    def train_flood_model(self, training_data, labels):
        """
        Train flood prediction model using Neural Network.
        """
        self.flood_model = MLPClassifier(**self.flood_config)
        
        # Prepare features
        X = []
        for data_point in training_data:
            features, _ = self.prepare_features(data_point)
            X.append(features)
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.flood_model.fit(X_scaled, labels)
        
        return self.flood_model
    
    def train_earthquake_model(self, training_data, labels):
        """
        Train earthquake prediction model using Random Forest.
        """
        self.earthquake_model = RandomForestRegressor(**self.earthquake_config)
        
        # Prepare features
        X = []
        for data_point in training_data:
            features, _ = self.prepare_features(data_point)
            X.append(features)
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.earthquake_model.fit(X_scaled, labels)
        
        return self.earthquake_model
    
    def predict_drought_risk(self, data):
        """
        Predict drought risk for given satellite data.
        """
        if self.drought_model is None:
            # Return mock prediction for demonstration
            return {
                'risk_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1]),
                'probability': np.random.uniform(0.1, 0.9),
                'confidence': np.random.uniform(0.7, 0.95),
                'factors': {
                    'ndvi_trend': np.random.uniform(-0.2, 0.1),
                    'soil_moisture': np.random.uniform(0.1, 0.4),
                    'temperature_anomaly': np.random.uniform(-2, 5)
                }
            }
        
        features, feature_names = self.prepare_features(data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.drought_model.predict(features_scaled)[0]
        probability = self.drought_model.predict_proba(features_scaled)[0]
        
        return {
            'risk_level': prediction,
            'probability': np.max(probability),
            'confidence': np.max(probability),
            'feature_importance': dict(zip(feature_names, self.drought_model.feature_importances_))
        }
    
    def predict_flood_risk(self, data):
        """
        Predict flood risk for given satellite data.
        """
        if self.flood_model is None:
            # Return mock prediction for demonstration
            return {
                'risk_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.35, 0.15]),
                'probability': np.random.uniform(0.1, 0.8),
                'confidence': np.random.uniform(0.7, 0.9),
                'factors': {
                    'precipitation_forecast': np.random.uniform(0, 50),
                    'soil_saturation': np.random.uniform(0.3, 0.9),
                    'river_level': np.random.uniform(0.5, 1.2)
                }
            }
        
        features, feature_names = self.prepare_features(data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.flood_model.predict(features_scaled)[0]
        probability = self.flood_model.predict_proba(features_scaled)[0]
        
        return {
            'risk_level': prediction,
            'probability': np.max(probability),
            'confidence': np.max(probability)
        }
    
    def predict_earthquake_risk(self, data):
        """
        Predict earthquake risk for given geophysical data.
        """
        if self.earthquake_model is None:
            # Return mock prediction for demonstration
            return {
                'magnitude_prediction': np.random.uniform(2.0, 6.5),
                'probability': np.random.uniform(0.05, 0.3),
                'confidence': np.random.uniform(0.6, 0.85),
                'factors': {
                    'seismic_activity': np.random.uniform(0.1, 0.8),
                    'tectonic_stress': np.random.uniform(0.2, 0.9),
                    'historical_frequency': np.random.uniform(0.1, 0.6)
                }
            }
        
        features, feature_names = self.prepare_features(data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.earthquake_model.predict(features_scaled)[0]
        
        return {
            'magnitude_prediction': prediction,
            'probability': min(prediction / 10.0, 1.0),  # Convert to probability
            'confidence': 0.8
        }
    
    def unified_prediction(self, data):
        """
        Generate unified multi-hazard prediction.
        """
        drought_pred = self.predict_drought_risk(data)
        flood_pred = self.predict_flood_risk(data)
        earthquake_pred = self.predict_earthquake_risk(data)
        
        # Calculate overall risk score
        drought_score = drought_pred['probability'] if drought_pred['risk_level'] == 'High' else drought_pred['probability'] * 0.5
        flood_score = flood_pred['probability'] if flood_pred['risk_level'] == 'High' else flood_pred['probability'] * 0.5
        earthquake_score = earthquake_pred['probability']
        
        overall_risk = max(drought_score, flood_score, earthquake_score)
        
        return {
            'overall_risk': overall_risk,
            'primary_hazard': max(
                [('drought', drought_score), ('flood', flood_score), ('earthquake', earthquake_score)],
                key=lambda x: x[1]
            )[0],
            'individual_predictions': {
                'drought': drought_pred,
                'flood': flood_pred,
                'earthquake': earthquake_pred
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def save_models(self, filepath):
        """
        Save trained models to disk.
        """
        model_data = {
            'drought_model': self.drought_model,
            'flood_model': self.flood_model,
            'earthquake_model': self.earthquake_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath):
        """
        Load trained models from disk.
        """
        try:
            model_data = joblib.load(filepath)
            self.drought_model = model_data['drought_model']
            self.flood_model = model_data['flood_model']
            self.earthquake_model = model_data['earthquake_model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            return False

class CNNLSTMModel:
    """
    Convolutional LSTM model for temporal satellite imagery analysis.
    This would be used for more complex spatio-temporal pattern recognition.
    """
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """
        Build CNN-LSTM architecture for satellite image sequence analysis.
        Note: This is a conceptual structure - would require TensorFlow/PyTorch for full implementation.
        """
        # This would be implemented with TensorFlow/PyTorch
        # For now, returning a placeholder structure
        model_architecture = {
            'conv_layers': [
                {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'}
            ],
            'lstm_layers': [
                {'units': 50, 'return_sequences': True},
                {'units': 50, 'return_sequences': False}
            ],
            'dense_layers': [
                {'units': 100, 'activation': 'relu'},
                {'units': self.num_classes, 'activation': 'softmax'}
            ]
        }
        return model_architecture
    
    def preprocess_satellite_sequence(self, image_sequence):
        """
        Preprocess satellite image sequences for model input.
        """
        # Normalize pixel values
        normalized_sequence = image_sequence / 255.0
        
        # Apply data augmentation techniques
        # This would include rotation, flipping, etc.
        
        return normalized_sequence
