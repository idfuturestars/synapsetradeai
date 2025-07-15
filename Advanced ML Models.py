#!/usr/bin/env python3
"""
SynapseTrade AI™ - Advanced Machine Learning Models
Chief Technical Architect Implementation
Including LSTM, GRU, Transformer, Ensemble Models
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# Advanced ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    print("Advanced ML libraries not available")

class ModelBackup:
    """Backup system for ML models"""
    
    @staticmethod
    def create_backup(model_name: str, model_data: dict):
        """Create backup before model update"""
        backup_dir = f"model_backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Save model metadata
        metadata = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "version": model_data.get("version", "1.0"),
            "metrics": model_data.get("metrics", {})
        }
        
        with open(f"{backup_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model backup created: {backup_dir}")
        return backup_dir

class AdvancedLSTM:
    """Advanced LSTM with attention mechanism"""
    
    def __init__(self, sequence_length=60, n_features=5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.training_history = None
        
    def build_model(self, lstm_units=[128, 64], dropout_rate=0.2):
        """Build advanced LSTM model with attention"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available")
            return None
            
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer
        lstm_out = layers.LSTM(
            lstm_units[0], 
            return_sequences=True,
            dropout=dropout_rate
        )(inputs)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=lstm_units[0]
        )(lstm_out, lstm_out)
        
        # Add & Norm
        attention_out = layers.LayerNormalization()(layers.Add()([lstm_out, attention]))
        
        # Second LSTM layer
        lstm_out2 = layers.LSTM(
            lstm_units[1],
            dropout=dropout_rate
        )(attention_out)
        
        # Dense layers
        dense1 = layers.Dense(32, activation='relu')(lstm_out2)
        dense1 = layers.Dropout(dropout_rate)(dense1)
        dense2 = layers.Dense(16, activation='relu')(dense1)
        
        # Output layer
        outputs = layers.Dense(1)(dense2)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return self.model
    
    def prepare_data(self, data: pd.DataFrame, target_col='close'):
        """Prepare data with advanced feature engineering"""
        # Technical indicators
        data['returns'] = data[target_col].pct_change()
        data['log_returns'] = np.log(data[target_col] / data[target_col].shift(1))
        data['volatility'] = data['returns'].rolling(20).std()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Price features
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            data[f'ma_{period}'] = data[target_col].rolling(period).mean()
            data[f'ma_ratio_{period}'] = data[target_col] / data[f'ma_{period}']
        
        # RSI
        delta = data[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Clean data
        data = data.dropna()
        
        # Select features
        feature_cols = [
            'close', 'volume_ratio', 'volatility', 'high_low_ratio',
            'close_open_ratio', 'ma_ratio_5', 'ma_ratio_20', 'rsi'
        ]
        
        return data[feature_cols]
    
    def create_sequences(self, data: np.ndarray, target_idx=0):
        """Create sequences for LSTM"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model with advanced callbacks"""
        # Backup before training
        ModelBackup.create_backup("AdvancedLSTM", {"version": "2.0"})
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        scaled_data = self.scaler.fit_transform(prepared_data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'models/lstm_best.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Train
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.training_history

class TransformerModel:
    """Transformer architecture for time series"""
    
    def __init__(self, sequence_length=60, n_features=5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        if not TENSORFLOW_AVAILABLE:
            return inputs
            
        # Multi-head attention
        x = layers.MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        
        # Feed forward
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        return x + res
    
    def build_model(self):
        """Build transformer model"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.n_features
        )(positions)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(3):
            x = self.transformer_encoder(
                x,
                head_size=256,
                num_heads=4,
                ff_dim=256,
                dropout=0.1
            )
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        return self.model

class GRUModel:
    """Gated Recurrent Unit model"""
    
    def __init__(self, sequence_length=60, n_features=5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        
    def build_model(self):
        """Build GRU model"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        self.model = keras.Sequential([
            layers.GRU(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(0.2),
            layers.GRU(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.GRU(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model

class EnsembleModel:
    """Ensemble of multiple models"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.predictions = {}
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def train_all(self, X_train, y_train, X_val, y_val):
        """Train all models in ensemble"""
        print("\n🤖 Training Ensemble Models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Backup before training
            ModelBackup.create_backup(name, {"ensemble": True})
            
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                
                results[name] = {
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'r2_score': r2_score(y_val, val_pred)
                }
                
                print(f"  Train RMSE: {train_rmse:.4f}")
                print(f"  Val RMSE: {val_rmse:.4f}")
                print(f"  R² Score: {results[name]['r2_score']:.4f}")
        
        return results
    
    def predict(self, X):
        """Weighted ensemble prediction"""
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                predictions[name] = model.predict(X) * self.weights[name]
        
        # Weighted average
        total_weight = sum(self.weights.values())
        ensemble_pred = sum(predictions.values()) / total_weight
        
        return ensemble_pred

class MLModelFactory:
    """Factory for creating and managing ML models"""
    
    @staticmethod
    def create_ensemble():
        """Create comprehensive ensemble model"""
        ensemble = EnsembleModel()
        
        # Add traditional ML models
        if ADVANCED_ML_AVAILABLE:
            # Random Forest
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            ensemble.add_model("RandomForest", rf, weight=1.0)
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            ensemble.add_model("GradientBoosting", gb, weight=1.2)
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            ensemble.add_model("XGBoost", xgb_model, weight=1.5)
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42
            )
            ensemble.add_model("LightGBM", lgb_model, weight=1.3)
        
        return ensemble
    
    @staticmethod
    def create_deep_ensemble():
        """Create deep learning ensemble"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available for deep ensemble")
            return None
            
        deep_ensemble = EnsembleModel()
        
        # LSTM
        lstm = AdvancedLSTM()
        lstm.build_model()
        deep_ensemble.add_model("LSTM", lstm, weight=1.5)
        
        # GRU
        gru = GRUModel()
        gru.build_model()
        deep_ensemble.add_model("GRU", gru, weight=1.2)
        
        # Transformer
        transformer = TransformerModel()
        transformer.build_model()
        deep_ensemble.add_model("Transformer", transformer, weight=1.8)
        
        return deep_ensemble

class ModelDiagnostics:
    """Comprehensive model diagnostics"""
    
    @staticmethod
    def run_diagnostics(model, X_test, y_test):
        """Run comprehensive diagnostics"""
        print("\n🔍 Model Diagnostics")
        print("=" * 50)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        y_test_direction = np.sign(np.diff(y_test))
        y_pred_direction = np.sign(np.diff(y_pred.flatten()))
        directional_accuracy = np.mean(y_test_direction == y_pred_direction)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")
        
        # Error distribution
        errors = y_test - y_pred.flatten()
        print(f"\nError Distribution:")
        print(f"  Mean Error: {np.mean(errors):.4f}")
        print(f"  Std Error: {np.std(errors):.4f}")
        print(f"  Max Error: {np.max(np.abs(errors)):.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }

class AutoML:
    """Automated machine learning pipeline"""
    
    def __init__(self):
        self.best_model = None
        self.best_score = float('inf')
        self.results = {}
        
    def auto_train(self, data: pd.DataFrame, target_col: str, test_size: float = 0.2):
        """Automatically train and select best model"""
        print("\n🚀 AutoML Training Pipeline")
        print("=" * 50)
        
        # Prepare features
        feature_eng = AdvancedLSTM()
        prepared_data = feature_eng.prepare_data(data, target_col)
        
        # Create train/test split
        X = prepared_data.drop(columns=[target_col]).values
        y = prepared_data[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Create and train ensemble
        ensemble = MLModelFactory.create_ensemble()
        
        if ensemble:
            results = ensemble.train_all(X_train, y_train, X_test, y_test)
            
            # Find best model
            for name, metrics in results.items():
                if metrics['val_rmse'] < self.best_score:
                    self.best_score = metrics['val_rmse']
                    self.best_model = name
                    
            print(f"\n🏆 Best Model: {self.best_model} (RMSE: {self.best_score:.4f})")
            
            # Run diagnostics on best model
            best_model_obj = ensemble.models[self.best_model]
            diagnostics = ModelDiagnostics.run_diagnostics(best_model_obj, X_test, y_test)
            
            return ensemble, diagnostics
        
        return None, None

# Model serving API
class ModelServer:
    """Serve ML models via API"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models"""
        # Load from disk if available
        model_dir = "models"
        if os.path.exists(model_dir):
            for model_file in os.listdir(model_dir):
                if model_file.endswith('.pkl'):
                    model_name = model_file.replace('.pkl', '')
                    with open(f"{model_dir}/{model_file}", 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                        print(f"Loaded model: {model_name}")
    
    def predict(self, model_name: str, features: np.ndarray):
        """Get prediction from specific model"""
        if model_name in self.models:
            return self.models[model_name].predict(features)
        else:
            return None
    
    def get_available_models(self):
        """List available models"""
        return list(self.models.keys())

# Example usage
if __name__ == "__main__":
    print("SynapseTrade AI™ - Advanced ML Models")
    print("=" * 50)
    
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'high': 102 + np.cumsum(np.random.randn(len(dates)) * 2),
        'low': 98 + np.cumsum(np.random.randn(len(dates)) * 2),
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Run AutoML
    automl = AutoML()
    best_model, diagnostics = automl.auto_train(sample_data, 'close')
    
    print("\n✅ Advanced ML Models Ready for Integration!")