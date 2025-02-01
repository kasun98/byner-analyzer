import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datetime import datetime, timedelta
from pymongo import MongoClient
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from logger import log_error, log_info

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['byner_analyzer']
historical_collection = db['historical_data']
predictions_collection = db['predictions']
metrics_collection = db['metrics']

# Synthentic historical data generation helper functions
def generate_realistic_sequence(n, frequency=0.1, noise_level=1.5):
    """
    Generates a sine wave with added noise.
    
    Parameters:
    - n (int): Number of time steps.
    - frequency (float): Frequency of the sine wave.
    - noise_level (float): Standard deviation of the noise.
    
    Returns:
    - np.array: Integer sequence between -10 and +10.
    """
    x = np.arange(n)
    
    # Generate sine wave
    sine_wave = 8 * np.sin(2 * np.pi * frequency * x)
    
    # Add noise
    noise = np.random.normal(loc=0, scale=noise_level, size=n)
    
    # Apply bounds and round to integer values
    values = np.clip(np.round(sine_wave + noise), -10, 10).astype(int)
    
    return values

def simulate_and_store_data(n):
    """Generate and store realistic market-like data"""
    sequence = generate_realistic_sequence(n)
    
    # Calculate technical indicators (keep your existing code)
    indicators = calculate_technical_indicators(sequence)
    
    # Store in MongoDB
    current_time = datetime.now()
    
    for i in range(n):
        doc = {
            'timestamp': current_time + timedelta(seconds=i),
            'price': int(sequence[i]),
            'indicators': {
                'ma': float(indicators['ma'][i]),
                'rsi': float(indicators['rsi'][i]),
                'upper_bb': float(indicators['upper_bb'][i]),
                'lower_bb': float(indicators['lower_bb'][i]),
                'macd': float(indicators['macd'][i]),
                'signal': float(indicators['signal'][i])
            }
        }
        historical_collection.insert_one(doc)
        
    print("Realistic data generation completed.")

def calculate_technical_indicators(prices):
    """Calculate indicators for modified price sequence"""
    n = len(prices)
    indicators = {
        'ma': np.zeros(n),
        'rsi': np.zeros(n),
        'upper_bb': np.zeros(n),
        'lower_bb': np.zeros(n),
        'macd': np.zeros(n),
        'signal': np.zeros(n)
    }
    
    # 10-period moving average
    ma_window = 10
    ma = np.convolve(prices, np.ones(ma_window)/ma_window, 'valid')
    indicators['ma'][ma_window-1:] = ma
    
    # Bollinger Bands (2 std)
    for i in range(ma_window-1, n):
        if i >= ma_window:
            window = prices[i-ma_window+1:i+1]
            std = np.std(window)
            indicators['upper_bb'][i] = indicators['ma'][i] + 2*std
            indicators['lower_bb'][i] = indicators['ma'][i] - 2*std
    
    # RSI (14-period)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi_period = 14
    
    if len(gains) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gains[:rsi_period])
        avg_loss[rsi_period] = np.mean(losses[:rsi_period])
        
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gains[i-1])/rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + losses[i-1])/rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.ones(n), where=avg_loss!=0)
    indicators['rsi'] = 100 - (100 / (1 + rs))
    indicators['rsi'][:rsi_period] = 50
    
    # MACD (8/17/9 for shorter range)
    ema8 = _calc_ema(prices, 8)
    ema17 = _calc_ema(prices, 17)
    indicators['macd'] = ema8 - ema17
    indicators['signal'] = _calc_ema(indicators['macd'], 9)
    
    return indicators

def _calc_ema(prices, period):
    """Helper function to calculate Exponential Moving Average"""
    ema = np.zeros(len(prices))
    multiplier = 2 / (period + 1)
    
    # Simple MA for first value
    ema[period-1] = np.mean(prices[:period])
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

# Data preprocessing / model training / evaluating
def preprocess_multi_step_data(n_steps=60, forecast_horizon=5):
    """
    Preprocess data for multi-step forecasting
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Retrieve data from MongoDB
    data = list(historical_collection.find().sort("timestamp", 1))
    
    # Extract features and target
    features = []
    target = []
    for d in data:
        features.append([
            d['price'],
            d['indicators']['ma'],
            d['indicators']['rsi'],
            d['indicators']['upper_bb'],
            d['indicators']['lower_bb'],
            d['indicators']['macd'],
            d['indicators']['signal']
        ])
        target.append(d['price'])
    
    features = np.array(features)
    target = np.array(target)
    
    # Normalize features
    # Separate scalers for features and target
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    #target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    #scaled_target = target_scaler.fit_transform(target.reshape(-1, 1))
    
    # Create sequences with multiple targets
    X, y = [], []
    for i in range(len(scaled_features) - n_steps - forecast_horizon + 1):
        X.append(scaled_features[i:i+n_steps])
        y.append(target[i+n_steps:i+n_steps+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train-test split (temporal)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, feature_scaler

def build_multi_step_lstm_model(input_shape, forecast_horizon=5):
    """
    Build multi-output LSTM model architecture
    """
    model = Sequential([
        LSTM(100, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(forecast_horizon)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_multi_step(X_train, X_test, y_train, y_test, feature_scaler):

    # Build model
    model = build_multi_step_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Train model
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    log_info(f"LSTM Model - Test Loss: {loss}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    log_info(f"Made predictions...")
    
    # Store predictions and evaluate
    evaluate_and_store_predictions(y_test, y_pred, X_test, feature_scaler)
    log_info(f"Evaluated and stored predictions...")
    
    return model, history

def evaluate_and_store_predictions(y_true, y_pred, X_test, feature_scaler):
    # Inverse transform predictions
    y_pred = y_pred.round().astype(int)

    # Inverse transform input sequences
    X_test_original = feature_scaler.inverse_transform(X_test.reshape(-1, X_test.shape[2])
                                                       ).reshape(X_test.shape)
    
    # Calculate metrics for each forecast step
    metrics = {
        'mse': [],
        'mae': [],
        'accuracy': []
    }
    
    for step in range(y_pred.shape[1]):
        mse = mean_squared_error(y_true[:, step], y_pred[:, step])
        mae = mean_absolute_error(y_true[:, step], y_pred[:, step])
        accuracy = np.mean(y_true[:, step] == y_pred[:, step])
        
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['accuracy'].append(accuracy)
    
    # Store metrics in MongoDB
    metrics_collection.insert_one({
        'timestamp': datetime.now(),
        'metrics': metrics,
        'forecast_horizon': 5,
        'model_version': 'multi_step_v1'
    })
    
    # Store predictions
    predictions = []
    for i in range(len(y_pred)):
        doc = {
            'timestamp': datetime.now(),
            'input_sequence': X_test_original[i].tolist(),
            'true_next_sequence': y_true[i].tolist(),
            'predicted_next_sequence': y_pred[i].tolist(),
            'model_version': 'multi_step_v1'
        }
        predictions_collection.insert_one(doc)
        predictions.append(doc)
    
    return predictions

def visualize_multi_step_results(y_true, y_pred, n_samples=5):
    plt.figure(figsize=(15, 8))
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(y_true[i], label='Actual')
        plt.plot(y_pred[i], label='Predicted')
        plt.title(f'Sample {i+1} Forecast')
        plt.legend()
    plt.tight_layout()
    plt.savefig('train_output/multi_step_forecasts.png')
    


