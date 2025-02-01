from flask import Flask, jsonify, Response
from pymongo import MongoClient
import numpy as np
import time
import json
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)

# MongoDB client and collection setup
client = MongoClient('mongodb://localhost:27017')  # Update with your MongoDB URI
db = client['byner_analyzer']
historical_collection = db['historical_data']

def get_last_50_prices():
    """Retrieve the last 50 prices from MongoDB"""
    cursor = historical_collection.find().sort("timestamp", -1).limit(50)
    prices = [doc['price'] for doc in cursor]
    return prices[::-1]  # Reverse the list to get oldest to newest

def generate_realistic_sequence(n, initial_data):
    """Generate n new prices, starting from the last known data"""
    prices = np.array(initial_data)
    new_prices = _generate_realistic_sequence(n)
    prices = np.append(prices, new_prices)
    
    return prices

def _generate_realistic_sequence(n, frequency=0.1, noise_level=1.5):
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

def calculate_technical_indicators(prices):
    """Calculate technical indicators based on updated prices"""
    n = len(prices)
    indicators = {
        'ma': np.zeros(n),
        'rsi': np.zeros(n),
        'upper_bb': np.zeros(n),
        'lower_bb': np.zeros(n),
        'macd': np.zeros(n),
        'signal': np.zeros(n)
    }
    
    # Calculate moving average (MA)
    ma_window = 10
    ma = np.convolve(prices, np.ones(ma_window)/ma_window, 'valid')
    indicators['ma'][ma_window-1:] = ma
    
    # Bollinger Bands (BB)
    for i in range(ma_window-1, n):
        if i >= ma_window:
            window = prices[i-ma_window+1:i+1]
            std = np.std(window)
            indicators['upper_bb'][i] = indicators['ma'][i] + 2*std
            indicators['lower_bb'][i] = indicators['ma'][i] - 2*std
    
    # Calculate RSI (14-period)
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
    
    # MACD (8/17/9)
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

# Function to stream the data
def stream_data():
    """Stream the data every second with the latest prices and indicators"""
    last_50_prices = get_last_50_prices()  # Retrieve the last 50 prices from MongoDB
    combined_prices = generate_realistic_sequence(10000, last_50_prices)  # Generate 1000 new prices
    
    # Calculate technical indicators for the updated price sequence
    indicators = calculate_technical_indicators(combined_prices)
    
    # Streaming data every second
    start_time = datetime.now()
    for i in range(50, len(combined_prices)):  # Start from the 51st price
        timestamp = start_time + timedelta(seconds=i-50)
        data_point = {
            'timestamp': timestamp.isoformat(),
            'price': int(combined_prices[i]),
            'indicators': {
                'ma': float(indicators['ma'][i]),
                'rsi': float(indicators['rsi'][i]),
                'upper_bb': float(indicators['upper_bb'][i]),
                'lower_bb': float(indicators['lower_bb'][i]),
                'macd': float(indicators['macd'][i]),
                'signal': float(indicators['signal'][i])
            }
        }
        
        yield f"data: {json.dumps(data_point)}\n\n"  # Format as server-sent events
        time.sleep(1)

# Real-time streaming endpoint
@app.route("/realtime_data")
def get_realtime_data():
    """Streaming data in real-time (1-second intervals)"""
    return Response(stream_data(), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
