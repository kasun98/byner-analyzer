
import json
import joblib
import time
from datetime import datetime
import requests
from collections import deque
import numpy as np
from pymongo import MongoClient
import os
from logger import log_error, log_info

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError



class InferenceClient:
    def __init__(self, model, feature_scaler):
        self.model = model
        self.feature_scaler = feature_scaler
        self.buffer = deque(maxlen=60)
        self.current_forecast = None

        # MongoDB setup
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['byner_analyzer']
        self.predictions_col = self.db['inf_predictions']
        self.hist_db = self.db['inf_historical_db']
        self.metrics_col = self.db['inf_metrics']

        # Store pending predictions
        self.pending_predictions = deque()  # (timestamp, predicted_value, expected_seconds)

        # Metrics tracking
        self.correct_predictions = 0
        self.total_valid_predictions = 0

    def process_event(self, data_point):
        """Process incoming SSE data points"""
        # Add to rolling buffer
        self.buffer.append(data_point)
        log_info(f"Buffer length: {len(self.buffer)}")

        
        # Check if we have enough data for prediction
        if len(self.buffer) == 60:
            # Prepare features for model input
            features = self._extract_features()
            
            # Make prediction
            forecast = self.predict(features)
            
            # Store in database
            inp_list = self.store_prediction(features, forecast)
            first_elements = [int(round(sublist[0])) for sublist in inp_list]

            last_actual_value = first_elements[-1]  # Last number in sequence

            # Evaluate past predictions
            self.evaluate_predictions(last_actual_value)

            # Store new prediction
            forecast_list = forecast.tolist()
            if last_actual_value in forecast_list:
                index = forecast_list.index(last_actual_value) + 1  # Convert 0-based index to 1-based seconds
                if index <= 5:
                    self.pending_predictions.append((datetime.now(), last_actual_value, index))

            # Store streamed data in hist db
            hists = self._extract_features_hist()
            self.store_stream(hists)

            return forecast, first_elements
        return None

    def evaluate_predictions(self, actual_value):
        """Check if the actual value matches any pending prediction within its expected window."""
        current_time = datetime.now()
        remaining_predictions = deque()

        for timestamp, predicted_value, expected_seconds in self.pending_predictions:
            time_diff = (current_time - timestamp).total_seconds()
            if time_diff >= expected_seconds:  # Time has elapsed for this prediction
                if actual_value == predicted_value:
                    self.correct_predictions += 1
                self.total_valid_predictions += 1
            else:
                remaining_predictions.append((timestamp, predicted_value, expected_seconds))

        self.pending_predictions = remaining_predictions
        self.update_metrics()

    def update_metrics(self):
        """Calculate and store real-time accuracy."""
        if self.total_valid_predictions > 0:
            accuracy = self.correct_predictions / self.total_valid_predictions
        else:
            accuracy = 0
        
        self.metrics_col.insert_one({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'total_predictions': self.total_valid_predictions,
            'correct_predictions': self.correct_predictions
        })

        # print(f"Realtime accuracy: {accuracy}\n Total predictions: {self.total_valid_predictions}\n Correct predictions: {self.correct_predictions}\n")
        log_info(f"Realtime accuracy: {accuracy} -- Total predictions: {self.total_valid_predictions} -- Correct predictions: {self.correct_predictions}")


    def _extract_features(self):
        """Convert buffer data to model input format"""
        return np.array([[
            dp['price'],
            dp['indicators']['ma'],
            dp['indicators']['rsi'],
            dp['indicators']['upper_bb'],
            dp['indicators']['lower_bb'],
            dp['indicators']['macd'],
            dp['indicators']['signal']
        ] for dp in self.buffer])
    
    def _extract_features_hist(self):
        """Convert buffer data to model input format"""
        return np.array([[
            dp['timestamp'],
            dp['price'],
            dp['indicators']['ma'],
            dp['indicators']['rsi'],
            dp['indicators']['upper_bb'],
            dp['indicators']['lower_bb'],
            dp['indicators']['macd'],
            dp['indicators']['signal']
        ] for dp in self.buffer])

    def predict(self, features):
        """Make prediction with proper preprocessing"""
        # Normalize features
        scaled = self.feature_scaler.transform(features)
        
        # Reshape for LSTM (batch_size, timesteps, features)
        input_data = scaled.reshape(1, 60, 7)
        
        # Make prediction
        prediction = self.model.predict(input_data)
        
        # Inverse transform and convert to integers
        return prediction.round().astype(int)[0]

    def store_prediction(self, input_seq, forecast):
        """Store prediction in MongoDB"""
        doc = {
            'timestamp': datetime.now(),
            'input_sequence': input_seq.tolist(),
            'forecast': forecast.tolist(),
            'model_version': 'prod_v1'
        }
        self.predictions_col.insert_one(doc)
        return input_seq

    def store_stream(self, hist):
        doc = {
            'timestamp': hist[0][0],
            'price': int(hist[0][1]),
            'indicators': {
                'ma': float(hist[0][2]),
                'rsi': float(hist[0][3]),
                'upper_bb': float(hist[0][4]),
                'lower_bb': float(hist[0][5]),
                'macd': float(hist[0][6]),
                'signal': float(hist[0][7])
            }
        }
        self.hist_db.insert_one(doc)


def start_inference_client(model_path, feature_scaler_path):
    """Initialize and run inference client"""
    # Load model and scalers
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    feature_scaler = joblib.load(feature_scaler_path)
    
    client = InferenceClient(model, feature_scaler)
    
    # Connect to SSE stream
    with requests.get('http://localhost:5000/realtime_data', stream=True) as stream:
        for line in stream.iter_lines():
            if line.startswith(b'data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    result = client.process_event(data)
                    
                    if result is not None:
                        forecast, inp_list = result
                        forecast = forecast.tolist()
                        
                        last_digit = inp_list[-1]  # Get the last digit from data

                        if last_digit in forecast:
                            index = forecast.index(last_digit)
                            print(f"Sequence: {inp_list}\n{last_digit} may appear after {index+1} seconds\n\n")
                        else:
                            print(f"Sequence: {inp_list}\n{last_digit} may appear after 5++ seconds\n\n")

                        # print(f"New 5-step forecast: {forecast}")
                        
                except json.JSONDecodeError:
                    log_error("Error decoding JSON")
                except KeyError as e:
                    log_error(f"Missing key in data: {str(e)}")

if __name__ == "__main__":
    start_inference_client(
        model_path='models/lstm_model.h5',
        feature_scaler_path='scalers/feature_scaler.pkl'
    )
