import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import matplotlib.pyplot as plt
from training_utils import simulate_and_store_data, train_and_evaluate_multi_step, preprocess_multi_step_data, visualize_multi_step_results
from logger import log_error, log_info


# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['byner_analyzer']
historical_collection = db['historical_data']
predictions_collection = db['tr_predictions']
metrics_collection = db['tr_metrics']

def main():
    log_info("-----Model Trainig pipeline started-----")
    if historical_collection.count_documents({}) > 0:
        log_info("Training db already exists")
    else:
        # Generate 10,000 ticks synthetic data
        simulate_and_store_data(10000)
        log_info("Training db created")


    try:
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_scaler = preprocess_multi_step_data()
        joblib.dump(feature_scaler, "scalers/feature_scaler.pkl")
        log_info("Historical data preprocessed for LSTM training")

        # Train and evaluate
        model, history = train_and_evaluate_multi_step(X_train, X_test, y_train, y_test, feature_scaler)
        log_info("LSTM training completed")
    
        # Create 'models' folder if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')

        # Save the model to the 'models' folder
        model.save('models/lstm_model.h5')
        log_info("Model saved to models/lstm_model.h5")

    except Exception as e:
        log_error(e)


    y_pred = model.predict(X_test)

    # Visualize results
    visualize_multi_step_results(y_test[:5], y_pred[:5])
    log_info("-----Model Trainig pipeline completed-----")


if __name__ == "__main__":
    main()