# Byner Analyzer Sequence Predictor

## Overview

The **Byner Analyzer Sequence Predictor** is a machine learning pipeline designed to predict the next 5 numbers in a sequence based on historical data and technical indicators. The project simulates realistic market-like data, preprocesses the information, trains a multi-step LSTM model, and performs live inference. All historical data, predictions, and evaluation metrics are stored in a MongoDB database.

The pipeline includes the following key components:
- **Data Generation & Storage:** Synthetic data is generated with a sine wave and noise, along with calculated technical indicators (moving average, RSI, Bollinger Bands, MACD). Data is then stored in MongoDB.
- **Preprocessing & Model Training:** Data is normalized, windowed into sequences, and used to train an LSTM network for multi-step forecasting.
- **Inference & Live Prediction:** A Flask API streams live synthetic prices. The inference module processes incoming data, makes predictions using the trained model, and stores predictions and real-time metrics in MongoDB.
- **Visualization:** Actual versus predicted sequences are visualized for validation.

## File Structure

- **`generator.py`**  
  Implements a Flask API to stream live synthetic price data.
  
- **`training_utils.py`**  
  Contains helper functions for (training.py):
  - Generating synthetic historical data.
  - Calculating technical indicators.
  - Preprocessing data (normalization, sequence generation).
  - Building, training, and evaluating the LSTM model.
  - Storing predictions and metrics into MongoDB.
  
- **`training.py`**  
  Main training script that:
  - Checks/creates the training database.
  - Calls the data simulation and storage functions.
  - Preprocesses the data.
  - Trains and evaluates the LSTM model.
  - Saves the trained model and scaler objects.
  
- **`inference.py`**  
  Implements the inference client which:
  - Loads the trained model and scaler.
  - Consumes real-time data from the Flask API.
  - Processes incoming events to predict the next number in the sequence.
  - Stores predictions and tracks real-time performance metrics in MongoDB.

- **`logger.py`**  
  Provides simple logging functions (`log_info`, `log_error`) for tracking the pipeline's progress.

- **`requirements.txt`**  
  Lists the Python dependencies needed to run the project.

## MongoDB Schema

The MongoDB database `byner_analyzer` contains several collections:

- **`historical_data`**  
  Stores each synthetic data tick with the following structure:
  ```json
  {
    "timestamp": "<datetime>",
    "price": <int>,
    "indicators": {
      "ma": <float>,
      "rsi": <float>,
      "upper_bb": <float>,
      "lower_bb": <float>,
      "macd": <float>,
      "signal": <float>
    }
  }
  ```

- **`predictions`** (`tr_predictions` / `inf_predictions`)  
  Records model predictions along with the corresponding input sequence:
  ```json
  {
    "timestamp": "<datetime>",
    "input_sequence": [<list of feature values>],
    "true_next_sequence": [<list of true values>],   // for training
    "predicted_next_sequence": [<list of predicted values>],  // for training
    "forecast": [<list of forecasted values>],        // for inference
    "model_version": "<string>"
  }
  ```

- **`metrics`** (or `tr_metrics` / `inf_metrics` depending on the context)  
  Stores evaluation metrics such as MSE, MAE, and real-time accuracy:
  ```json
  {
    "timestamp": "<datetime>",
    "metrics": {
      "mse": [<list of mse values>],
      "mae": [<list of mae values>],
      "accuracy": [<list of accuracy values>]
    },
    "forecast_horizon": <int>,
    "model_version": "<string>"
  }
  ```

- **`inf_historical_db`**  
  Used by the inference client to store the streamed historical data.

## Setup Instructions

### Prerequisites
- **Python 3.7+**
- **MongoDB** installed and running on `localhost:27017`  
  *For installation, visit: [MongoDB Installation Documentation](https://docs.mongodb.com/manual/installation/)*
- **Pipenv/virtualenv** (optional but recommended)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/kasun98/byner-analyzer.git
   cd byner-analyzer
   ```

2. **Create a Virtual Environment (optional):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Train the Model:**
   Execute the training script to generate synthetic data, train/test the LSTM model and save, and store everything in MongoDB.
   ```bash
   python training.py
   ```
2. **Start the Data Generator:**
   Ensure your `generator.py` (Flask API) is running to stream live data.
   ```bash
   python generator.py
   ```

3. **Run Inference:**
   Important! While running the generator.py, open a new terminal and run the inference.py. (Do not stop the generator.py)
   Start the inference client to process live data, (wait 60 seconds to load the first 60 prices into the buffer) generate predictions, and update the MongoDB collections with predictions and metrics.
   ```bash
   python inference.py
   ```

## Model Architecture and Reasoning

- **Model Choice:**  
  A multi-step LSTM network was chosen due to its strength in handling sequential data and time series forecasting.
  
- **Architecture Details:**
  - **LSTM Layers:** Two LSTM layers with 100 and 50 units respectively. The first layer returns sequences to allow the second LSTM to further process the time steps.
  - **Dropout Layers:** Used after each LSTM to reduce overfitting.
  - **Dense Layer:** Outputs the forecast horizon values (5).
  
- **Preprocessing:**  
  Data is normalized using `MinMaxScaler` and transformed into sliding window sequences to capture temporal dependencies.

## MongoDB Setup

1. **Installation:**  
   Follow MongoDB’s official guide to install and run a local instance.
   
2. **Connection String:**  
   The project uses `mongodb://localhost:27017/` by default. Adjust in the code if needed.
   
3. **Database Initialization:**  
   - The training script (`training.py`) automatically checks if data exists in the `historical_data` collection. If not, it generates and stores synthetic data.
   - Collections for predictions and metrics are also created automatically when new documents are inserted.

## Documentation & Further Improvements

- **Visualization:**  
  After training, the sample test results are visualized by comparing actual versus predicted sequences. The output image is saved to `train_output/multi_step_forecasts.png`.
  
- **Future Enhancements:**
  - Extend the model architecture.
  - Improve indicator calculations.
  - Enhance error handling and logging.

