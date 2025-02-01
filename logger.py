import logging
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Generate a single timestamp for the entire script run
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_file = f'logs/{timestamp}_logs.log'

# Configure logger only once
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Formatter to include time, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

# Function to log informational messages
def log_info(message):
    logger.info(message)

# Function to log error messages
def log_error(message):
    logger.error(message)
