"""
Anomaly Detection Function Using Isolation Forest

This code defines a function `anomaly_dect` that continuously monitors incoming 
data points for anomalies using a pre-trained Isolation Forest model. The function 
logs any detected anomalies and can visualize the data if enabled.

### Overview
The `anomaly_dect` function performs the following tasks:
1. Sets up logging to capture events and anomalies in a file named 'anomaly.log'.
2. Continuously generates random data points and checks for anomalies using the 
   loaded Isolation Forest model.
3. Provides optional visualization of the data points and detected anomalies.
4. Pauses between iterations to allow for visualization updates.
5. Runs indefinitely until the model file is not found or an explicit interruption occurs.

### Key Features
1. **Logging**: All anomaly detections are logged with timestamps for later review.
2. **Data Generation**: Random data points are generated, with a certain probability of being abnormal (outliers).
3. **Model Loading**: The Isolation Forest model is loaded from a file, allowing for real-time anomaly detection without needing to retrain.
4. **Visualization**: If enabled, the function visualizes incoming data points and highlights detected anomalies in real-time.

### Functionality Steps
1. **Setup**: Initializes logging and prepares for visualization if enabled.
2. **Data Generation**: Generates either normal or abnormal data points based on a predefined probability.
3. **Model Loading**: Attempts to load the pre-trained Isolation Forest model from disk.
4. **Anomaly Detection**:
   - Predicts whether the generated data point is an anomaly or not.
   - Logs and visualizes any detected anomalies.
5. **Loop Control**: The function runs indefinitely until interrupted or if the model file is missing.

"""

# Importing necessary libraries
import os
import random
import time
from datetime import datetime
from joblib import load
import logging
import matplotlib.pyplot as plt
import numpy as np

# Importing constants from a settings module
from settings import DELAY, OUTLIERS_GENERATION_PROBABILITY, VISUALIZATION

# Configuring logging to write to a file named 'anomaly.log'
logging.basicConfig(filename='anomaly.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# List to store incoming real-time data
data_ls = []

def anomaly_dect():
    """
    Continuously monitors incoming data for anomalies using an Isolation Forest model.

    This function generates random data points, checks them against a pre-trained 
    Isolation Forest model, and logs any detected anomalies. It also visualizes 
    the incoming data if visualization is enabled.

    The function runs indefinitely until the model file is not found or interrupted.

    Returns:
        None
    """

    _id = 0  # Initialize an identifier for each incoming data point

    # Visualization setup
    if VISUALIZATION:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_facecolor("black")
        fig.show()
    
    while True:
        # Generating some abnormal observations based on a given probability in settings.py
        if random.random() <= OUTLIERS_GENERATION_PROBABILITY:
            X_test = np.random.uniform(low=-4, high=4, size=(1, 1))  # Generate an outlier
        else:
            X = 0.3 * np.random.randn(1, 1)  # Generate normal data points
            X_test = (X + np.random.choice(a=[2, -2], size=1, p=[0.5, 0.5]))  # Shift normal points

        X_test = np.round(X_test, 3).tolist()  # Round the generated point

        current_time = datetime.utcnow().isoformat()  # Get current UTC time

        # Creating a record for the incoming data
        record = {"id": _id, "data": X_test, "current_time": current_time}
        print(f"Incoming: {record}")

        # Loading the Isolation Forest model from the file if it exists
        try:
            model_path = os.path.abspath("isolation_forest.joblib")
            clf = load(model_path)  # Load the trained model
        except:
            logging.warning(f"Model file not found")
            print(f'Model file not available')
            break  # Exit loop if the model cannot be loaded

        data = record['data']        
        data_ls.append(data[0][0])  # Append new data to list for visualization
        prediction = clf.predict(data)  # Predict if the new point is an anomaly

        # Visualization update
        if VISUALIZATION:
            ax.plot(data_ls, color='b')  # Plot all incoming data points in blue
            fig.canvas.draw()  # Update figure canvas
            ax.set_xlim(left=0, right=_id + 2)  # Adjust x-axis limits
        
        # Checking if an anomaly is detected
        if prediction[0] == -1:  # Anomaly detected if prediction is -1
            score = clf.score_samples(data)  # Get anomaly score
            record["score"] = np.round(score, 3).tolist()  # Round score for clarity
            
            if VISUALIZATION:
                plt.scatter(_id, data_ls[_id], color='r', linewidth=2)  # Mark anomaly in red
            
            logging.info(f"Anomaly Detected : {record}")  # Log anomaly detection event
            print(f'Anomaly Detected : {record}')  # Print anomaly detection message
            
        _id += 1  # Increment identifier for next iteration
        
        plt.pause(DELAY)  # Pause for specified delay to allow visualization updates
    
    plt.show()  # Show final plot (if applicable)