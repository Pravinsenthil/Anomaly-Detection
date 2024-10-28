# Efficient Data Stream Anomaly Detection

## Project Description
The goal of this project is to develop a Python script capable of detecting anomalies in a continuous data stream. This stream simulates real-time sequences of floating-point numbers, which could represent various metrics such as financial transactions or system metrics. The focus is on identifying unusual patterns, such as exceptionally high values or deviations from the norm.

## Objectives
1. **Algorithm Selection**: Identify and implement a suitable algorithm for anomaly detection that can adapt to concept drift and seasonal variations.
2. **Data Stream Simulation**: Design a function to emulate a data stream, incorporating regular patterns, seasonal elements, and random noise.
3. **Anomaly Detection**: Develop a real-time mechanism to accurately flag anomalies as the data is streamed.
4. **Optimization**: Ensure the algorithm is optimized for both speed and efficiency.
5. **Visualization**: Create a straightforward real-time visualization tool to display both the data stream and any detected anomalies.

## Features
- **Anomaly Detection with Isolation Forest**: Utilizes the Isolation Forest algorithm from scikit-learn for effective anomaly detection.
- **Real-Time Data Simulation**: Generates synthetic data that mimics real-world scenarios.
- **Logging**: Logs detected anomalies for further analysis.
- **Visualization**: Visualizes incoming data points and highlights detected anomalies in real-time.


├── model_training.py        
# Script to train the Isolation Forest model
├── anomaly_detection.py      
# Script for real-time anomaly detection 
├── settings.py             
# Configuration file with constants 
├── requirements.txt         
# List of required Python libraries 
└── anomaly.log             
# Log file for recorded anomalies

## Requirements
- Python 3.x
- Libraries:
  - NumPy
  - scikit-learn
  - Matplotlib
  - Joblib

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
