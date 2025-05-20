import streamlit as st
import pandas as pd
import numpy as np
from data_loader import DataLoader
from fft_processor import FFTProcessor
from feature_extraction import FeatureExtractor
from classifier import ActivityClassifier
from visualizer import DataVisualizer
from tcp_server import TCPServer
import threading
import queue
import time
import os

# Set page config
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# Initialize components
data_loader = DataLoader()
fft_processor = FFTProcessor()
feature_extractor = FeatureExtractor()

# Load the trained model
model_path = "activity_classifier.pkl"
if not os.path.exists(model_path):
    st.error("Trained model not found. Please run train_classifier.py first.")
    st.stop()

classifier = ActivityClassifier(model_path=model_path)
visualizer = DataVisualizer()

# Create a queue for real-time data
realtime_queue = queue.Queue()

# Initialize TCP server
tcp_server = TCPServer(port=5005, window_size=200, sampling_rate=100)

def process_realtime_data():
    """Process real-time data and update the display."""
    while True:
        try:
            # Get data from queue
            data = realtime_queue.get(timeout=1)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Process data through pipeline
            features = tcp_server._process_data(df)
            
            # Make prediction
            prediction = classifier.predict(features.reshape(1, -1))[0]
            probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
            
            # Convert probabilities to dictionary
            prob_dict = {
                str(label): float(prob) 
                for label, prob in zip(classifier.get_activity_labels(), probabilities)
            }
            
            # Update the display
            st.session_state['latest_prediction'] = prediction
            st.session_state['latest_probabilities'] = prob_dict
            st.session_state['latest_data'] = df
            
        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"Error processing real-time data: {e}")

# Main title
st.title("Human Activity Recognition using FFT Analysis")
st.markdown("""
This application analyzes motion data to classify human activities.
Upload a data file or connect to a real-time sensor to get started.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["File Analysis", "Real-Time Analysis"]
)

if page == "File Analysis":
    # Data format selection
    data_format = st.sidebar.radio(
        "Select data format",
        ["Raw CSV", "UCI HAR"]
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        f"Upload {'CSV' if data_format == 'Raw CSV' else 'UCI HAR'} file",
        type=["csv", "txt"]
    )
    
    if uploaded_file is not None:
        try:
            # Load and process data
            if data_format == "Raw CSV":
                data = data_loader.load_data(uploaded_file, data_format='raw')
                
                # Check if data has the required columns
                required_columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
                if not all(col in data.columns for col in required_columns):
                    st.error("CSV file must contain timestamp, acc_x, acc_y, and acc_z columns")
                    st.stop()
                
                # Convert accelerometer data to float
                for col in ['acc_x', 'acc_y', 'acc_z']:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Remove any rows with NaN values
                data = data.dropna()
                
                if len(data) == 0:
                    st.error("No valid data found in the CSV file")
                    st.stop()
                
                sampling_rate = data_loader.get_sampling_rate(data)
                
                # Display time series
                st.subheader("1. Raw Accelerometer Data")
                visualizer.plot_time_series(data)
                
                # Window selection
                window_size = st.slider(
                    "Select window size (seconds)",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1
                )
                
                # Process windows
                windows = data_loader.segment_data(data, int(sampling_rate * window_size))
                
                st.subheader("2. Activity Classification")
                for i, window in enumerate(windows):
                    # Compute FFT features
                    fft_features = fft_processor.process_window(window[['acc_x', 'acc_y', 'acc_z']].values)
                    
                    # Extract features
                    features = feature_extractor.extract_features(fft_features, data_format='raw')
                    
                    # Make prediction
                    prediction = classifier.predict(features.reshape(1, -1))[0]
                    probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
                    
                    # Convert probabilities to dictionary
                    prob_dict = {
                        str(label): float(prob) 
                        for label, prob in zip(classifier.get_activity_labels(), probabilities)
                    }
                    
                    # Get the activity with highest probability
                    max_prob_activity = max(prob_dict.items(), key=lambda x: x[1])[0]
                    
                    # Display results
                    st.write(f"Window {i+1}:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Predicted Activity:", str(max_prob_activity))
                    with col2:
                        visualizer.plot_activity_probabilities(prob_dict)
                    
                    # Plot FFT spectrum
                    st.subheader(f"3. FFT Analysis - Window {i+1}")
                    for axis, axis_data in [('X', window['acc_x']), 
                                          ('Y', window['acc_y']), 
                                          ('Z', window['acc_z'])]:
                        freqs, amps = fft_processor.compute_fft(axis_data.values)
                        visualizer.plot_fft_spectrum(freqs, amps, axis)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

else:  # Real-Time Analysis page
    st.subheader("Real-Time Activity Analysis")
    
    # Initialize session state for real-time data
    if 'latest_prediction' not in st.session_state:
        st.session_state['latest_prediction'] = None
    if 'latest_probabilities' not in st.session_state:
        st.session_state['latest_probabilities'] = None
    if 'latest_data' not in st.session_state:
        st.session_state['latest_data'] = None
    
    # Start/Stop server controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Server"):
            try:
                # Start TCP server in a background thread
                tcp_server.start()
                st.success("Server started! Waiting for data...")
                
                # Start data processing thread
                processing_thread = threading.Thread(target=process_realtime_data)
                processing_thread.daemon = True
                processing_thread.start()
                
            except Exception as e:
                st.error(f"Error starting server: {e}")
    
    with col2:
        if st.button("Stop Server"):
            try:
                tcp_server.stop()
                st.success("Server stopped")
            except Exception as e:
                st.error(f"Error stopping server: {e}")
    
    # Display real-time data
    if st.session_state['latest_data'] is not None:
        st.subheader("Latest Data")
        
        # Display time series
        visualizer.plot_time_series(st.session_state['latest_data'])
        
        # Display prediction
        st.subheader("Activity Classification")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Predicted Activity:", str(st.session_state['latest_prediction']))
        with col2:
            visualizer.plot_activity_probabilities(st.session_state['latest_probabilities'])
        
        # Display FFT analysis
        st.subheader("FFT Analysis")
        for axis, axis_data in [('X', st.session_state['latest_data']['acc_x']), 
                              ('Y', st.session_state['latest_data']['acc_y']), 
                              ('Z', st.session_state['latest_data']['acc_z'])]:
            freqs, amps = fft_processor.compute_fft(axis_data.values)
            visualizer.plot_fft_spectrum(freqs, amps, axis)
    else:
        st.info("No data received yet. Start the server and connect your device.") 