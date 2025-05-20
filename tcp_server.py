import socket
import threading
import numpy as np
import pandas as pd
from typing import List, Optional
from queue import Queue
import time
from data_loader import DataLoader
from fft_processor import FFTProcessor
from feature_extraction import FeatureExtractor
from classifier import ActivityClassifier
from visualizer import DataVisualizer

class TCPServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5005, 
                 window_size: int = 200, sampling_rate: int = 100):
        """
        Initialize TCP server for real-time accelerometer data.
        
        This server:
        1. Listens for incoming accelerometer data from Physics Toolbox Sensor Suite
        2. Buffers the data into windows
        3. Processes each window through the activity classification pipeline
        4. Sends results to the web interface
        
        Args:
            host: Host address to bind to (default: '0.0.0.0' for all interfaces)
            port: Port to listen on (default: 5005)
            window_size: Number of samples per window (default: 200 for 2 seconds at 100Hz)
            sampling_rate: Sampling rate in Hz (default: 100)
        """
        self.host = host
        self.port = port
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.buffer: List[str] = []  # Buffer for incoming data
        self.running = False
        self.socket = None
        self.thread = None
        self.data_queue = Queue()  # Queue for thread-safe data transfer
        
        # Initialize processing pipeline components
        self.data_loader = DataLoader()
        self.fft_processor = FFTProcessor(sampling_rate=sampling_rate)
        self.feature_extractor = FeatureExtractor()
        self.classifier = ActivityClassifier()
        self.visualizer = DataVisualizer()
        
    def start(self) -> None:
        """Start the TCP server in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True  # Thread will exit when main program exits
        self.thread.start()
        print(f"TCP server started on {self.host}:{self.port}")
        
    def stop(self) -> None:
        """Stop the TCP server and clean up resources."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join()
        print("TCP server stopped")
        
    def _run_server(self) -> None:
        """
        Run the TCP server in a background thread.
        
        This method:
        1. Creates and configures the server socket
        2. Listens for incoming connections
        3. Handles each connection in a loop
        4. Manages socket errors gracefully
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # Bind to the specified host and port
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)  # Allow one connection at a time
            print(f"Waiting for connection on {self.host}:{self.port}")
            
            while self.running:
                try:
                    # Accept incoming connection
                    conn, addr = self.socket.accept()
                    print(f"Connected to {addr}")
                    self._handle_connection(conn)
                except socket.error:
                    if self.running:
                        print("Socket error, retrying...")
                        time.sleep(1)
        finally:
            self.socket.close()
            
    def _handle_connection(self, conn: socket.socket) -> None:
        """
        Handle incoming data from a client connection.
        
        This method:
        1. Receives data in chunks
        2. Splits data into lines
        3. Processes each line of accelerometer data
        4. Handles disconnections and errors
        """
        try:
            while self.running:
                # Receive data in chunks of 1024 bytes
                data = conn.recv(1024).decode('utf-8')
                if not data:  # Connection closed by client
                    break
                    
                # Split data into lines and process each line
                lines = data.strip().split('\n')
                for line in lines:
                    if line.strip():
                        self._process_line(line)
                        
        except Exception as e:
            print(f"Error handling connection: {e}")
        finally:
            conn.close()
            
    def _process_line(self, line: str) -> None:
        """
        Process a single line of accelerometer data.
        
        This method:
        1. Parses the timestamp and accelerometer values
        2. Adds the data to the buffer
        3. Processes a window when enough samples are collected
        
        Args:
            line: String containing timestamp,acc_x,acc_y,acc_z
        """
        try:
            # Parse the line into timestamp and accelerometer values
            timestamp, acc_x, acc_y, acc_z = map(float, line.strip().split(','))
            
            # Add to buffer
            self.buffer.append(line)
            
            # Process window if we have enough samples
            if len(self.buffer) >= self.window_size:
                self._process_window()
                
        except ValueError as e:
            print(f"Error parsing line: {e}")
            
    def _process_window(self) -> None:
        """
        Process a complete window of data.
        
        This method:
        1. Creates a DataFrame from the buffered data
        2. Processes the data through the classification pipeline
        3. Makes activity predictions
        4. Sends results to the web interface
        5. Clears processed data from buffer
        """
        try:
            # Create DataFrame from buffer
            data = []
            for line in self.buffer[:self.window_size]:
                timestamp, acc_x, acc_y, acc_z = map(float, line.strip().split(','))
                data.append({
                    'timestamp': timestamp,
                    'acc_x': acc_x,
                    'acc_y': acc_y,
                    'acc_z': acc_z
                })
            
            df = pd.DataFrame(data)
            
            # Put data in queue for web interface
            self.data_queue.put(data)
            
            # Process data through pipeline
            features = self._process_data(df)
            
            # Make prediction using the classifier
            prediction = self.classifier.predict(features.reshape(1, -1))[0]
            probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
            
            # Convert probabilities to dictionary
            prob_dict = {
                str(label): float(prob) 
                for label, prob in zip(self.classifier.get_activity_labels(), probabilities)
            }
            
            # Get the activity with highest probability
            max_prob_activity = max(prob_dict.items(), key=lambda x: x[1])[0]
            
            # Print results
            print(f"\nPredicted Activity: {max_prob_activity}")
            print("Probabilities:")
            for activity, prob in prob_dict.items():
                print(f"  {activity}: {prob:.2f}")
            
            # Clear processed data from buffer
            self.buffer = self.buffer[self.window_size:]
            
        except Exception as e:
            print(f"Error processing window: {e}")
            
    def _process_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Process data through the classification pipeline.
        
        This method:
        1. Computes FFT of the accelerometer data
        2. Extracts features from the FFT results
        
        Args:
            df: DataFrame containing accelerometer data
            
        Returns:
            np.ndarray: Extracted features for classification
        """
        # Compute FFT
        fft_data = self.fft_processor.compute_fft(df)
        
        # Extract features
        features = self.feature_extractor.extract_features(fft_data)
        
        return features 