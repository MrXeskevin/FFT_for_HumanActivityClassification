from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
from queue import Queue
import threading
import asyncio
from typing import List
import logging
import os
from classifier import ActivityClassifier  # Import the classifier directly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Sensor Data Receiver")

# Add CORS middleware to allow requests from Android devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data buffer and processing queue
data_buffer: List[str] = []
processing_queue = Queue()
window_size = 200  # 2 seconds at 100Hz

# Initialize classifier
model_path = "activity_classifier.pkl"
if not os.path.exists(model_path):
    logger.error("Trained model not found. Please run train_classifier.py first.")
    raise FileNotFoundError("Trained model not found")

classifier = ActivityClassifier(model_path=model_path)

def process_window(window_data: List[str]):
    """Process a complete window of data."""
    try:
        # Convert window data to DataFrame
        data = []
        for line in window_data:
            timestamp, acc_x, acc_y, acc_z = map(float, line.strip().split(','))
            data.append({
                'timestamp': timestamp,
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z
            })
        
        df = pd.DataFrame(data)
        
        # Process through the classification pipeline
        features = classifier.extract_features(df)
        prediction = classifier.predict(features.reshape(1, -1))[0]
        probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
        
        # Convert probabilities to dictionary
        prob_dict = {
            str(label): float(prob) 
            for label, prob in zip(classifier.get_activity_labels(), probabilities)
        }
        
        # Get the activity with highest probability
        max_prob_activity = max(prob_dict.items(), key=lambda x: x[1])[0]
        
        # Log the results
        logger.info(f"Predicted Activity: {max_prob_activity}")
        logger.info("Probabilities:")
        for activity, prob in prob_dict.items():
            logger.info(f"  {activity}: {prob:.2f}")
            
    except Exception as e:
        logger.error(f"Error processing window: {e}")

async def process_queue():
    """Background task to process data from the queue."""
    while True:
        try:
            # Get data from queue
            data = processing_queue.get(timeout=1)
            
            # Add to buffer
            data_buffer.extend(data)
            
            # Process complete windows
            while len(data_buffer) >= window_size:
                window = data_buffer[:window_size]
                process_window(window)
                data_buffer = data_buffer[window_size:]
                
        except queue.Empty:
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in queue processing: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Start background processing task on startup."""
    asyncio.create_task(process_queue())

@app.post("/api/receive-sensor-data")
async def receive_sensor_data(request: Request):
    """Receive sensor data from Android app."""
    try:
        # Get raw data from request body
        body = await request.body()
        data = body.decode('utf-8').strip()
        
        # Split into lines and filter empty ones
        lines = [line for line in data.split('\n') if line.strip()]
        
        # Add to processing queue
        processing_queue.put(lines)
        
        return {"status": "success", "message": f"Received {len(lines)} data points"}
        
    except Exception as e:
        logger.error(f"Error receiving sensor data: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Get local IP address
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Start the server
    logger.info(f"Starting HTTP server at http://{local_ip}:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 