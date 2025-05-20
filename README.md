# Human Activity Recognition using FFT Analysis

This application classifies human activities using motion data collected from the Sensor Logger app on iOS. It uses Fast Fourier Transform (FFT) to analyze accelerometer signals and extract features for activity classification.

## Features

- CSV file upload for motion data analysis
- Time-series visualization of accelerometer signals
- FFT-based frequency spectrum analysis
- Activity classification (walking, running, climbing stairs)
- Future provision for real-time data streaming

## Technical Details

### FFT Analysis
The application uses Fast Fourier Transform to convert time-domain accelerometer signals into the frequency domain. This transformation helps identify dominant frequencies associated with different activities.

### Feature Extraction
For each 2-second window of data (with 50% overlap), we extract:
- Dominant frequency components
- Signal energy in different frequency bands
- Mean spectral amplitude
- Peak-to-peak amplitude

### Classification
The system uses a pre-trained Random Forest classifier to predict activities based on the extracted FFT features. The model is trained on labeled motion data from various activities.

## Project Structure

- `main.py`: Application entry point
- `data_loader.py`: CSV parsing and data formatting
- `fft_processor.py`: FFT computation and frequency analysis
- `feature_extraction.py`: Feature extraction from FFT results
- `classifier.py`: Activity classification using pre-trained model
- `visualizer.py`: Data visualization using matplotlib
- `realtime_stub.py`: Placeholder for future real-time streaming

## Future Development
- Real-time data streaming from mobile devices
- Additional activity types
- Improved feature extraction
- Real-time visualization

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage
1. Export motion data from Sensor Logger app as CSV
2. Upload the CSV file through the application interface
3. View the time-series plots and FFT analysis
4. See the predicted activity classification 