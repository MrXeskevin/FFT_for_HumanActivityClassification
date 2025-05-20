import pandas as pd
import numpy as np
from typing import Tuple, List, Union
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, window_size: int = 2, overlap: float = 0.5):
        """
        Initialize the DataLoader with window parameters.
        
        Args:
            window_size (int): Size of the analysis window in seconds
            overlap (float): Overlap between consecutive windows (0 to 1)
        """
        self.window_size = window_size
        self.overlap = overlap
        
    def load_data(self, file_path: str, data_format: str = 'raw') -> Union[pd.DataFrame, np.ndarray]:
        """
        Load data from file, supporting multiple formats.
        
        Args:
            file_path (str): Path to the data file
            data_format (str): Format of the data ('raw' or 'uci')
            
        Returns:
            Union[pd.DataFrame, np.ndarray]: Loaded data
        """
        if data_format == 'raw':
            return self.load_csv(file_path)
        elif data_format == 'uci':
            return self.load_uci_har(file_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess accelerometer data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Preprocessed accelerometer data
        """
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV file must contain timestamp, acc_x, acc_y, and acc_z columns")
        
        # Convert accelerometer data to numeric
        for col in ['acc_x', 'acc_y', 'acc_z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Convert timestamp to datetime if it's in seconds
        if df['timestamp'].dtype in [np.float64, np.int64]:
            # Convert seconds to datetime starting from current time
            start_time = datetime.now()
            df['timestamp'] = df['timestamp'].apply(
                lambda x: start_time + timedelta(seconds=x)
            )
        else:
            # Try to convert to datetime if it's not already
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Keep only required columns
        df = df[required_columns]
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def load_uci_har(self, file_path: str) -> np.ndarray:
        """
        Load UCI HAR format data.
        
        Args:
            file_path (str): Path to the UCI HAR data file
            
        Returns:
            np.ndarray: Loaded data
        """
        # Load the fixed-width format data
        data = np.loadtxt(file_path)
        return data
    
    def segment_data(self, data: Union[pd.DataFrame, np.ndarray], 
                    sampling_rate: int = 100,
                    data_format: str = 'raw') -> List[Union[pd.DataFrame, np.ndarray]]:
        """
        Segment the data into overlapping windows.
        
        Args:
            data (Union[pd.DataFrame, np.ndarray]): Input data
            sampling_rate (int): Sampling rate in Hz
            data_format (str): Format of the data ('raw' or 'uci')
            
        Returns:
            List[Union[pd.DataFrame, np.ndarray]]: List of data windows
        """
        if data_format == 'raw':
            return self._segment_raw_data(data, sampling_rate)
        elif data_format == 'uci':
            return self._segment_uci_data(data)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _segment_raw_data(self, df: pd.DataFrame, sampling_rate: int) -> List[pd.DataFrame]:
        """
        Segment raw time-series data into windows.
        """
        window_samples = self.window_size * sampling_rate
        step_size = int(window_samples * (1 - self.overlap))
        
        if len(df) < window_samples:
            raise ValueError(f"Not enough data points. Need at least {window_samples} samples, got {len(df)}")
        
        windows = []
        for start_idx in range(0, len(df) - window_samples + 1, step_size):
            window = df.iloc[start_idx:start_idx + window_samples]
            windows.append(window)
            
        return windows
    
    def _segment_uci_data(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Segment UCI HAR format data into windows.
        Each row in UCI HAR data is already a window of features.
        """
        return [data[i:i+1] for i in range(len(data))]
    
    def get_sampling_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate the sampling rate from the timestamp data.
        
        Args:
            df (pd.DataFrame): Input accelerometer data
            
        Returns:
            float: Sampling rate in Hz
        """
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        sampling_rate = 1 / time_diffs.mean()
        return sampling_rate
    
    def get_window_data(self, window: pd.DataFrame) -> np.ndarray:
        """
        Extract accelerometer data from a window as a numpy array.
        
        Args:
            window (pd.DataFrame): Window of accelerometer data
            
        Returns:
            np.ndarray: Array of accelerometer data
        """
        return window[['acc_x', 'acc_y', 'acc_z']].values 