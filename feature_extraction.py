import numpy as np
from typing import Dict, List, Union
from scipy import stats

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_features(self, data: Union[Dict, np.ndarray], data_format: str = 'raw') -> np.ndarray:
        """
        Extract features from data, supporting multiple formats.
        
        Args:
            data: Input data (FFT features or raw window)
            data_format: Format of the data ('raw' or 'uci')
            
        Returns:
            np.ndarray: Array of numerical features
        """
        if data_format == 'raw':
            return self._extract_raw_features(data)
        elif data_format == 'uci':
            return self._extract_uci_features(data)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _extract_raw_features(self, fft_features: Dict) -> np.ndarray:
        """
        Extract features from raw time-series data using FFT.
        """
        features = []
        
        # Extract dominant frequencies and their amplitudes
        for axis in ['x', 'y', 'z']:
            dominant = fft_features[f'{axis}_dominant']
            # Add top 3 dominant frequencies and their amplitudes
            for freq, amp in dominant.items():
                features.extend([freq, amp])
            # Pad with zeros if less than 3 dominant frequencies
            while len(features) % 6 != 0:
                features.extend([0.0, 0.0])
        
        # Add energy features
        for axis in ['x', 'y', 'z']:
            for band in ['low', 'mid', 'high']:
                features.append(fft_features[f'{axis}_{band}_energy'])
        
        # Add mean spectral amplitude
        features.append(fft_features['mean_spectral_amplitude'])
        
        return np.array(features)
    
    def _extract_uci_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract features from UCI HAR format data.
        Each row in UCI HAR data is already a window of features.
        """
        return window
    
    def get_feature_names(self, data_format: str = 'raw') -> List[str]:
        """
        Get names of all features in the same order as extract_features output.
        
        Args:
            data_format: Format of the data ('raw' or 'uci')
            
        Returns:
            List[str]: List of feature names
        """
        if data_format == 'raw':
            return self._get_raw_feature_names()
        elif data_format == 'uci':
            return self._get_uci_feature_names()
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _get_raw_feature_names(self) -> List[str]:
        """Get names of features for raw time-series data."""
        feature_names = []
        
        # Dominant frequency features
        for axis in ['x', 'y', 'z']:
            for i in range(3):
                feature_names.extend([
                    f'{axis}_dominant_freq_{i+1}',
                    f'{axis}_dominant_amp_{i+1}'
                ])
        
        # Energy features
        for axis in ['x', 'y', 'z']:
            for band in ['low', 'mid', 'high']:
                feature_names.append(f'{axis}_{band}_energy')
        
        # Mean spectral amplitude
        feature_names.append('mean_spectral_amplitude')
        
        return feature_names
    
    def _get_uci_feature_names(self) -> List[str]:
        """Get names of features for UCI HAR data."""
        # Load feature names from UCI HAR dataset
        try:
            with open('UCI HAR Dataset/features.txt', 'r') as f:
                feature_names = [line.strip() for line in f]
            return feature_names
        except FileNotFoundError:
            return [f'feature_{i}' for i in range(561)]  # Default names if file not found 