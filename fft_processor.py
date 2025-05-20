import numpy as np
from scipy.fft import fft, fftfreq
from typing import Tuple, Dict

class FFTProcessor:
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize the FFT processor with sampling rate.
        
        The FFT processor converts time-domain accelerometer signals into frequency domain
        to analyze the frequency components of different activities.
        
        Args:
            sampling_rate (int): Sampling rate in Hz (default: 100Hz)
        """
        self.sampling_rate = sampling_rate
        
    def compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of a signal and return frequency spectrum.
        
        This method:
        1. Computes the FFT of the input signal
        2. Calculates the corresponding frequencies
        3. Returns only positive frequencies and their amplitudes
        
        Args:
            signal (np.ndarray): Input time-domain signal (accelerometer data)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Frequencies array (in Hz)
                - Corresponding amplitude spectrum
        """
        # Get number of samples
        n = len(signal)
        
        # Compute FFT using scipy's fft function
        # This converts time-domain signal to frequency domain
        fft_result = fft(signal)
        
        # Compute corresponding frequencies using fftfreq
        # fftfreq returns the frequencies in cycles per unit of the sample spacing
        freqs = fftfreq(n, 1/self.sampling_rate)
        
        # Get only positive frequencies (first half of the FFT result)
        # The FFT is symmetric, so we only need the first half
        positive_freqs = freqs[:n//2]
        amplitudes = np.abs(fft_result[:n//2])
        
        return positive_freqs, amplitudes
    
    def get_dominant_frequencies(self, freqs: np.ndarray, amplitudes: np.ndarray, 
                               n_peaks: int = 3) -> Dict[float, float]:
        """
        Find dominant frequencies in the spectrum.
        
        This method identifies the most significant frequency components
        in the signal, which are often characteristic of specific activities.
        
        Args:
            freqs (np.ndarray): Frequency array from FFT
            amplitudes (np.ndarray): Amplitude array from FFT
            n_peaks (int): Number of dominant frequencies to find
            
        Returns:
            Dict[float, float]: Dictionary mapping frequencies to their amplitudes
        """
        # Find indices of the n_peaks highest amplitudes
        peak_indices = np.argsort(amplitudes)[-n_peaks:]
        
        # Get corresponding frequencies and amplitudes
        peak_freqs = freqs[peak_indices]
        peak_amps = amplitudes[peak_indices]
        
        return dict(zip(peak_freqs, peak_amps))
    
    def process_window(self, window_data: np.ndarray) -> Dict:
        """
        Process a window of accelerometer data and return FFT features.
        
        This method:
        1. Computes FFT for each axis (X, Y, Z)
        2. Finds dominant frequencies
        3. Calculates energy in different frequency bands
        4. Computes mean spectral amplitude
        
        Args:
            window_data (np.ndarray): Window of accelerometer data (n_samples x 3)
            
        Returns:
            Dict: Dictionary containing various FFT-based features
        """
        # Compute FFT for each axis
        x_freqs, x_amps = self.compute_fft(window_data[:, 0])
        y_freqs, y_amps = self.compute_fft(window_data[:, 1])
        z_freqs, z_amps = self.compute_fft(window_data[:, 2])
        
        # Get dominant frequencies for each axis
        # These are important for activity recognition
        x_dominant = self.get_dominant_frequencies(x_freqs, x_amps)
        y_dominant = self.get_dominant_frequencies(y_freqs, y_amps)
        z_dominant = self.get_dominant_frequencies(z_freqs, z_amps)
        
        # Helper function to calculate energy in a frequency band
        def get_band_energy(freqs, amplitudes, freq_range):
            # Create mask for frequencies in the specified range
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            # Calculate energy as sum of squared amplitudes
            return np.sum(amplitudes[mask]**2)
        
        # Define frequency bands for energy calculation
        # Different activities have different energy distributions in these bands
        low_band = (0, 1)   # Low frequency movements (e.g., slow walking)
        mid_band = (1, 3)   # Medium frequency movements (e.g., normal walking)
        high_band = (3, 5)  # High frequency movements (e.g., running)
        
        # Create feature dictionary
        features = {
            # Dominant frequencies and their amplitudes
            'x_dominant': x_dominant,
            'y_dominant': y_dominant,
            'z_dominant': z_dominant,
            
            # Energy in different frequency bands for each axis
            'x_low_energy': get_band_energy(x_freqs, x_amps, low_band),
            'x_mid_energy': get_band_energy(x_freqs, x_amps, mid_band),
            'x_high_energy': get_band_energy(x_freqs, x_amps, high_band),
            'y_low_energy': get_band_energy(y_freqs, y_amps, low_band),
            'y_mid_energy': get_band_energy(y_freqs, y_amps, mid_band),
            'y_high_energy': get_band_energy(y_freqs, y_amps, high_band),
            'z_low_energy': get_band_energy(z_freqs, z_amps, low_band),
            'z_mid_energy': get_band_energy(z_freqs, z_amps, mid_band),
            'z_high_energy': get_band_energy(z_freqs, z_amps, high_band),
            
            # Overall signal strength
            'mean_spectral_amplitude': np.mean([x_amps, y_amps, z_amps])
        }
        
        return features 