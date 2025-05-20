import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from typing import Dict, List, Optional
import pandas as pd

class DataVisualizer:
    def __init__(self):
        """Initialize the data visualizer with improved styling."""
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Consistent color scheme
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the dataframe contains required columns."""
        required_columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def plot_time_series(self, df: pd.DataFrame, title: str = "Accelerometer Data",
                        show_legend: bool = True) -> None:
        """
        Plot time series of accelerometer data with improved styling.
        
        Args:
            df: DataFrame containing timestamp and accelerometer data
            title: Plot title
            show_legend: Whether to show the legend
        """
        try:
            self._validate_dataframe(df)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot each axis with consistent colors
            for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
                ax.plot(df['timestamp'], df[axis], 
                       label=axis.replace('acc_', '').upper(),
                       color=self.colors[i],
                       alpha=0.7,
                       linewidth=1.5)
            
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
            
            if show_legend:
                ax.legend(loc='upper right', framealpha=0.9)
            
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error plotting time series: {str(e)}")
    
    def plot_fft_spectrum(self, freqs: np.ndarray, amplitudes: np.ndarray, 
                         axis: str, title: Optional[str] = None,
                         max_freq: Optional[float] = None) -> None:
        """
        Plot FFT frequency spectrum with improved visualization.
        
        Args:
            freqs: Frequency array
            amplitudes: Amplitude array
            axis: Axis label (X, Y, or Z)
            title: Optional plot title
            max_freq: Optional maximum frequency to display
        """
        try:
            if title is None:
                title = f"FFT Spectrum - {axis.upper()}-axis"
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot spectrum
            ax.plot(freqs, amplitudes, color=self.colors[0], linewidth=1.5)
            
            # Find and highlight dominant frequencies
            peak_indices = np.argsort(amplitudes)[-3:]
            for idx in peak_indices:
                ax.axvline(x=freqs[idx], color='r', linestyle='--', alpha=0.5)
                ax.annotate(f'{freqs[idx]:.2f} Hz',
                          xy=(freqs[idx], amplitudes[idx]),
                          xytext=(10, 10),
                          textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
            
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_xlabel('Frequency (Hz)', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            
            if max_freq is not None:
                ax.set_xlim(0, max_freq)
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error plotting FFT spectrum: {str(e)}")
    
    def plot_activity_probabilities(self, probabilities: Dict[str, float],
                                  threshold: float = 0.1) -> None:
        """
        Plot activity classification probabilities with improved visualization.
        
        Args:
            probabilities: Dictionary of activity probabilities
            threshold: Minimum probability to display
        """
        try:
            # Filter activities above threshold
            activities = [act for act, prob in probabilities.items() if prob >= threshold]
            probs = [prob for prob in probabilities.values() if prob >= threshold]
            
            if not activities:
                st.warning("No activities above the probability threshold")
                return
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Create bar plot with consistent colors
            bars = ax.bar(activities, probs, color=self.colors[0])
            
            ax.set_title('Activity Classification Probabilities', fontsize=14, pad=20)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_ylim(0, 1)
            
            # Add probability values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error plotting activity probabilities: {str(e)}")
    
    def plot_window_selection(self, df: pd.DataFrame, window_indices: List[int], 
                            window_size: int) -> None:
        """
        Visualize selected windows on the time series with improved styling.
        
        Args:
            df: DataFrame containing the time series data
            window_indices: List of window start indices
            window_size: Size of each window
        """
        try:
            self._validate_dataframe(df)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the full time series with lower opacity
            for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
                ax.plot(df['timestamp'], df[axis],
                       label=axis.replace('acc_', '').upper(),
                       color=self.colors[i],
                       alpha=0.3,
                       linewidth=1)
            
            # Highlight selected windows
            for i, start_idx in enumerate(window_indices):
                # Convert start_idx to integer if it's a float
                start_idx = int(start_idx)
                end_idx = start_idx + window_size
                
                # Ensure indices are within bounds
                start_idx = max(0, min(start_idx, len(df) - 1))
                end_idx = max(0, min(end_idx, len(df)))
                
                # Get window data
                window_data = df.iloc[start_idx:end_idx]
                
                # Plot window boundaries
                ax.axvspan(df['timestamp'].iloc[start_idx],
                          df['timestamp'].iloc[end_idx-1],
                          alpha=0.2, color=f'C{i%10}')
                
                # Plot window data with higher opacity
                for j, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
                    ax.plot(window_data['timestamp'], window_data[axis],
                           color=self.colors[j],
                           alpha=0.7,
                           linewidth=1.5)
            
            ax.set_title('Selected Analysis Windows', fontsize=14, pad=20)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error plotting window selection: {str(e)}") 