"""Muscle activation level features.

Based on literature, these features are effective for assessing muscle activation:
- RMS (Root Mean Square): Overall muscle activation intensity
- MAV (Mean Absolute Value): Average muscle activity
- iEMG (Integrated EMG): Total muscle activity
- VAR (Variance): Signal variability
- WL (Waveform Length): Signal complexity
"""

import numpy as np
from typing import Dict, Union, Optional
from emg_prostudio.core.signal import EMGSignal


class ActivationFeatures:
    """Extract muscle activation level features from EMG signals."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize activation feature extractor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
    
    def rms(self, data: np.ndarray) -> float:
        """
        Root Mean Square (RMS).
        
        Reflects the overall muscle activation intensity.
        
        Args:
            data: EMG signal data (1D or 2D array)
            
        Returns:
            RMS value
        """
        return np.sqrt(np.mean(data ** 2, axis=0))
    
    def mav(self, data: np.ndarray) -> float:
        """
        Mean Absolute Value (MAV).
        
        Represents average muscle activity level.
        
        Args:
            data: EMG signal data
            
        Returns:
            MAV value
        """
        return np.mean(np.abs(data), axis=0)
    
    def iemg(self, data: np.ndarray) -> float:
        """
        Integrated EMG (iEMG).
        
        Sum of absolute values, indicates total muscle activity.
        
        Args:
            data: EMG signal data
            
        Returns:
            iEMG value
        """
        return np.sum(np.abs(data), axis=0)
    
    def variance(self, data: np.ndarray) -> float:
        """
        Variance (VAR).
        
        Measures signal variability and power.
        
        Args:
            data: EMG signal data
            
        Returns:
            Variance value
        """
        return np.var(data, axis=0)
    
    def waveform_length(self, data: np.ndarray) -> float:
        """
        Waveform Length (WL).
        
        Cumulative length of the waveform, indicates signal complexity.
        
        Args:
            data: EMG signal data
            
        Returns:
            WL value
        """
        return np.sum(np.abs(np.diff(data, axis=0)), axis=0)
    
    def ssi(self, data: np.ndarray) -> float:
        """
        Simple Square Integral (SSI).
        
        Sum of squared values, energy indicator.
        
        Args:
            data: EMG signal data
            
        Returns:
            SSI value
        """
        return np.sum(data ** 2, axis=0)
    
    def extract_all(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        window_size: Optional[float] = None,
        overlap: float = 0.5
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract all activation features.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            window_size: Window size in seconds (None for whole signal)
            overlap: Window overlap ratio
            
        Returns:
            Dictionary of activation features
        """
        if isinstance(emg_signal, EMGSignal):
            data = emg_signal.data
        else:
            data = emg_signal
        
        features = {
            'rms': self.rms(data),
            'mav': self.mav(data),
            'iemg': self.iemg(data),
            'variance': self.variance(data),
            'waveform_length': self.waveform_length(data),
            'ssi': self.ssi(data)
        }
        
        return features
