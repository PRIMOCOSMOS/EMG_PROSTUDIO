"""Muscle fatigue features.

Based on literature, these features are effective for fatigue monitoring:
- MDF (Median Frequency): Frequency that divides power spectrum in half
- MPF (Mean Power Frequency): Average frequency of power spectrum
- ZC (Zero Crossings): Number of times signal crosses zero
- MNF (Mean Frequency): First moment of power spectrum
- Spectral moments and ratios
"""

import numpy as np
from scipy import signal
from typing import Dict, Union, Optional
from emg_prostudio.core.signal import EMGSignal


class FatigueFeatures:
    """Extract muscle fatigue features from EMG signals."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize fatigue feature extractor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
    
    def median_frequency(self, data: np.ndarray) -> float:
        """
        Median Frequency (MDF).
        
        Frequency that divides the power spectrum into two equal parts.
        Decreases with muscle fatigue.
        
        Args:
            data: EMG signal data (1D array for single channel)
            
        Returns:
            MDF value in Hz
        """
        # Compute power spectral density
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=min(256, len(data)))
        
        # Calculate cumulative sum
        cumsum = np.cumsum(psd)
        total_power = cumsum[-1]
        
        # Find median frequency
        median_idx = np.where(cumsum >= total_power / 2)[0][0]
        return freqs[median_idx]
    
    def mean_power_frequency(self, data: np.ndarray) -> float:
        """
        Mean Power Frequency (MPF).
        
        Average frequency of the power spectrum weighted by power.
        Decreases with muscle fatigue.
        
        Args:
            data: EMG signal data
            
        Returns:
            MPF value in Hz
        """
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=min(256, len(data)))
        
        # Weighted average
        mpf = np.sum(freqs * psd) / np.sum(psd)
        return mpf
    
    def mean_frequency(self, data: np.ndarray) -> float:
        """
        Mean Frequency (MNF).
        
        First moment of power spectrum.
        
        Args:
            data: EMG signal data
            
        Returns:
            MNF value in Hz
        """
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=min(256, len(data)))
        mnf = np.sum(freqs * psd) / np.sum(psd)
        return mnf
    
    def zero_crossings(self, data: np.ndarray, threshold: float = 0.0) -> int:
        """
        Zero Crossings (ZC).
        
        Number of times the signal crosses zero.
        Related to dominant frequency content.
        
        Args:
            data: EMG signal data
            threshold: Threshold for crossing detection
            
        Returns:
            Number of zero crossings
        """
        # Find zero crossings
        crossings = np.where(np.diff(np.sign(data - threshold)))[0]
        return len(crossings)
    
    def slope_sign_changes(self, data: np.ndarray, threshold: float = 0.0) -> int:
        """
        Slope Sign Changes (SSC).
        
        Number of times the slope changes sign.
        
        Args:
            data: EMG signal data
            threshold: Threshold for change detection
            
        Returns:
            Number of slope sign changes
        """
        # Compute differences
        diff = np.diff(data)
        
        # Find sign changes
        changes = 0
        for i in range(len(diff) - 1):
            if abs(diff[i]) > threshold and abs(diff[i+1]) > threshold:
                if diff[i] * diff[i+1] < 0:
                    changes += 1
        
        return changes
    
    def spectral_moments(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate spectral moments (M0, M1, M2).
        
        Args:
            data: EMG signal data
            
        Returns:
            Dictionary with spectral moments
        """
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=min(256, len(data)))
        
        M0 = np.sum(psd)  # Total power
        M1 = np.sum(freqs * psd)  # First moment
        M2 = np.sum((freqs ** 2) * psd)  # Second moment
        
        return {
            'M0': M0,
            'M1': M1,
            'M2': M2,
            'MNF': M1 / M0 if M0 > 0 else 0,  # Mean frequency
            'frequency_variance': (M2 / M0 - (M1 / M0) ** 2) if M0 > 0 else 0
        }
    
    def extract_all(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        window_size: Optional[float] = None,
        overlap: float = 0.5
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract all fatigue features.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            window_size: Window size in seconds (None for whole signal)
            overlap: Window overlap ratio
            
        Returns:
            Dictionary of fatigue features
        """
        if isinstance(emg_signal, EMGSignal):
            data = emg_signal.data
        else:
            data = np.atleast_2d(emg_signal).T if emg_signal.ndim == 1 else emg_signal
        
        features = {}
        
        # Extract features per channel
        for ch in range(data.shape[1]):
            ch_data = data[:, ch]
            ch_prefix = f'ch{ch}_' if data.shape[1] > 1 else ''
            
            features[f'{ch_prefix}mdf'] = self.median_frequency(ch_data)
            features[f'{ch_prefix}mpf'] = self.mean_power_frequency(ch_data)
            features[f'{ch_prefix}mnf'] = self.mean_frequency(ch_data)
            features[f'{ch_prefix}zc'] = self.zero_crossings(ch_data)
            features[f'{ch_prefix}ssc'] = self.slope_sign_changes(ch_data)
            
            # Spectral moments
            moments = self.spectral_moments(ch_data)
            for key, value in moments.items():
                features[f'{ch_prefix}{key.lower()}'] = value
        
        return features
