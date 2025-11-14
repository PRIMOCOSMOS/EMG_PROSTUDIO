"""Feature extraction for EMG signal analysis.

This module provides comprehensive feature extraction for:
- Muscle activation level assessment
- Fatigue monitoring
- Movement quality evaluation
"""

import numpy as np
from typing import Dict, List, Optional, Union
from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.features.activation import ActivationFeatures
from emg_prostudio.features.fatigue import FatigueFeatures


class FeatureExtractor:
    """
    Main feature extraction interface.
    
    Provides unified access to activation and fatigue feature extraction.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        self.activation_features = ActivationFeatures(sampling_rate)
        self.fatigue_features = FatigueFeatures(sampling_rate)
    
    def extract_all(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        window_size: Optional[float] = None,
        overlap: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Extract all features from EMG signal.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            window_size: Window size in seconds (None for whole signal)
            overlap: Window overlap ratio (0-1)
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        # Extract activation features
        activation_feats = self.activation_features.extract_all(
            emg_signal, window_size, overlap
        )
        features.update(activation_feats)
        
        # Extract fatigue features
        fatigue_feats = self.fatigue_features.extract_all(
            emg_signal, window_size, overlap
        )
        features.update(fatigue_feats)
        
        return features
    
    def extract_windowed(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        window_size: float = 0.25,
        overlap: float = 0.5,
        features: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract features using sliding window approach.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            window_size: Window size in seconds
            overlap: Window overlap ratio (0-1)
            features: List of feature names to extract (None for all)
            
        Returns:
            Dictionary of feature time series
        """
        if isinstance(emg_signal, EMGSignal):
            data = emg_signal.data
        else:
            data = emg_signal
        
        window_samples = int(window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - overlap))
        
        n_windows = (data.shape[0] - window_samples) // step_samples + 1
        
        result = {}
        
        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            window_data = data[start_idx:end_idx, :]
            
            # Extract features for this window
            window_features = self.extract_all(window_data)
            
            # Store features
            for feat_name, feat_value in window_features.items():
                if features is None or feat_name in features:
                    if feat_name not in result:
                        result[feat_name] = []
                    result[feat_name].append(feat_value)
        
        # Convert lists to arrays
        for feat_name in result:
            result[feat_name] = np.array(result[feat_name])
        
        return result
