"""
Hybrid feature extraction combining traditional signal processing with deep learning embeddings.

This module implements the core feature extraction strategy:
1. Traditional EMG features (time-domain and frequency-domain)
2. Deep learning embeddings from pre-trained models
3. Feature fusion for downstream ML models
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.features.activation import ActivationFeatures
from emg_prostudio.features.fatigue import FatigueFeatures


class HybridFeatureExtractor:
    """
    Extract hybrid features combining traditional signal processing and deep learning.
    
    This is the "soul" of the analysis system - features are extracted for each
    identified action segment (rep) and used as input to downstream ML models.
    """
    
    def __init__(
        self,
        use_deep_features: bool = False,
        embedding_model: Optional[any] = None
    ):
        """
        Initialize hybrid feature extractor.
        
        Args:
            use_deep_features: Whether to extract deep learning embeddings
            embedding_model: Pre-trained model for embedding extraction
        """
        self.use_deep_features = use_deep_features
        self.embedding_model = embedding_model
        
        # Initialize traditional feature extractors
        self.activation_extractor = ActivationFeatures()
        self.fatigue_extractor = FatigueFeatures()
    
    def extract_traditional_features(
        self,
        signal: EMGSignal,
        window_size: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract traditional EMG features (time-domain and frequency-domain).
        
        Args:
            signal: EMG signal segment
            window_size: Window size in seconds for windowed features
            
        Returns:
            Dictionary of feature vectors
        """
        features = {}
        
        # Time-domain features (activation)
        features['rms'] = self.activation_extractor.root_mean_square(signal.data)
        features['mav'] = self.activation_extractor.mean_absolute_value(signal.data)
        features['iemg'] = self.activation_extractor.integrated_emg(signal.data)
        features['var'] = self.activation_extractor.variance(signal.data)
        features['wl'] = self.activation_extractor.waveform_length(signal.data)
        features['ssi'] = self.activation_extractor.simple_square_integral(signal.data)
        
        # Frequency-domain features (fatigue)
        features['mdf'] = self.fatigue_extractor.median_frequency(
            signal.data, signal.sampling_rate
        )
        features['mpf'] = self.fatigue_extractor.mean_power_frequency(
            signal.data, signal.sampling_rate
        )
        features['mnf'] = self.fatigue_extractor.mean_frequency(
            signal.data, signal.sampling_rate
        )
        features['zc'] = self.fatigue_extractor.zero_crossings(signal.data)
        features['ssc'] = self.fatigue_extractor.slope_sign_changes(signal.data)
        
        return features
    
    def extract_deep_features(
        self,
        signal: EMGSignal
    ) -> Optional[np.ndarray]:
        """
        Extract deep learning embeddings using pre-trained model.
        
        Uses linear probe technique: embeddings are extracted from a pre-trained
        EMG model and will be concatenated with traditional features.
        
        Args:
            signal: EMG signal segment
            
        Returns:
            Embedding vector or None if no model available
        """
        if not self.use_deep_features or self.embedding_model is None:
            return None
        
        # Extract embeddings using the pre-trained model
        # This is a placeholder - actual implementation depends on model architecture
        try:
            embeddings = self.embedding_model.extract_embeddings(signal.data)
            return embeddings
        except Exception as e:
            print(f"Warning: Failed to extract deep features: {e}")
            return None
    
    def extract_hybrid_features(
        self,
        signal: EMGSignal,
        flatten: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract hybrid features combining traditional and deep learning approaches.
        
        This is the core feature extraction method that creates the feature vector
        for downstream ML models (LightGBM/RandomForest).
        
        Args:
            signal: EMG signal segment (one rep)
            flatten: If True, return flattened feature vector; else return dict
            
        Returns:
            Feature vector (if flatten=True) or feature dictionary
        """
        # Extract traditional features
        trad_features = self.extract_traditional_features(signal)
        
        # Extract deep features if enabled
        deep_features = None
        if self.use_deep_features:
            deep_features = self.extract_deep_features(signal)
        
        if not flatten:
            result = {
                'traditional': trad_features,
                'deep': deep_features
            }
            return result
        
        # Flatten and concatenate features
        feature_vector = []
        
        # Add traditional features (flatten multi-channel features)
        for key, value in trad_features.items():
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    # Multi-channel: flatten
                    feature_vector.extend(value.flatten())
                else:
                    # Single value per channel
                    feature_vector.extend(value)
            else:
                feature_vector.append(value)
        
        # Add deep features if available
        if deep_features is not None:
            if isinstance(deep_features, np.ndarray):
                feature_vector.extend(deep_features.flatten())
        
        return np.array(feature_vector)
    
    def extract_rep_features(
        self,
        rep_segments: List[EMGSignal]
    ) -> np.ndarray:
        """
        Extract features for multiple rep segments.
        
        Args:
            rep_segments: List of EMG signal segments (one per rep)
            
        Returns:
            Feature matrix (n_reps x n_features)
        """
        features = []
        for rep in rep_segments:
            feat = self.extract_hybrid_features(rep, flatten=True)
            features.append(feat)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features in the order they appear in feature vector.
        
        Returns:
            List of feature names
        """
        names = []
        
        # Traditional feature names
        trad_features = [
            'rms', 'mav', 'iemg', 'var', 'wl', 'ssi',  # Time-domain
            'mdf', 'mpf', 'mnf', 'zc', 'ssc'  # Frequency-domain
        ]
        names.extend(trad_features)
        
        # Deep feature names (if enabled)
        if self.use_deep_features and self.embedding_model is not None:
            # Placeholder - actual names depend on model
            embedding_dim = getattr(self.embedding_model, 'embedding_dim', 128)
            names.extend([f'embedding_{i}' for i in range(embedding_dim)])
        
        return names
