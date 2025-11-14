"""Feature extraction module."""

from emg_prostudio.features.extractor import FeatureExtractor
from emg_prostudio.features.activation import ActivationFeatures
from emg_prostudio.features.fatigue import FatigueFeatures

__all__ = ['FeatureExtractor', 'ActivationFeatures', 'FatigueFeatures']
