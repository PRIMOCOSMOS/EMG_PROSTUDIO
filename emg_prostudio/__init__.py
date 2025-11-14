"""EMG_PROSTUDIO - Professional EMG Signal Analysis Studio.

A comprehensive application for analyzing electromyography (EMG) signals
during dumbbell exercises, with features for:
- Signal preprocessing and noise removal
- Automatic action recognition and counting
- Muscle activation and fatigue analysis
- Movement quality assessment
- Real-time visualization with futuristic UI

This package provides a modular, extensible framework supporting:
1. Traditional signal processing methods
2. Lightweight deep learning approaches
3. Integration with existing ML models
"""

__version__ = "0.1.0"
__author__ = "PRIMOCOSMOS"

from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.preprocessing.filters import EMGPreprocessor
from emg_prostudio.features.extractor import FeatureExtractor
from emg_prostudio.analysis.action_detector import ActionDetector
from emg_prostudio.analysis.fatigue_monitor import FatigueMonitor

__all__ = [
    "EMGSignal",
    "EMGPreprocessor",
    "FeatureExtractor",
    "ActionDetector",
    "FatigueMonitor",
]
