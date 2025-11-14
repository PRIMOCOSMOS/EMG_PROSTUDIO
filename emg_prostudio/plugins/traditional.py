"""Traditional signal processing plugin."""

from typing import Dict, Any
import numpy as np
from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.plugins import AnalysisPlugin
from emg_prostudio.preprocessing.filters import EMGPreprocessor
from emg_prostudio.features.extractor import FeatureExtractor
from emg_prostudio.analysis.action_detector import ActionDetector
from emg_prostudio.analysis.fatigue_monitor import FatigueMonitor


class TraditionalSignalProcessing(AnalysisPlugin):
    """
    Traditional signal processing analysis method.
    
    Uses classical EMG analysis techniques:
    - Bandpass filtering
    - Time and frequency domain features
    - Statistical thresholding
    """
    
    def __init__(self):
        """Initialize traditional processing plugin."""
        super().__init__("TraditionalSignalProcessing", "1.0.0")
        self.preprocessor = None
        self.feature_extractor = None
        self.action_detector = None
        self.fatigue_monitor = None
    
    def analyze(
        self,
        emg_signal: EMGSignal,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform traditional signal processing analysis.
        
        Args:
            emg_signal: Input EMG signal
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive analysis results
        """
        # Initialize components
        fs = emg_signal.sampling_rate
        self.preprocessor = EMGPreprocessor(fs)
        self.feature_extractor = FeatureExtractor(fs)
        self.action_detector = ActionDetector(fs)
        self.fatigue_monitor = FatigueMonitor(fs)
        
        # Preprocessing
        preprocessed = self.preprocessor.preprocess_pipeline(
            emg_signal,
            remove_powerline=True,
            bandpass=True,
            rectify=False,
            extract_envelope=False
        )
        
        # Feature extraction
        features = self.feature_extractor.extract_all(preprocessed)
        
        # Action detection
        actions = self.action_detector.detect_actions(preprocessed)
        actions = self.action_detector.classify_amplitude(actions)
        action_counts = self.action_detector.count_actions(actions)
        
        # Fatigue analysis
        fatigue_trajectory = self.fatigue_monitor.compute_fatigue_trajectory(
            preprocessed,
            window_size=1.0,
            overlap=0.5
        )
        fatigue_analysis = self.fatigue_monitor.analyze_fatigue_progression(
            fatigue_trajectory
        )
        
        return {
            'method': 'traditional',
            'preprocessed_signal': preprocessed,
            'features': features,
            'actions': actions,
            'action_counts': action_counts,
            'fatigue_trajectory': fatigue_trajectory,
            'fatigue_analysis': fatigue_analysis
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.name,
            'version': self.version,
            'description': 'Traditional signal processing methods for EMG analysis',
            'type': 'signal_processing',
            'capabilities': [
                'preprocessing',
                'feature_extraction',
                'action_detection',
                'fatigue_monitoring'
            ]
        }
