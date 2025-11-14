"""Action detection and counting for dumbbell exercises.

This module implements automatic detection and counting of elbow flexion
and wrist extension actions, with classification of movement amplitude
(full, half, invalid).
"""

import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Union, Optional
from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.features.activation import ActivationFeatures


class ActionDetector:
    """
    Automatic action detection and counting for repetitive exercises.
    
    Detects individual repetitions and classifies them by movement amplitude.
    """
    
    def __init__(
        self,
        sampling_rate: float,
        threshold_factor: float = 2.0,
        min_duration: float = 0.3,
        max_duration: float = 3.0
    ):
        """
        Initialize action detector.
        
        Args:
            sampling_rate: Sampling frequency in Hz
            threshold_factor: Multiplier for detection threshold (relative to baseline)
            min_duration: Minimum action duration in seconds
            max_duration: Maximum action duration in seconds
        """
        self.sampling_rate = sampling_rate
        self.threshold_factor = threshold_factor
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.activation_features = ActivationFeatures(sampling_rate)
    
    def detect_actions(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        channel_idx: int = 0
    ) -> List[Dict]:
        """
        Detect individual actions (repetitions) in EMG signal.
        
        Uses envelope detection and peak finding to identify actions.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            channel_idx: Channel index to use for detection
            
        Returns:
            List of detected actions with metadata
        """
        if isinstance(emg_signal, EMGSignal):
            data = emg_signal.get_channel(channel_idx)
            fs = emg_signal.sampling_rate
        else:
            data = emg_signal if emg_signal.ndim == 1 else emg_signal[:, channel_idx]
            fs = self.sampling_rate
        
        # Compute signal envelope (RMS with sliding window)
        window_size = int(0.05 * fs)  # 50ms window
        envelope = self._compute_envelope(data, window_size)
        
        # Estimate baseline and threshold
        baseline = np.percentile(envelope, 25)
        threshold = baseline * self.threshold_factor
        
        # Find peaks (action onsets)
        min_distance = int(self.min_duration * fs)
        peaks, properties = find_peaks(
            envelope,
            height=threshold,
            distance=min_distance
        )
        
        # Extract actions
        actions = []
        for i, peak_idx in enumerate(peaks):
            # Find action boundaries (start and end)
            start_idx, end_idx = self._find_action_boundaries(
                envelope, peak_idx, threshold * 0.5
            )
            
            # Check duration
            duration = (end_idx - start_idx) / fs
            if duration < self.min_duration or duration > self.max_duration:
                continue
            
            # Extract action segment
            action_data = data[start_idx:end_idx]
            
            # Compute action features
            action_rms = np.sqrt(np.mean(action_data ** 2))
            action_peak = np.max(np.abs(action_data))
            
            actions.append({
                'action_id': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'peak_idx': peak_idx,
                'start_time': start_idx / fs,
                'end_time': end_idx / fs,
                'duration': duration,
                'rms': action_rms,
                'peak_amplitude': action_peak,
                'peak_height': envelope[peak_idx]
            })
        
        return actions
    
    def classify_amplitude(
        self,
        actions: List[Dict],
        full_threshold: float = 0.8,
        half_threshold: float = 0.4
    ) -> List[Dict]:
        """
        Classify actions by movement amplitude.
        
        Classification based on relative activation intensity:
        - Full: High activation (>80% of max)
        - Half: Medium activation (40-80% of max)
        - Invalid: Low activation (<40% of max)
        
        Args:
            actions: List of detected actions
            full_threshold: Threshold for full movement (fraction of max)
            half_threshold: Threshold for half movement (fraction of max)
            
        Returns:
            Actions with amplitude classification added
        """
        if not actions:
            return actions
        
        # Find maximum RMS across all actions
        max_rms = max(action['rms'] for action in actions)
        
        # Classify each action
        for action in actions:
            relative_rms = action['rms'] / max_rms
            
            if relative_rms >= full_threshold:
                amplitude_class = 'full'
                quality_score = 1.0
            elif relative_rms >= half_threshold:
                amplitude_class = 'half'
                quality_score = 0.5
            else:
                amplitude_class = 'invalid'
                quality_score = 0.0
            
            action['amplitude_class'] = amplitude_class
            action['relative_intensity'] = relative_rms
            action['quality_score'] = quality_score
        
        return actions
    
    def count_actions(
        self,
        actions: List[Dict],
        min_quality: float = 0.0
    ) -> Dict[str, int]:
        """
        Count actions by amplitude classification.
        
        Args:
            actions: List of detected actions with classification
            min_quality: Minimum quality score to count
            
        Returns:
            Dictionary with action counts
        """
        counts = {
            'total': len(actions),
            'full': 0,
            'half': 0,
            'invalid': 0,
            'valid': 0  # full + half
        }
        
        for action in actions:
            if action.get('quality_score', 0) >= min_quality:
                amp_class = action.get('amplitude_class', 'invalid')
                counts[amp_class] += 1
                if amp_class in ['full', 'half']:
                    counts['valid'] += 1
        
        return counts
    
    def _compute_envelope(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute RMS envelope of signal."""
        squared = data ** 2
        envelope = np.sqrt(
            np.convolve(squared, np.ones(window_size) / window_size, mode='same')
        )
        return envelope
    
    def _find_action_boundaries(
        self,
        envelope: np.ndarray,
        peak_idx: int,
        threshold: float
    ) -> Tuple[int, int]:
        """
        Find start and end indices of an action.
        
        Args:
            envelope: Signal envelope
            peak_idx: Peak index
            threshold: Threshold for boundary detection
            
        Returns:
            Tuple of (start_idx, end_idx)
        """
        # Find start (backward from peak)
        start_idx = peak_idx
        while start_idx > 0 and envelope[start_idx] > threshold:
            start_idx -= 1
        
        # Find end (forward from peak)
        end_idx = peak_idx
        while end_idx < len(envelope) - 1 and envelope[end_idx] > threshold:
            end_idx += 1
        
        return start_idx, end_idx
    
    def get_action_statistics(self, actions: List[Dict]) -> Dict[str, float]:
        """
        Compute statistics about detected actions.
        
        Args:
            actions: List of detected actions
            
        Returns:
            Dictionary of statistics
        """
        if not actions:
            return {
                'mean_duration': 0,
                'std_duration': 0,
                'mean_rms': 0,
                'std_rms': 0,
                'mean_quality': 0
            }
        
        durations = [a['duration'] for a in actions]
        rms_values = [a['rms'] for a in actions]
        quality_scores = [a.get('quality_score', 0) for a in actions]
        
        return {
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'mean_rms': np.mean(rms_values),
            'std_rms': np.std(rms_values),
            'mean_quality': np.mean(quality_scores),
            'quality_variation': np.std(quality_scores)
        }
