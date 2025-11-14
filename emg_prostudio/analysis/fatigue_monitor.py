"""Fatigue monitoring and analysis.

This module monitors muscle fatigue progression during exercises by:
- Tracking fatigue-related features over time
- Comparing objective indicators with subjective fatigue reports
- Identifying fatigue onset and exhaustion points
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple
from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.features.fatigue import FatigueFeatures
from emg_prostudio.features.activation import ActivationFeatures


class FatigueMonitor:
    """
    Monitor muscle fatigue progression during exercise.
    
    Tracks fatigue indicators and correlates with subjective experience.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize fatigue monitor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        self.fatigue_features = FatigueFeatures(sampling_rate)
        self.activation_features = ActivationFeatures(sampling_rate)
    
    def compute_fatigue_trajectory(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        window_size: float = 1.0,
        overlap: float = 0.5,
        channel_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Compute fatigue indicators over time using sliding window.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            window_size: Window size in seconds
            overlap: Window overlap ratio (0-1)
            channel_idx: Channel index to analyze
            
        Returns:
            Dictionary with time series of fatigue indicators
        """
        if isinstance(emg_signal, EMGSignal):
            data = emg_signal.get_channel(channel_idx)
            fs = emg_signal.sampling_rate
        else:
            data = emg_signal if emg_signal.ndim == 1 else emg_signal[:, channel_idx]
            fs = self.sampling_rate
        
        window_samples = int(window_size * fs)
        step_samples = int(window_samples * (1 - overlap))
        
        n_windows = (len(data) - window_samples) // step_samples + 1
        
        # Initialize arrays for features
        time_points = []
        mdf_values = []
        mpf_values = []
        rms_values = []
        mav_values = []
        
        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            window_data = data[start_idx:end_idx]
            
            # Time point (center of window)
            time_points.append((start_idx + end_idx) / 2 / fs)
            
            # Fatigue features (decrease with fatigue)
            mdf_values.append(self.fatigue_features.median_frequency(window_data))
            mpf_values.append(self.fatigue_features.mean_power_frequency(window_data))
            
            # Activation features (may increase to compensate)
            rms_values.append(self.activation_features.rms(window_data))
            mav_values.append(self.activation_features.mav(window_data))
        
        return {
            'time': np.array(time_points),
            'mdf': np.array(mdf_values),
            'mpf': np.array(mpf_values),
            'rms': np.array(rms_values),
            'mav': np.array(mav_values)
        }
    
    def detect_fatigue_onset(
        self,
        trajectory: Dict[str, np.ndarray],
        feature: str = 'mdf',
        sensitivity: float = 0.15
    ) -> Optional[float]:
        """
        Detect when fatigue begins based on feature decline.
        
        Args:
            trajectory: Fatigue trajectory from compute_fatigue_trajectory
            feature: Feature to use for detection ('mdf', 'mpf', 'rms')
            sensitivity: Sensitivity threshold (fraction of initial value)
            
        Returns:
            Time of fatigue onset in seconds, or None if not detected
        """
        if feature not in trajectory:
            return None
        
        time = trajectory['time']
        values = trajectory[feature]
        
        if len(values) < 3:
            return None
        
        # Use baseline (first 20% of data) as reference
        baseline_end = max(1, len(values) // 5)
        baseline = np.mean(values[:baseline_end])
        
        # For MDF/MPF, look for decline
        if feature in ['mdf', 'mpf']:
            threshold = baseline * (1 - sensitivity)
            # Find first point below threshold
            below_threshold = np.where(values < threshold)[0]
            if len(below_threshold) > 0:
                # Require sustained decline (at least 3 consecutive points)
                for i in range(len(below_threshold) - 2):
                    if (below_threshold[i+1] == below_threshold[i] + 1 and
                        below_threshold[i+2] == below_threshold[i] + 2):
                        return time[below_threshold[i]]
        
        # For RMS/MAV, look for increase (compensation)
        elif feature in ['rms', 'mav']:
            threshold = baseline * (1 + sensitivity)
            above_threshold = np.where(values > threshold)[0]
            if len(above_threshold) > 0:
                for i in range(len(above_threshold) - 2):
                    if (above_threshold[i+1] == above_threshold[i] + 1 and
                        above_threshold[i+2] == above_threshold[i] + 2):
                        return time[above_threshold[i]]
        
        return None
    
    def compute_fatigue_index(
        self,
        trajectory: Dict[str, np.ndarray],
        method: str = 'frequency_decline'
    ) -> np.ndarray:
        """
        Compute normalized fatigue index (0 = no fatigue, 1 = exhausted).
        
        Args:
            trajectory: Fatigue trajectory from compute_fatigue_trajectory
            method: Method for computing index
                    'frequency_decline': Based on MDF/MPF decline
                    'combined': Combination of multiple indicators
            
        Returns:
            Array of fatigue index values over time
        """
        if method == 'frequency_decline':
            # Use MDF decline as primary indicator
            mdf = trajectory['mdf']
            mdf_initial = np.mean(mdf[:max(1, len(mdf)//10)])
            mdf_normalized = mdf / mdf_initial
            # Invert so decline = increase in fatigue
            fatigue_index = 1 - mdf_normalized
            # Clip to [0, 1]
            fatigue_index = np.clip(fatigue_index, 0, 1)
            
        elif method == 'combined':
            # Combine MDF decline and RMS increase
            mdf = trajectory['mdf']
            rms = trajectory['rms']
            
            mdf_initial = np.mean(mdf[:max(1, len(mdf)//10)])
            rms_initial = np.mean(rms[:max(1, len(rms)//10)])
            
            # Normalize
            mdf_normalized = mdf / mdf_initial
            rms_normalized = rms / rms_initial
            
            # Combine (MDF decline + RMS increase)
            fatigue_index = 0.6 * (1 - mdf_normalized) + 0.4 * (rms_normalized - 1)
            fatigue_index = np.clip(fatigue_index, 0, 1)
        
        else:
            raise ValueError(f"Unknown fatigue index method: {method}")
        
        return fatigue_index
    
    def analyze_fatigue_progression(
        self,
        trajectory: Dict[str, np.ndarray],
        subjective_fatigue_time: Optional[float] = None,
        exhaustion_time: Optional[float] = None
    ) -> Dict:
        """
        Comprehensive fatigue analysis with subjective correlation.
        
        Args:
            trajectory: Fatigue trajectory
            subjective_fatigue_time: Time when subject first felt fatigued (seconds)
            exhaustion_time: Time when subject reached exhaustion (seconds)
            
        Returns:
            Dictionary with analysis results
        """
        time = trajectory['time']
        mdf = trajectory['mdf']
        mpf = trajectory['mpf']
        rms = trajectory['rms']
        
        # Compute trends (linear regression)
        mdf_slope, mdf_intercept, mdf_r, _, _ = stats.linregress(time, mdf)
        mpf_slope, mpf_intercept, mpf_r, _, _ = stats.linregress(time, mpf)
        rms_slope, rms_intercept, rms_r, _, _ = stats.linregress(time, rms)
        
        # Detect objective fatigue onset
        onset_time = self.detect_fatigue_onset(trajectory, 'mdf')
        
        # Compute fatigue index
        fatigue_index = self.compute_fatigue_index(trajectory, method='combined')
        
        analysis = {
            'duration': time[-1] - time[0],
            'mdf_trend': {
                'slope': mdf_slope,
                'r_squared': mdf_r ** 2,
                'decline_rate': abs(mdf_slope) / np.mean(mdf) * 100  # % per second
            },
            'mpf_trend': {
                'slope': mpf_slope,
                'r_squared': mpf_r ** 2,
                'decline_rate': abs(mpf_slope) / np.mean(mpf) * 100
            },
            'rms_trend': {
                'slope': rms_slope,
                'r_squared': rms_r ** 2,
                'increase_rate': rms_slope / np.mean(rms) * 100
            },
            'objective_onset_time': onset_time,
            'fatigue_index': fatigue_index,
            'max_fatigue_index': np.max(fatigue_index),
            'final_fatigue_index': fatigue_index[-1]
        }
        
        # Compare with subjective reports
        if subjective_fatigue_time is not None:
            analysis['subjective_fatigue_time'] = subjective_fatigue_time
            if onset_time is not None:
                analysis['onset_agreement'] = abs(onset_time - subjective_fatigue_time)
                analysis['onset_sensitivity'] = onset_time <= subjective_fatigue_time
            
            # Find fatigue index at subjective fatigue time
            closest_idx = np.argmin(np.abs(time - subjective_fatigue_time))
            analysis['fatigue_index_at_subjective'] = fatigue_index[closest_idx]
        
        if exhaustion_time is not None:
            analysis['exhaustion_time'] = exhaustion_time
            # Find fatigue index at exhaustion
            closest_idx = np.argmin(np.abs(time - exhaustion_time))
            analysis['fatigue_index_at_exhaustion'] = fatigue_index[closest_idx]
        
        return analysis
