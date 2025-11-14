"""EMG signal preprocessing and filtering.

This module provides comprehensive preprocessing capabilities for EMG signals:
- Bandpass filtering for motion artifacts and baseline drift removal
- Notch filtering for power line interference (50/60 Hz)
- Signal rectification and envelope extraction
- Noise reduction and smoothing
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from typing import Optional, Union, Tuple
import warnings

from emg_prostudio.core.signal import EMGSignal


class EMGPreprocessor:
    """
    EMG signal preprocessing with various filtering methods.
    
    This class implements standard EMG preprocessing techniques based on
    literature recommendations for removing noise and artifacts.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize preprocessor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
    
    def bandpass_filter(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        lowcut: float = 20.0,
        highcut: float = 450.0,
        order: int = 4
    ) -> Union[EMGSignal, np.ndarray]:
        """
        Apply bandpass Butterworth filter.
        
        Recommended for EMG: 20-450 Hz to remove motion artifacts (< 20 Hz)
        and high-frequency noise (> 450 Hz).
        
        Args:
            emg_signal: Input EMG signal or numpy array
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered signal (same type as input)
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if high >= 1.0:
            warnings.warn(f"High cutoff {highcut} Hz exceeds Nyquist frequency. "
                         f"Setting to {nyquist * 0.99} Hz")
            high = 0.99
        
        b, a = butter(order, [low, high], btype='band')
        
        if isinstance(emg_signal, EMGSignal):
            filtered_data = filtfilt(b, a, emg_signal.data, axis=0)
            result = emg_signal.copy()
            result.data = filtered_data
            result.metadata['preprocessing'] = result.metadata.get('preprocessing', [])
            result.metadata['preprocessing'].append({
                'method': 'bandpass',
                'lowcut': lowcut,
                'highcut': highcut,
                'order': order
            })
            return result
        else:
            return filtfilt(b, a, emg_signal, axis=0)
    
    def notch_filter(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        freq: float = 50.0,
        quality_factor: float = 30.0
    ) -> Union[EMGSignal, np.ndarray]:
        """
        Apply notch filter to remove power line interference.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            freq: Frequency to remove (50 Hz or 60 Hz for power line)
            quality_factor: Quality factor (higher = narrower notch)
            
        Returns:
            Filtered signal (same type as input)
        """
        nyquist = 0.5 * self.sampling_rate
        w0 = freq / nyquist
        
        b, a = iirnotch(w0, quality_factor)
        
        if isinstance(emg_signal, EMGSignal):
            filtered_data = filtfilt(b, a, emg_signal.data, axis=0)
            result = emg_signal.copy()
            result.data = filtered_data
            result.metadata['preprocessing'] = result.metadata.get('preprocessing', [])
            result.metadata['preprocessing'].append({
                'method': 'notch',
                'frequency': freq,
                'quality_factor': quality_factor
            })
            return result
        else:
            return filtfilt(b, a, emg_signal, axis=0)
    
    def highpass_filter(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        cutoff: float = 20.0,
        order: int = 4
    ) -> Union[EMGSignal, np.ndarray]:
        """
        Apply highpass filter to remove baseline drift.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            cutoff: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered signal (same type as input)
        """
        nyquist = 0.5 * self.sampling_rate
        high = cutoff / nyquist
        
        b, a = butter(order, high, btype='high')
        
        if isinstance(emg_signal, EMGSignal):
            filtered_data = filtfilt(b, a, emg_signal.data, axis=0)
            result = emg_signal.copy()
            result.data = filtered_data
            result.metadata['preprocessing'] = result.metadata.get('preprocessing', [])
            result.metadata['preprocessing'].append({
                'method': 'highpass',
                'cutoff': cutoff,
                'order': order
            })
            return result
        else:
            return filtfilt(b, a, emg_signal, axis=0)
    
    def rectify(
        self,
        emg_signal: Union[EMGSignal, np.ndarray]
    ) -> Union[EMGSignal, np.ndarray]:
        """
        Full-wave rectification (absolute value).
        
        Args:
            emg_signal: Input EMG signal or numpy array
            
        Returns:
            Rectified signal (same type as input)
        """
        if isinstance(emg_signal, EMGSignal):
            result = emg_signal.copy()
            result.data = np.abs(result.data)
            result.metadata['preprocessing'] = result.metadata.get('preprocessing', [])
            result.metadata['preprocessing'].append({'method': 'rectification'})
            return result
        else:
            return np.abs(emg_signal)
    
    def envelope(
        self,
        emg_signal: Union[EMGSignal, np.ndarray],
        method: str = 'rms',
        window_size: Optional[int] = None
    ) -> Union[EMGSignal, np.ndarray]:
        """
        Extract signal envelope.
        
        Args:
            emg_signal: Input EMG signal or numpy array
            method: Envelope extraction method ('rms', 'moving_average', 'hilbert')
            window_size: Window size in samples (for RMS and moving average)
            
        Returns:
            Envelope signal (same type as input)
        """
        if window_size is None:
            # Default: 100ms window
            window_size = int(0.1 * self.sampling_rate)
        
        if isinstance(emg_signal, EMGSignal):
            data = emg_signal.data
        else:
            data = emg_signal
        
        if method == 'rms':
            # Root mean square envelope
            envelope_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                channel_data = data[:, i]
                envelope_data[:, i] = self._rms_envelope(channel_data, window_size)
        
        elif method == 'moving_average':
            # Moving average of rectified signal
            rectified = np.abs(data)
            envelope_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                envelope_data[:, i] = self._moving_average(rectified[:, i], window_size)
        
        elif method == 'hilbert':
            # Hilbert transform envelope
            envelope_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                analytic_signal = hilbert(data[:, i])
                envelope_data[:, i] = np.abs(analytic_signal)
        
        else:
            raise ValueError(f"Unknown envelope method: {method}")
        
        if isinstance(emg_signal, EMGSignal):
            result = emg_signal.copy()
            result.data = envelope_data
            result.metadata['preprocessing'] = result.metadata.get('preprocessing', [])
            result.metadata['preprocessing'].append({
                'method': 'envelope',
                'envelope_method': method,
                'window_size': window_size
            })
            return result
        else:
            return envelope_data
    
    def _rms_envelope(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute RMS envelope using sliding window."""
        squared = data ** 2
        rms = np.sqrt(np.convolve(squared, np.ones(window_size) / window_size, mode='same'))
        return rms
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute moving average."""
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')
    
    def preprocess_pipeline(
        self,
        emg_signal: EMGSignal,
        remove_powerline: bool = True,
        powerline_freq: float = 50.0,
        bandpass: bool = True,
        lowcut: float = 20.0,
        highcut: float = 450.0,
        rectify: bool = False,
        extract_envelope: bool = False,
        envelope_method: str = 'rms'
    ) -> EMGSignal:
        """
        Complete preprocessing pipeline.
        
        Args:
            emg_signal: Input EMG signal
            remove_powerline: Apply notch filter
            powerline_freq: Power line frequency (50 or 60 Hz)
            bandpass: Apply bandpass filter
            lowcut: Bandpass low cutoff
            highcut: Bandpass high cutoff
            rectify: Apply rectification
            extract_envelope: Extract signal envelope
            envelope_method: Envelope extraction method
            
        Returns:
            Preprocessed EMG signal
        """
        result = emg_signal.copy()
        
        # 1. Remove power line interference
        if remove_powerline:
            result = self.notch_filter(result, freq=powerline_freq)
        
        # 2. Bandpass filtering
        if bandpass:
            result = self.bandpass_filter(result, lowcut=lowcut, highcut=highcut)
        
        # 3. Rectification
        if rectify:
            result = self.rectify(result)
        
        # 4. Envelope extraction
        if extract_envelope:
            result = self.envelope(result, method=envelope_method)
        
        return result
