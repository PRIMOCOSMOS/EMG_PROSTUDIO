"""Core EMG signal representation and handling."""

import numpy as np
from typing import Optional, Dict, Any
import json


class EMGSignal:
    """
    Core class for representing and managing EMG signals.
    
    This class provides a unified interface for EMG data with metadata,
    supporting various operations and transformations.
    
    Attributes:
        data (np.ndarray): Raw EMG signal data
        sampling_rate (float): Sampling frequency in Hz
        channels (list): Channel names
        metadata (dict): Additional metadata
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: float,
        channels: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize EMG signal.
        
        Args:
            data: Signal data array (samples x channels) or (samples,) for single channel
            sampling_rate: Sampling frequency in Hz
            channels: List of channel names
            metadata: Additional metadata dictionary
        """
        self.data = np.atleast_2d(data).T if data.ndim == 1 else data
        self.sampling_rate = sampling_rate
        self.channels = channels or [f"Channel_{i}" for i in range(self.data.shape[1])]
        self.metadata = metadata or {}
        
        # Validate data
        if self.data.shape[1] != len(self.channels):
            raise ValueError(
                f"Number of channels ({len(self.channels)}) doesn't match "
                f"data shape ({self.data.shape[1]})"
            )
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the signal."""
        return self.data.shape[0]
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.data.shape[1]
    
    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return self.n_samples / self.sampling_rate
    
    @property
    def time_vector(self) -> np.ndarray:
        """Time vector for the signal."""
        return np.arange(self.n_samples) / self.sampling_rate
    
    def get_channel(self, channel_idx: int) -> np.ndarray:
        """
        Get data for a specific channel.
        
        Args:
            channel_idx: Channel index
            
        Returns:
            Channel data as 1D array
        """
        return self.data[:, channel_idx]
    
    def get_segment(self, start_time: float, end_time: float) -> 'EMGSignal':
        """
        Extract a temporal segment of the signal.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            New EMGSignal object with the segment
        """
        start_idx = int(start_time * self.sampling_rate)
        end_idx = int(end_time * self.sampling_rate)
        
        segment_data = self.data[start_idx:end_idx, :]
        new_metadata = self.metadata.copy()
        new_metadata['segment'] = {'start': start_time, 'end': end_time}
        
        return EMGSignal(
            data=segment_data,
            sampling_rate=self.sampling_rate,
            channels=self.channels.copy(),
            metadata=new_metadata
        )
    
    def copy(self) -> 'EMGSignal':
        """Create a deep copy of the signal."""
        return EMGSignal(
            data=self.data.copy(),
            sampling_rate=self.sampling_rate,
            channels=self.channels.copy(),
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization."""
        return {
            'data': self.data.tolist(),
            'sampling_rate': self.sampling_rate,
            'channels': self.channels,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'EMGSignal':
        """Create EMGSignal from dictionary."""
        return cls(
            data=np.array(data_dict['data']),
            sampling_rate=data_dict['sampling_rate'],
            channels=data_dict.get('channels'),
            metadata=data_dict.get('metadata')
        )
    
    def __repr__(self) -> str:
        return (
            f"EMGSignal(n_samples={self.n_samples}, "
            f"n_channels={self.n_channels}, "
            f"sampling_rate={self.sampling_rate} Hz, "
            f"duration={self.duration:.2f} s)"
        )
