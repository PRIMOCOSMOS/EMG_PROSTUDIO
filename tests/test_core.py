"""Tests for core EMG signal functionality."""

import pytest
import numpy as np
from emg_prostudio.core.signal import EMGSignal


def test_emg_signal_creation():
    """Test EMG signal object creation."""
    data = np.random.randn(1000, 2)
    signal = EMGSignal(data=data, sampling_rate=1000.0)
    
    assert signal.n_samples == 1000
    assert signal.n_channels == 2
    assert signal.sampling_rate == 1000.0
    assert signal.duration == 1.0


def test_emg_signal_1d_input():
    """Test creation with 1D input."""
    data = np.random.randn(1000)
    signal = EMGSignal(data=data, sampling_rate=500.0)
    
    assert signal.n_samples == 1000
    assert signal.n_channels == 1
    assert signal.data.shape == (1000, 1)


def test_emg_signal_channels():
    """Test channel handling."""
    data = np.random.randn(1000, 3)
    channels = ['Ch1', 'Ch2', 'Ch3']
    signal = EMGSignal(data=data, sampling_rate=1000.0, channels=channels)
    
    assert signal.channels == channels
    assert len(signal.channels) == signal.n_channels


def test_get_channel():
    """Test getting individual channel."""
    data = np.random.randn(1000, 2)
    signal = EMGSignal(data=data, sampling_rate=1000.0)
    
    ch0 = signal.get_channel(0)
    assert ch0.shape == (1000,)
    np.testing.assert_array_equal(ch0, data[:, 0])


def test_get_segment():
    """Test extracting time segment."""
    data = np.random.randn(5000, 1)
    signal = EMGSignal(data=data, sampling_rate=1000.0)
    
    segment = signal.get_segment(1.0, 3.0)
    assert segment.duration == pytest.approx(2.0, abs=0.01)
    assert segment.n_samples == 2000


def test_copy():
    """Test signal copying."""
    data = np.random.randn(1000, 1)
    signal = EMGSignal(data=data, sampling_rate=1000.0)
    
    copy = signal.copy()
    assert copy.n_samples == signal.n_samples
    assert copy.sampling_rate == signal.sampling_rate
    
    # Modify copy - should not affect original
    copy.data[0, 0] = 999
    assert signal.data[0, 0] != 999


def test_to_dict_from_dict():
    """Test serialization/deserialization."""
    data = np.random.randn(100, 2)
    signal = EMGSignal(
        data=data,
        sampling_rate=1000.0,
        channels=['A', 'B'],
        metadata={'test': 'value'}
    )
    
    # Convert to dict
    signal_dict = signal.to_dict()
    
    # Recreate from dict
    restored = EMGSignal.from_dict(signal_dict)
    
    assert restored.n_samples == signal.n_samples
    assert restored.n_channels == signal.n_channels
    assert restored.sampling_rate == signal.sampling_rate
    assert restored.channels == signal.channels
    assert restored.metadata == signal.metadata
