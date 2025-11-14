"""Test configuration."""

import pytest
import numpy as np
from emg_prostudio.core.signal import EMGSignal


@pytest.fixture
def sample_emg_signal():
    """Create a sample EMG signal for testing."""
    np.random.seed(42)
    sampling_rate = 1000.0
    duration = 5.0
    n_samples = int(duration * sampling_rate)
    
    # Generate synthetic EMG-like signal
    time = np.arange(n_samples) / sampling_rate
    freq = 100  # Hz
    data = np.sin(2 * np.pi * freq * time) + np.random.randn(n_samples) * 0.1
    
    return EMGSignal(
        data=data.reshape(-1, 1),
        sampling_rate=sampling_rate,
        channels=['Channel_0']
    )


@pytest.fixture
def multi_channel_signal():
    """Create a multi-channel EMG signal."""
    np.random.seed(42)
    sampling_rate = 1000.0
    duration = 10.0
    n_samples = int(duration * sampling_rate)
    n_channels = 2
    
    data = np.random.randn(n_samples, n_channels) * 0.5
    
    return EMGSignal(
        data=data,
        sampling_rate=sampling_rate,
        channels=['Biceps', 'Forearm']
    )
