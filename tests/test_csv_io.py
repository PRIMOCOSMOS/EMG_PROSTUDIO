"""Test CSV loading and saving functionality."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from emg_prostudio.utils.data_io import DataLoader, DataSaver
from emg_prostudio.core.signal import EMGSignal


def test_csv_load_save_basic():
    """Test basic CSV load and save."""
    # Create test data
    n_samples = 1000
    n_channels = 3
    sampling_rate = 1000.0
    
    data = np.random.randn(n_samples, n_channels)
    channels = ['CH1', 'CH2', 'CH3']
    
    signal = EMGSignal(
        data=data,
        sampling_rate=sampling_rate,
        channels=channels
    )
    
    # Save to temporary CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'test.csv'
        
        # Save
        DataSaver.save_csv(signal, filepath, include_time=True)
        
        # Load with time column specified
        loaded_signal = DataLoader.load_csv(
            filepath,
            time_column='Time'
        )
        
        # Verify
        assert loaded_signal.n_samples == n_samples
        assert loaded_signal.n_channels == n_channels
        # Sampling rate should be calculated from time column
        assert abs(loaded_signal.sampling_rate - sampling_rate) < 1.0
        np.testing.assert_array_almost_equal(loaded_signal.data, data, decimal=6)


def test_csv_load_with_time_column():
    """Test CSV loading with time column."""
    # Create test data with time
    n_samples = 1000
    n_channels = 2
    sampling_rate = 500.0
    
    data = np.random.randn(n_samples, n_channels)
    time = np.arange(n_samples) / sampling_rate
    
    signal = EMGSignal(
        data=data,
        sampling_rate=sampling_rate,
        channels=['Biceps', 'Triceps']
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'test_with_time.csv'
        
        # Save with time column
        DataSaver.save_csv(signal, filepath, include_time=True)
        
        # Load with time column specified
        loaded_signal = DataLoader.load_csv(
            filepath,
            time_column='Time'
        )
        
        # Verify sampling rate calculated from time
        assert abs(loaded_signal.sampling_rate - sampling_rate) < 1.0  # Allow 1 Hz tolerance
        assert loaded_signal.n_channels == n_channels


def test_csv_load_specific_channels():
    """Test loading specific channels from CSV."""
    # Create test data with multiple channels
    n_samples = 500
    data = np.random.randn(n_samples, 5)
    
    signal = EMGSignal(
        data=data,
        sampling_rate=1000.0,
        channels=['CH1', 'CH2', 'CH3', 'CH4', 'CH5']
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'test_multi.csv'
        
        # Save
        DataSaver.save_csv(signal, filepath, include_time=False)
        
        # Load only specific channels
        loaded_signal = DataLoader.load_csv(
            filepath,
            sampling_rate=1000.0,
            channel_columns=['CH1', 'CH3', 'CH5']
        )
        
        # Verify only 3 channels loaded
        assert loaded_signal.n_channels == 3
        assert loaded_signal.channels == ['CH1', 'CH3', 'CH5']


def test_csv_load_no_header():
    """Test loading CSV without header."""
    # Create test data
    n_samples = 100
    n_channels = 2
    data = np.random.randn(n_samples, n_channels)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'test_no_header.csv'
        
        # Save without header using numpy
        np.savetxt(filepath, data, delimiter=',')
        
        # Load without header
        loaded_signal = DataLoader.load_csv(
            filepath,
            sampling_rate=1000.0,
            header=None
        )
        
        # Verify
        assert loaded_signal.n_channels == n_channels
        assert loaded_signal.channels == ['Channel_0', 'Channel_1']
        np.testing.assert_array_almost_equal(loaded_signal.data, data, decimal=6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
