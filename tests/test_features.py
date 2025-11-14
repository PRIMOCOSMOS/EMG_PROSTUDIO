"""Tests for feature extraction."""

import pytest
import numpy as np
from emg_prostudio.features.activation import ActivationFeatures
from emg_prostudio.features.fatigue import FatigueFeatures


def test_activation_features_rms():
    """Test RMS feature."""
    data = np.array([1, -1, 2, -2, 3, -3])
    features = ActivationFeatures(sampling_rate=1000.0)
    
    rms = features.rms(data)
    expected = np.sqrt(np.mean(data ** 2))
    assert rms == pytest.approx(expected)


def test_activation_features_mav():
    """Test MAV feature."""
    data = np.array([1, -1, 2, -2, 3, -3])
    features = ActivationFeatures(sampling_rate=1000.0)
    
    mav = features.mav(data)
    expected = np.mean(np.abs(data))
    assert mav == pytest.approx(expected)


def test_activation_features_extract_all():
    """Test extracting all activation features."""
    data = np.random.randn(1000, 2)
    features = ActivationFeatures(sampling_rate=1000.0)
    
    result = features.extract_all(data)
    
    assert 'rms' in result
    assert 'mav' in result
    assert 'iemg' in result
    assert 'variance' in result
    assert 'waveform_length' in result


def test_fatigue_features_median_frequency():
    """Test median frequency calculation."""
    # Generate signal with known frequency
    fs = 1000.0
    t = np.arange(1000) / fs
    freq = 100  # Hz
    data = np.sin(2 * np.pi * freq * t)
    
    features = FatigueFeatures(sampling_rate=fs)
    mdf = features.median_frequency(data)
    
    # Should be close to 100 Hz
    assert 80 < mdf < 120


def test_fatigue_features_zero_crossings():
    """Test zero crossing count."""
    # Simple test signal
    data = np.array([1, 2, 1, -1, -2, -1, 1, 2])
    features = FatigueFeatures(sampling_rate=1000.0)
    
    zc = features.zero_crossings(data)
    assert zc >= 2  # At least 2 crossings


def test_fatigue_features_extract_all():
    """Test extracting all fatigue features."""
    data = np.random.randn(1000)
    features = FatigueFeatures(sampling_rate=1000.0)
    
    result = features.extract_all(data)
    
    assert 'mdf' in result
    assert 'mpf' in result
    assert 'zc' in result
