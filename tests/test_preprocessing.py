"""Tests for preprocessing functionality."""

import pytest
import numpy as np
from emg_prostudio.preprocessing.filters import EMGPreprocessor
from emg_prostudio.core.signal import EMGSignal


def test_bandpass_filter(sample_emg_signal):
    """Test bandpass filtering."""
    preprocessor = EMGPreprocessor(sample_emg_signal.sampling_rate)
    filtered = preprocessor.bandpass_filter(sample_emg_signal, lowcut=20, highcut=450)
    
    assert isinstance(filtered, EMGSignal)
    assert filtered.n_samples == sample_emg_signal.n_samples
    assert 'preprocessing' in filtered.metadata


def test_notch_filter(sample_emg_signal):
    """Test notch filtering."""
    preprocessor = EMGPreprocessor(sample_emg_signal.sampling_rate)
    filtered = preprocessor.notch_filter(sample_emg_signal, freq=50.0)
    
    assert isinstance(filtered, EMGSignal)
    assert filtered.n_samples == sample_emg_signal.n_samples


def test_rectify(sample_emg_signal):
    """Test signal rectification."""
    preprocessor = EMGPreprocessor(sample_emg_signal.sampling_rate)
    rectified = preprocessor.rectify(sample_emg_signal)
    
    assert isinstance(rectified, EMGSignal)
    assert np.all(rectified.data >= 0)


def test_envelope_rms(sample_emg_signal):
    """Test RMS envelope extraction."""
    preprocessor = EMGPreprocessor(sample_emg_signal.sampling_rate)
    envelope = preprocessor.envelope(sample_emg_signal, method='rms')
    
    assert isinstance(envelope, EMGSignal)
    assert envelope.n_samples == sample_emg_signal.n_samples


def test_preprocess_pipeline(sample_emg_signal):
    """Test complete preprocessing pipeline."""
    preprocessor = EMGPreprocessor(sample_emg_signal.sampling_rate)
    
    processed = preprocessor.preprocess_pipeline(
        sample_emg_signal,
        remove_powerline=True,
        bandpass=True,
        rectify=True,
        extract_envelope=False
    )
    
    assert isinstance(processed, EMGSignal)
    assert len(processed.metadata.get('preprocessing', [])) > 0
