"""Tests for analysis modules."""

import pytest
import numpy as np
from emg_prostudio.analysis.action_detector import ActionDetector
from emg_prostudio.analysis.fatigue_monitor import FatigueMonitor
from emg_prostudio.utils.sample_data import generate_sample_emg


def test_action_detector():
    """Test action detection."""
    # Generate sample with known repetitions
    signal = generate_sample_emg(
        duration=20.0,
        n_repetitions=5,
        sampling_rate=1000.0
    )
    
    detector = ActionDetector(sampling_rate=signal.sampling_rate)
    actions = detector.detect_actions(signal)
    
    # Should detect some actions
    assert len(actions) > 0
    assert all('start_time' in a for a in actions)
    assert all('end_time' in a for a in actions)


def test_action_classification():
    """Test action amplitude classification."""
    signal = generate_sample_emg(
        duration=20.0,
        n_repetitions=5,
        movement_quality_variation=True
    )
    
    detector = ActionDetector(sampling_rate=signal.sampling_rate)
    actions = detector.detect_actions(signal)
    actions = detector.classify_amplitude(actions)
    
    # Should have classification
    assert all('amplitude_class' in a for a in actions)
    assert all('quality_score' in a for a in actions)


def test_action_counting():
    """Test action counting."""
    signal = generate_sample_emg(duration=20.0, n_repetitions=5)
    
    detector = ActionDetector(sampling_rate=signal.sampling_rate)
    actions = detector.detect_actions(signal)
    actions = detector.classify_amplitude(actions)
    counts = detector.count_actions(actions)
    
    assert 'total' in counts
    assert 'full' in counts
    assert 'half' in counts
    assert 'invalid' in counts
    assert counts['total'] >= 0


def test_fatigue_trajectory():
    """Test fatigue trajectory computation."""
    signal = generate_sample_emg(
        duration=30.0,
        n_repetitions=10,
        fatigue_effect=True
    )
    
    monitor = FatigueMonitor(sampling_rate=signal.sampling_rate)
    trajectory = monitor.compute_fatigue_trajectory(
        signal,
        window_size=1.0,
        overlap=0.5
    )
    
    assert 'time' in trajectory
    assert 'mdf' in trajectory
    assert 'mpf' in trajectory
    assert 'rms' in trajectory
    assert len(trajectory['time']) > 0


def test_fatigue_analysis():
    """Test complete fatigue analysis."""
    signal = generate_sample_emg(
        duration=30.0,
        fatigue_effect=True
    )
    
    monitor = FatigueMonitor(sampling_rate=signal.sampling_rate)
    trajectory = monitor.compute_fatigue_trajectory(signal)
    analysis = monitor.analyze_fatigue_progression(trajectory)
    
    assert 'duration' in analysis
    assert 'mdf_trend' in analysis
    assert 'mpf_trend' in analysis
    assert 'rms_trend' in analysis
