"""Generate sample EMG data for testing and demonstration."""

import numpy as np
from emg_prostudio.core.signal import EMGSignal


def generate_sample_emg(
    duration: float = 30.0,
    sampling_rate: float = 1000.0,
    n_repetitions: int = 10,
    n_channels: int = 2,
    noise_level: float = 0.1,
    fatigue_effect: bool = True,
    movement_quality_variation: bool = True
) -> EMGSignal:
    """
    Generate synthetic EMG signal for dumbbell curl exercise.
    
    This function creates realistic-looking EMG data with:
    - Simulated muscle activation during repetitions
    - Progressive fatigue effects (frequency shift, amplitude increase)
    - Variable movement quality (full, half, invalid reps)
    - Background noise
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Sampling frequency in Hz
        n_repetitions: Number of exercise repetitions
        n_channels: Number of EMG channels
        noise_level: Background noise level (0-1)
        fatigue_effect: Simulate muscle fatigue progression
        movement_quality_variation: Vary movement quality across reps
        
    Returns:
        EMGSignal object with synthetic data
    """
    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    
    # Initialize signal
    data = np.zeros((n_samples, n_channels))
    
    # Parameters for each repetition
    rep_duration = duration / (n_repetitions + 2)  # Leave some rest time
    
    for rep in range(n_repetitions):
        # Timing
        rep_start = (rep + 1) * rep_duration
        rep_end = rep_start + rep_duration * 0.7  # 70% active, 30% rest
        
        # Find indices
        start_idx = int(rep_start * sampling_rate)
        end_idx = int(rep_end * sampling_rate)
        
        if end_idx >= n_samples:
            break
        
        rep_samples = end_idx - start_idx
        rep_time = np.arange(rep_samples) / sampling_rate
        
        # Movement quality (decreases with fatigue)
        if movement_quality_variation:
            # First 60% are full, next 30% are half, last 10% might be invalid
            if rep < n_repetitions * 0.6:
                quality = 1.0  # Full movement
            elif rep < n_repetitions * 0.9:
                quality = 0.6  # Half movement
            else:
                quality = 0.3  # Invalid/minimal movement
        else:
            quality = 1.0
        
        # Fatigue progression
        fatigue_factor = 1.0
        frequency_shift = 0.0
        if fatigue_effect:
            # Amplitude increases with fatigue (compensation)
            fatigue_factor = 1.0 + (rep / n_repetitions) * 0.5
            # Frequency content shifts down with fatigue
            frequency_shift = -(rep / n_repetitions) * 20  # Hz
        
        # Generate activation pattern for this rep
        for ch in range(n_channels):
            # Base frequency (muscle fiber firing rate)
            base_freq = 60 + ch * 10 + frequency_shift  # Hz
            
            # Amplitude envelope (burst pattern)
            envelope = quality * fatigue_factor * np.sin(np.pi * rep_time / (rep_duration * 0.7)) ** 2
            
            # Generate EMG-like signal
            # Mix of multiple frequency components
            signal_component = np.zeros(rep_samples)
            for harmonic in range(1, 5):
                freq = base_freq * harmonic
                phase = np.random.rand() * 2 * np.pi
                amplitude = 1.0 / harmonic
                signal_component += amplitude * np.sin(2 * np.pi * freq * rep_time + phase)
            
            # Apply envelope
            signal_component *= envelope
            
            # Add to data
            data[start_idx:end_idx, ch] += signal_component
    
    # Add background noise
    for ch in range(n_channels):
        # White noise
        noise = np.random.randn(n_samples) * noise_level
        
        # Low-frequency baseline wander
        baseline_freq = 0.5  # Hz
        baseline = 0.05 * np.sin(2 * np.pi * baseline_freq * time)
        
        # Power line interference (50 Hz)
        powerline = 0.02 * np.sin(2 * np.pi * 50 * time)
        
        data[:, ch] += noise + baseline + powerline
    
    # Normalize to realistic EMG amplitude range (μV to mV)
    data = data * 0.5  # Scale to reasonable amplitude
    
    # Create EMG signal object
    channel_names = [f"Biceps_{i+1}" if i == 0 else f"Forearm_{i}" 
                     for i in range(n_channels)]
    
    metadata = {
        'type': 'synthetic',
        'exercise': 'dumbbell_curl',
        'n_repetitions': n_repetitions,
        'fatigue_simulation': fatigue_effect,
        'movement_quality_variation': movement_quality_variation,
        'noise_level': noise_level
    }
    
    return EMGSignal(
        data=data,
        sampling_rate=sampling_rate,
        channels=channel_names,
        metadata=metadata
    )


def generate_fatigue_progression_emg(
    duration: float = 60.0,
    sampling_rate: float = 1000.0,
    initial_activation: float = 1.0,
    fatigue_rate: float = 0.5
) -> EMGSignal:
    """
    Generate EMG signal showing clear fatigue progression.
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Sampling frequency in Hz
        initial_activation: Initial activation level
        fatigue_rate: Rate of fatigue development
        
    Returns:
        EMGSignal with fatigue progression
    """
    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    
    # Single channel for simplicity
    data = np.zeros((n_samples, 1))
    
    # Continuous activation with fatigue
    for i, t in enumerate(time):
        # Fatigue factor (increases amplitude, decreases frequency)
        fatigue = 1 + fatigue_rate * (t / duration)
        freq_shift = -30 * (t / duration)  # Frequency decreases
        
        # Mean frequency starts at 100 Hz, decreases to ~70 Hz
        mean_freq = 100 + freq_shift
        
        # Generate signal
        signal_value = 0
        for harmonic in range(1, 6):
            freq = mean_freq * harmonic / 3
            amplitude = initial_activation * fatigue / harmonic
            signal_value += amplitude * np.sin(2 * np.pi * freq * t)
        
        data[i, 0] = signal_value
    
    # Add noise
    data += np.random.randn(n_samples, 1) * 0.1
    
    return EMGSignal(
        data=data,
        sampling_rate=sampling_rate,
        channels=['Muscle'],
        metadata={'type': 'fatigue_progression'}
    )
