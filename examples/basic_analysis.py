#!/usr/bin/env python3
"""Example usage of EMG_PROSTUDIO for analyzing dumbbell curl exercise.

This script demonstrates the complete workflow:
1. Generate/load EMG data
2. Preprocess the signal
3. Detect and count actions
4. Analyze muscle activation
5. Monitor fatigue progression
6. Generate comprehensive report
"""

import numpy as np
from emg_prostudio import EMGSignal, EMGPreprocessor, FeatureExtractor
from emg_prostudio.analysis import ActionDetector, FatigueMonitor
from emg_prostudio.utils import generate_sample_emg


def main():
    """Run example analysis."""
    print("=" * 70)
    print("EMG_PROSTUDIO - Example Analysis")
    print("=" * 70)
    
    # Step 1: Generate sample data
    print("\n[1/6] Generating sample EMG data...")
    emg_signal = generate_sample_emg(
        duration=30.0,           # 30 seconds
        sampling_rate=1000.0,    # 1000 Hz
        n_repetitions=10,        # 10 dumbbell curls
        n_channels=2,            # 2 EMG channels
        noise_level=0.1,
        fatigue_effect=True,     # Simulate fatigue
        movement_quality_variation=True  # Vary movement quality
    )
    
    print(f"   Signal info: {emg_signal}")
    print(f"   Channels: {emg_signal.channels}")
    
    # Step 2: Preprocess the signal
    print("\n[2/6] Preprocessing signal...")
    preprocessor = EMGPreprocessor(emg_signal.sampling_rate)
    
    clean_signal = preprocessor.preprocess_pipeline(
        emg_signal,
        remove_powerline=True,   # Remove 50 Hz interference
        powerline_freq=50.0,
        bandpass=True,           # Bandpass filter 20-450 Hz
        lowcut=20.0,
        highcut=450.0,
        rectify=False,
        extract_envelope=False
    )
    
    print(f"   Applied filters: {len(clean_signal.metadata['preprocessing'])}")
    
    # Step 3: Detect and count actions
    print("\n[3/6] Detecting actions...")
    detector = ActionDetector(
        sampling_rate=clean_signal.sampling_rate,
        threshold_factor=2.0,
        min_duration=0.3,
        max_duration=3.0
    )
    
    actions = detector.detect_actions(clean_signal, channel_idx=0)
    print(f"   Detected {len(actions)} potential actions")
    
    # Classify by amplitude
    actions = detector.classify_amplitude(
        actions,
        full_threshold=0.8,
        half_threshold=0.4
    )
    
    counts = detector.count_actions(actions)
    print(f"\n   Action Classification:")
    print(f"   ├─ Full range:  {counts['full']}")
    print(f"   ├─ Half range:  {counts['half']}")
    print(f"   └─ Invalid:     {counts['invalid']}")
    print(f"   Quality rate:   {counts['valid']/counts['total']*100:.1f}%")
    
    # Step 4: Extract features
    print("\n[4/6] Extracting features...")
    extractor = FeatureExtractor(clean_signal.sampling_rate)
    features = extractor.extract_all(clean_signal)
    
    print(f"   Activation features:")
    print(f"   ├─ RMS:      {features['rms'][0]:.4f}")
    print(f"   ├─ MAV:      {features['mav'][0]:.4f}")
    print(f"   └─ iEMG:     {features['iemg'][0]:.4f}")
    
    # Get first channel fatigue features
    fatigue_keys = [k for k in features.keys() if 'mdf' in k or 'mpf' in k or 'zc' in k]
    if fatigue_keys:
        print(f"\n   Fatigue features:")
        for key in fatigue_keys[:3]:  # Show first 3
            print(f"   ├─ {key.upper()}:      {features[key]}")
    
    # Step 5: Monitor fatigue progression
    print("\n[5/6] Analyzing fatigue progression...")
    monitor = FatigueMonitor(clean_signal.sampling_rate)
    
    trajectory = monitor.compute_fatigue_trajectory(
        clean_signal,
        window_size=1.0,
        overlap=0.5,
        channel_idx=0
    )
    
    fatigue_analysis = monitor.analyze_fatigue_progression(
        trajectory,
        subjective_fatigue_time=None,
        exhaustion_time=None
    )
    
    print(f"   MDF trend:")
    print(f"   ├─ Decline rate: {fatigue_analysis['mdf_trend']['decline_rate']:.2f}%/s")
    print(f"   └─ R²:          {fatigue_analysis['mdf_trend']['r_squared']:.3f}")
    
    print(f"\n   Fatigue index:")
    print(f"   ├─ Maximum:     {fatigue_analysis['max_fatigue_index']:.2f}")
    print(f"   └─ Final:       {fatigue_analysis['final_fatigue_index']:.2f}")
    
    # Step 6: Generate summary
    print("\n[6/6] Summary...")
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nExercise Duration:     {emg_signal.duration:.1f} seconds")
    print(f"Total Repetitions:     {counts['total']}")
    print(f"Movement Quality:      {counts['valid']/counts['total']*100:.1f}%")
    print(f"Fatigue Level:         {fatigue_analysis['final_fatigue_index']:.2f}/1.0")
    
    if counts['valid']/counts['total'] > 0.8:
        quality_msg = "Excellent - Maintain this quality!"
    elif counts['valid']/counts['total'] > 0.6:
        quality_msg = "Good - Watch for fatigue effects"
    else:
        quality_msg = "Needs Improvement - Consider rest"
    
    print(f"\nRecommendation:        {quality_msg}")
    print("=" * 70)


if __name__ == '__main__':
    main()
