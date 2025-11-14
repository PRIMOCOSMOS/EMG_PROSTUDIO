#!/usr/bin/env python3
"""
Example demonstrating CSV file loading for multi-channel EMG data.

This example shows how to:
1. Load multi-channel CSV files
2. Handle different CSV formats
3. Process and analyze the data
"""

import numpy as np
from emg_prostudio.utils.data_io import DataLoader, DataSaver
from emg_prostudio import EMGPreprocessor
from emg_prostudio.analysis import ActionDetector, FatigueMonitor
from emg_prostudio.core.signal import EMGSignal


def create_sample_csv():
    """Create a sample multi-channel CSV file for demonstration."""
    import pandas as pd
    
    # Generate sample data
    n_samples = 5000  # 5 seconds at 1000 Hz
    sampling_rate = 1000.0
    
    # Simulate 3-channel EMG data
    time = np.arange(n_samples) / sampling_rate
    ch1 = 0.5 * np.sin(2 * np.pi * 60 * time) + np.random.randn(n_samples) * 0.1
    ch2 = 0.3 * np.sin(2 * np.pi * 80 * time) + np.random.randn(n_samples) * 0.1
    ch3 = 0.4 * np.sin(2 * np.pi * 100 * time) + np.random.randn(n_samples) * 0.1
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': time,
        'Biceps': ch1,
        'Triceps': ch2,
        'Forearm': ch3
    })
    
    # Save to CSV
    df.to_csv('sample_emg_data.csv', index=False)
    print("✓ Created sample_emg_data.csv")
    return 'sample_emg_data.csv'


def example_1_basic_loading():
    """Example 1: Basic CSV loading with time column."""
    print("\n" + "="*70)
    print("Example 1: Basic CSV Loading")
    print("="*70)
    
    filepath = create_sample_csv()
    
    # Load CSV with time column
    signal = DataLoader.load_csv(
        filepath,
        time_column='Time'  # Sampling rate calculated from time
    )
    
    print(f"\nLoaded signal: {signal}")
    print(f"Channels: {signal.channels}")
    print(f"Sampling rate: {signal.sampling_rate:.2f} Hz")
    print(f"Duration: {signal.duration:.2f} seconds")


def example_2_specific_channels():
    """Example 2: Load only specific channels."""
    print("\n" + "="*70)
    print("Example 2: Loading Specific Channels")
    print("="*70)
    
    filepath = 'sample_emg_data.csv'
    
    # Load only Biceps and Forearm channels
    signal = DataLoader.load_csv(
        filepath,
        channel_columns=['Biceps', 'Forearm'],
        time_column='Time'
    )
    
    print(f"\nLoaded signal: {signal}")
    print(f"Selected channels: {signal.channels}")


def example_3_without_time_column():
    """Example 3: Load CSV without time column (specify sampling rate)."""
    print("\n" + "="*70)
    print("Example 3: Loading Without Time Column")
    print("="*70)
    
    # Create CSV without time column
    signal = DataLoader.load_csv(
        'sample_emg_data.csv',
        time_column='Time'
    )
    
    # Save without time column
    DataSaver.save_csv(signal, 'emg_no_time.csv', include_time=False)
    print("✓ Created emg_no_time.csv (without time column)")
    
    # Load with specified sampling rate
    loaded = DataLoader.load_csv(
        'emg_no_time.csv',
        sampling_rate=1000.0  # Must specify sampling rate
    )
    
    print(f"\nLoaded signal: {loaded}")
    print(f"Specified sampling rate: {loaded.sampling_rate} Hz")


def example_4_preprocessing_and_analysis():
    """Example 4: Complete workflow - load, preprocess, analyze."""
    print("\n" + "="*70)
    print("Example 4: Complete Analysis Workflow")
    print("="*70)
    
    # Load data
    signal = DataLoader.load_csv(
        'sample_emg_data.csv',
        time_column='Time'
    )
    print(f"✓ Loaded signal: {signal}")
    
    # Preprocess
    preprocessor = EMGPreprocessor(signal.sampling_rate)
    clean_signal = preprocessor.preprocess_pipeline(
        signal,
        remove_powerline=True,
        powerline_freq=50.0,
        bandpass=True,
        lowcut=20.0,
        highcut=450.0
    )
    print("✓ Preprocessed signal")
    
    # Detect actions (using first channel)
    detector = ActionDetector(clean_signal.sampling_rate)
    actions = detector.detect_actions(clean_signal, channel_idx=0)
    actions = detector.classify_amplitude(actions)
    counts = detector.count_actions(actions)
    
    print(f"\n✓ Action Detection Results:")
    print(f"  Total actions: {counts['total']}")
    print(f"  Full range: {counts['full']}")
    print(f"  Half range: {counts['half']}")
    print(f"  Invalid: {counts['invalid']}")
    
    # Analyze fatigue
    monitor = FatigueMonitor(clean_signal.sampling_rate)
    trajectory = monitor.compute_fatigue_trajectory(
        clean_signal,
        window_size=1.0,
        channel_idx=0
    )
    
    print(f"\n✓ Fatigue Analysis:")
    print(f"  Initial MDF: {trajectory['mdf'][0]:.2f} Hz")
    print(f"  Final MDF: {trajectory['mdf'][-1]:.2f} Hz")
    print(f"  MDF change: {trajectory['mdf'][-1] - trajectory['mdf'][0]:.2f} Hz")


def example_5_different_formats():
    """Example 5: Handle different CSV formats."""
    print("\n" + "="*70)
    print("Example 5: Different CSV Formats")
    print("="*70)
    
    # Format 1: CSV without header
    data = np.random.randn(1000, 3)
    np.savetxt('no_header.csv', data, delimiter=',')
    
    signal1 = DataLoader.load_csv(
        'no_header.csv',
        sampling_rate=1000.0,
        header=None
    )
    print(f"\n✓ No header CSV: {signal1}")
    print(f"  Auto-generated channels: {signal1.channels}")
    
    # Format 2: CSV with custom delimiter (tab-separated)
    signal = DataLoader.load_csv('sample_emg_data.csv', time_column='Time')
    DataSaver.save_csv(signal, 'tab_separated.tsv', delimiter='\t', include_time=False)
    
    signal2 = DataLoader.load_csv(
        'tab_separated.tsv',
        sampling_rate=1000.0,
        delimiter='\t'
    )
    print(f"\n✓ Tab-separated CSV: {signal2}")


def main():
    """Run all examples."""
    print("="*70)
    print("EMG PROSTUDIO - CSV Loading Examples")
    print("="*70)
    
    example_1_basic_loading()
    example_2_specific_channels()
    example_3_without_time_column()
    example_4_preprocessing_and_analysis()
    example_5_different_formats()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    
    # Cleanup
    import os
    for f in ['sample_emg_data.csv', 'emg_no_time.csv', 'no_header.csv', 'tab_separated.tsv']:
        if os.path.exists(f):
            os.remove(f)
    print("\n✓ Cleaned up temporary files")


if __name__ == '__main__':
    main()
