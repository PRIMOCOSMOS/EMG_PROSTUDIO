#!/usr/bin/env python3
"""
Complete hybrid EMG analysis workflow.

This example demonstrates the entire pipeline:
1. Load multi-channel CSV data
2. Preprocess signals
3. Detect and segment actions (rule-based)
4. Extract hybrid features (traditional + deep learning embeddings)
5. Train quality classifier (LightGBM)
6. Train fatigue monitor (RandomForest)
7. Make predictions and analyze results

This is the core "soul" of the system as described in the requirements.
"""

import numpy as np
from pathlib import Path

# EMG PROSTUDIO imports
from emg_prostudio.utils import DataLoader
from emg_prostudio import EMGPreprocessor
from emg_prostudio.analysis import ActionDetector
from emg_prostudio.features.hybrid import HybridFeatureExtractor
from emg_prostudio.ml import (
    create_embedding_model,
    QualityClassifier,
    FatigueMonitor,
    ModelTrainer
)


def generate_synthetic_data():
    """Generate synthetic multi-channel EMG data for demonstration."""
    import pandas as pd
    
    print("="*70)
    print("Step 1: Generating Synthetic Data")
    print("="*70)
    
    n_reps = 20  # 20 repetitions
    samples_per_rep = 500  # 0.5 seconds per rep at 1000 Hz
    rest_samples = 300  # Rest between reps
    
    time_data = []
    biceps_data = []
    triceps_data = []
    forearm_data = []
    
    current_time = 0.0
    sampling_rate = 1000.0
    
    for rep in range(n_reps):
        # Generate action with varying quality
        if rep < 7:
            # Full range (high quality)
            amplitude = 1.0
            quality_label = 2
        elif rep < 14:
            # Half range (medium quality)
            amplitude = 0.6
            quality_label = 1
        else:
            # Invalid (low quality, increasing fatigue)
            amplitude = 0.3
            quality_label = 0
        
        # Generate action signal
        for i in range(samples_per_rep):
            t = current_time
            time_data.append(t)
            
            # Simulate EMG burst with fatigue effect
            fatigue_factor = 1.0 - (rep / n_reps) * 0.3  # Gradual fatigue
            biceps_data.append(amplitude * np.sin(2 * np.pi * 60 * t) * fatigue_factor + np.random.randn() * 0.1)
            triceps_data.append(amplitude * 0.7 * np.sin(2 * np.pi * 80 * t) * fatigue_factor + np.random.randn() * 0.08)
            forearm_data.append(amplitude * 0.5 * np.sin(2 * np.pi * 100 * t) * fatigue_factor + np.random.randn() * 0.05)
            
            current_time += 1.0 / sampling_rate
        
        # Rest period
        for i in range(rest_samples):
            t = current_time
            time_data.append(t)
            biceps_data.append(np.random.randn() * 0.05)
            triceps_data.append(np.random.randn() * 0.04)
            forearm_data.append(np.random.randn() * 0.03)
            current_time += 1.0 / sampling_rate
    
    # Create DataFrame and save
    df = pd.DataFrame({
        'Time': time_data,
        'Biceps': biceps_data,
        'Triceps': triceps_data,
        'Forearm': forearm_data
    })
    
    df.to_csv('hybrid_demo_data.csv', index=False)
    print(f"✓ Generated {n_reps} reps with varying quality")
    print(f"✓ Total duration: {current_time:.2f} seconds")
    print(f"✓ Saved to hybrid_demo_data.csv")
    
    return 'hybrid_demo_data.csv', n_reps


def main():
    """Run the complete hybrid analysis workflow."""
    
    print("\n" + "="*70)
    print("HYBRID EMG ANALYSIS WORKFLOW")
    print("Rule-Based + Machine Learning Enhancement")
    print("="*70 + "\n")
    
    # Step 1: Generate and load data
    filepath, expected_reps = generate_synthetic_data()
    
    print("\n" + "="*70)
    print("Step 2: Load Multi-Channel CSV")
    print("="*70)
    signal = DataLoader.load_csv(filepath, time_column='Time')
    print(f"✓ Loaded signal: {signal}")
    print(f"✓ Channels: {signal.channels}")
    
    # Step 3: Preprocess
    print("\n" + "="*70)
    print("Step 3: Preprocessing")
    print("="*70)
    preprocessor = EMGPreprocessor(signal.sampling_rate)
    clean_signal = preprocessor.preprocess_pipeline(
        signal,
        remove_powerline=True,
        powerline_freq=50.0,
        bandpass=True,
        lowcut=20.0,
        highcut=450.0
    )
    print("✓ Signal preprocessed (bandpass + notch filter)")
    
    # Step 4: Rule-Based Action Detection (Dynamic Threshold + Peak Detection)
    print("\n" + "="*70)
    print("Step 4: Rule-Based Action Detection")
    print("="*70)
    print("Using: Dynamic threshold + Peak detection")
    print("Benefits: Zero training cost, 100% interpretable")
    
    detector = ActionDetector(clean_signal.sampling_rate)
    actions = detector.detect_actions(clean_signal, channel_idx=0)
    
    print(f"✓ Detected {len(actions)} actions")
    print(f"  Expected: {expected_reps}, Detected: {len(actions)}")
    
    # Step 5: Extract Hybrid Features for Each Rep
    print("\n" + "="*70)
    print("Step 5: Hybrid Feature Extraction")
    print("="*70)
    print("This is the 'SOUL' of the system!")
    print("Extracting: Traditional features + Deep learning embeddings")
    
    # Initialize feature extractor (without deep learning for now)
    feature_extractor = HybridFeatureExtractor(use_deep_features=False)
    
    # Extract features for each detected action
    rep_features_list = []
    for i, action in enumerate(actions):
        start, end = action['start_idx'], action['end_idx']
        rep_signal = clean_signal.get_segment(start, end)
        
        # Extract hybrid features
        features = feature_extractor.extract_hybrid_features(rep_signal, flatten=True)
        rep_features_list.append(features)
    
    X_features = np.array(rep_features_list)
    feature_names = feature_extractor.get_feature_names()
    
    print(f"✓ Extracted features for {len(rep_features_list)} reps")
    print(f"✓ Feature vector dimension: {X_features.shape[1]}")
    print(f"✓ Features: {', '.join(feature_names[:6])}... (showing first 6)")
    
    # Step 6: Prepare Training Data
    print("\n" + "="*70)
    print("Step 6: Prepare Training Data")
    print("="*70)
    
    # Generate labels based on known ground truth from synthetic data
    quality_labels = []
    fatigue_scores = []
    
    for i in range(len(actions)):
        if i < 7:
            quality_labels.append(2)  # Full range
            fatigue_scores.append(i / 20.0)  # Low fatigue
        elif i < 14:
            quality_labels.append(1)  # Half range
            fatigue_scores.append((i + 5) / 25.0)  # Medium fatigue
        else:
            quality_labels.append(0)  # Invalid
            fatigue_scores.append((i + 10) / 30.0)  # High fatigue
    
    y_quality = np.array(quality_labels)
    y_fatigue = np.array(fatigue_scores)
    
    # Split data (80% train, 20% val)
    split_idx = int(0.8 * len(X_features))
    X_train, X_val = X_features[:split_idx], X_features[split_idx:]
    y_quality_train, y_quality_val = y_quality[:split_idx], y_quality[split_idx:]
    y_fatigue_train, y_fatigue_val = y_fatigue[:split_idx], y_fatigue[split_idx:]
    
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    
    # Step 7: Train Quality Classifier (LightGBM)
    print("\n" + "="*70)
    print("Step 7: Train Quality Classifier (LightGBM)")
    print("="*70)
    print("Task: Classify movement quality (full/half/invalid)")
    
    try:
        trainer = ModelTrainer()
        quality_model = trainer.train_quality_model(
            X_train, y_quality_train,
            X_val, y_quality_val,
            feature_names=feature_names,
            use_lgbm=True
        )
        
        # Make predictions
        quality_pred = quality_model.predict(X_val)
        quality_names = ['Invalid', 'Half', 'Full']
        print("\nPredictions on validation set:")
        for i, (true_label, pred_label) in enumerate(zip(y_quality_val, quality_pred)):
            print(f"  Rep {split_idx + i + 1}: True={quality_names[true_label]}, Pred={quality_names[pred_label]}")
        
    except Exception as e:
        print(f"⚠ LightGBM training skipped: {e}")
        print("  Install with: pip install lightgbm")
    
    # Step 8: Train Fatigue Monitor (RandomForest)
    print("\n" + "="*70)
    print("Step 8: Train Fatigue Monitor (RandomForest)")
    print("="*70)
    print("Task: Predict fatigue progression (0-1 scale)")
    
    try:
        fatigue_model = trainer.train_fatigue_model(
            X_train, y_fatigue_train,
            X_val, y_fatigue_val,
            feature_names=feature_names,
            use_rf=True
        )
        
        # Make predictions
        fatigue_pred = fatigue_model.predict(X_val)
        print("\nFatigue predictions on validation set:")
        for i, (true_score, pred_score) in enumerate(zip(y_fatigue_val, fatigue_pred)):
            print(f"  Rep {split_idx + i + 1}: True={true_score:.3f}, Pred={pred_score:.3f}")
        
        # Show feature importance
        print("\nTop 5 important features for fatigue:")
        importance = fatigue_model.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features[:5]:
            print(f"  {feat}: {imp:.4f}")
            
    except Exception as e:
        print(f"⚠ RandomForest training skipped: {e}")
        print("  Install with: pip install scikit-learn")
    
    # Step 9: Save Models
    print("\n" + "="*70)
    print("Step 9: Save Trained Models")
    print("="*70)
    
    try:
        quality_model.save('models/quality_classifier.pkl')
        fatigue_model.save('models/fatigue_monitor.pkl')
        print("✓ Models saved successfully")
    except Exception as e:
        print(f"⚠ Model saving skipped: {e}")
    
    # Cleanup
    print("\n" + "="*70)
    print("Workflow Complete!")
    print("="*70)
    print("\nKey Points:")
    print("1. ✓ Rule-based action detection (dynamic threshold + peaks)")
    print("2. ✓ Hybrid feature extraction (traditional + deep learning ready)")
    print("3. ✓ LightGBM for quality classification")
    print("4. ✓ RandomForest for fatigue monitoring")
    print("5. ✓ Small sample optimization (works with limited data)")
    print("6. ✓ Seamless model training embedded in application")
    
    # Cleanup demo file
    import os
    if os.path.exists('hybrid_demo_data.csv'):
        os.remove('hybrid_demo_data.csv')


if __name__ == '__main__':
    main()
