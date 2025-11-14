#!/usr/bin/env python3
"""
Quick test to verify imports work in VSCode development environment.

Run this file in VSCode with F5 (using "Python: Current File" configuration)
or from terminal with: PYTHONPATH=. python test_imports.py
"""

print("Testing EMG_PROSTUDIO imports...")
print("=" * 70)

try:
    from emg_prostudio.core.signal import EMGSignal
    print("✓ EMGSignal imported successfully")
except ImportError as e:
    print(f"✗ EMGSignal import failed: {e}")
    print("  Make sure dependencies are installed: pip install -r requirements.txt")

try:
    from emg_prostudio.plugins import AnalysisPlugin
    print("✓ AnalysisPlugin imported successfully")
except ImportError as e:
    print(f"✗ AnalysisPlugin import failed: {e}")

try:
    from emg_prostudio.preprocessing.filters import EMGPreprocessor
    print("✓ EMGPreprocessor imported successfully")
except ImportError as e:
    print(f"✗ EMGPreprocessor import failed: {e}")

try:
    from emg_prostudio.features.extractor import FeatureExtractor
    print("✓ FeatureExtractor imported successfully")
except ImportError as e:
    print(f"✗ FeatureExtractor import failed: {e}")

try:
    from emg_prostudio.analysis.action_detector import ActionDetector
    print("✓ ActionDetector imported successfully")
except ImportError as e:
    print(f"✗ ActionDetector import failed: {e}")

try:
    from emg_prostudio.analysis.fatigue_monitor import FatigueMonitor
    print("✓ FatigueMonitor imported successfully")
except ImportError as e:
    print(f"✗ FatigueMonitor import failed: {e}")

print("=" * 70)
print("\n✓ All imports successful!")
print("\nYou can now develop and test EMG_PROSTUDIO in VSCode")
print("without running 'pip install -e .'")
