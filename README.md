# EMG_PROSTUDIO

**Professional EMG Signal Analysis Studio for Exercise Monitoring**

A comprehensive, extensible Python application for analyzing electromyography (EMG) signals during dumbbell exercises. Features real-time visualization with a futuristic UI, advanced signal processing, muscle activation and fatigue analysis, and automatic movement quality assessment.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

### 📊 Signal Processing
- **Advanced Preprocessing Pipeline**
  - Bandpass filtering (20-450 Hz) for motion artifact and baseline drift removal
  - Notch filtering for power line interference (50/60 Hz)
  - Signal rectification and envelope extraction
  - Multiple filtering methods (Butterworth, IIR)

### 💪 Muscle Activation Analysis
- **Comprehensive Feature Extraction**
  - RMS (Root Mean Square) - activation intensity
  - MAV (Mean Absolute Value) - average activity
  - iEMG (Integrated EMG) - total muscle activity
  - Variance - signal variability
  - Waveform Length - signal complexity

### 🏋️ Action Recognition & Counting
- **Automatic Repetition Detection**
  - Peak detection with envelope analysis
  - Individual action segmentation
  - Movement amplitude classification (Full/Half/Invalid)
  - Quality score computation
  - Real-time action counting

### 📉 Fatigue Monitoring
- **Objective Fatigue Indicators**
  - MDF (Median Frequency) - decreases with fatigue
  - MPF (Mean Power Frequency) - frequency shift tracking
  - Zero Crossings - dominant frequency analysis
  - Spectral moments - comprehensive frequency analysis
- **Fatigue Progression Analysis**
  - Real-time fatigue index (0-1 scale)
  - Trend analysis with linear regression
  - Fatigue onset detection
  - Subjective vs objective correlation

### 🎨 Futuristic UI
- **Modern Interface Design**
  - Dark theme with cyan/blue accents
  - Sci-fi inspired aesthetics
  - Real-time signal visualization
  - Interactive plots with PyQtGraph
  - Multi-tab organization

### 🔌 Extensible Architecture
- **Plugin System**
  - Traditional signal processing methods
  - Lightweight deep learning integration (PyTorch)
  - External model API support
  - Easy integration of new analysis methods

## 🚀 Quick Start

### VSCode Development (No Installation Required!)

**For VSCode users**: Simply open the project folder in VSCode with dependencies installed:

```bash
# 1. Clone and open in VSCode
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
code EMG_PROSTUDIO

# 2. Install only dependencies (not the package itself)
pip install -r requirements.txt
# or for core only: pip install -r requirements-core.txt

# 3. Start coding! Imports work immediately, no pip install -e . needed
```

The `.vscode/settings.json` automatically configures Python paths for development. See [.vscode/README.md](.vscode/README.md) for details.

### Installation (For Non-VSCode or Production Use)

#### Option 1: Full Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO

# Install the package with all dependencies
pip install -e .
```

#### Option 2: Core Dependencies Only (No GUI)
```bash
# Clone the repository
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO

# Install only core dependencies (for programmatic use)
pip install -r requirements-core.txt
```

#### Option 3: All Dependencies Separately
```bash
# Clone the repository
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO

# Install all dependencies including GUI support
pip install -r requirements.txt
```

#### Optional: Deep Learning Support
```bash
# After installing core dependencies, add PyTorch
pip install torch torchvision
```

### Running the Application

```bash
# Launch the GUI
python main.py
```

Or use the package:

```python
from emg_prostudio.ui.main_window import main
main()
```

### Basic Usage Example

```python
from emg_prostudio import EMGSignal, EMGPreprocessor, FeatureExtractor
from emg_prostudio.analysis import ActionDetector, FatigueMonitor
from emg_prostudio.utils import generate_sample_emg

# Generate or load EMG data
emg_signal = generate_sample_emg(duration=30, n_repetitions=10)

# Preprocess the signal
preprocessor = EMGPreprocessor(emg_signal.sampling_rate)
clean_signal = preprocessor.preprocess_pipeline(
    emg_signal,
    remove_powerline=True,
    bandpass=True
)

# Detect and count actions
detector = ActionDetector(sampling_rate=clean_signal.sampling_rate)
actions = detector.detect_actions(clean_signal)
actions = detector.classify_amplitude(actions)
counts = detector.count_actions(actions)

print(f"Detected {counts['total']} repetitions")
print(f"Full range: {counts['full']}, Half: {counts['half']}, Invalid: {counts['invalid']}")

# Analyze fatigue
monitor = FatigueMonitor(sampling_rate=clean_signal.sampling_rate)
trajectory = monitor.compute_fatigue_trajectory(clean_signal)
fatigue_analysis = monitor.analyze_fatigue_progression(trajectory)

print(f"MDF decline rate: {fatigue_analysis['mdf_trend']['decline_rate']:.2f}%/s")
print(f"Final fatigue index: {fatigue_analysis['final_fatigue_index']:.2f}")
```

## 📁 Project Structure

```
EMG_PROSTUDIO/
├── emg_prostudio/           # Main package
│   ├── core/                # Core signal representation
│   │   └── signal.py        # EMGSignal class
│   ├── preprocessing/       # Signal preprocessing
│   │   └── filters.py       # Filtering methods
│   ├── features/            # Feature extraction
│   │   ├── activation.py    # Muscle activation features
│   │   ├── fatigue.py       # Fatigue features
│   │   └── extractor.py     # Main extractor
│   ├── analysis/            # Analysis modules
│   │   ├── action_detector.py   # Action detection
│   │   └── fatigue_monitor.py   # Fatigue monitoring
│   ├── plugins/             # Plugin system
│   │   ├── traditional.py   # Traditional methods
│   │   └── lightweight_dl.py # Deep learning
│   ├── ui/                  # User interface
│   │   └── main_window.py   # Main GUI
│   └── utils/               # Utilities
│       ├── data_io.py       # Data loading/saving
│       └── sample_data.py   # Sample generation
├── tests/                   # Unit tests
├── data/                    # Data directory
├── main.py                  # Application entry point
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
└── README.md               # This file
```

## 🎯 Technical Approaches

The application supports three complementary technical paths:

### 1. Traditional Signal Processing ✅
- Butterworth bandpass/highpass/notch filters
- Time-domain features (RMS, MAV, iEMG, VAR, WL)
- Frequency-domain features (MDF, MPF, spectral moments)
- Statistical threshold-based detection
- **Status**: Fully implemented

### 2. Lightweight Deep Learning 🤖
- Small feedforward neural networks
- Perceptron-based classifiers
- Minimal computational requirements
- Optional PyTorch integration
- **Status**: Framework ready, training interface available

### 3. External Model Integration 🔌
- Plugin-based architecture
- Easy integration of pre-trained models
- API support for external services
- **Status**: Plugin system implemented, ready for extensions

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=emg_prostudio --cov-report=html

# Run specific test module
pytest tests/test_core.py
```

## 📊 Analysis Capabilities

### Movement Quality Assessment
- **Full Range**: Proper elbow flexion angle, high muscle activation
- **Half Range**: Insufficient elbow flexion, moderate activation
- **Invalid**: Minimal movement, low activation
- Quality score: 0.0 (invalid) to 1.0 (perfect)

### Fatigue Indicators
Based on scientific literature:
- **MDF/MPF**: Decrease indicates muscle fatigue (spectral shift)
- **RMS/MAV**: May increase due to motor unit recruitment
- **Fatigue Index**: Combined metric (0=fresh, 1=exhausted)

### Subjective Correlation
- Compare objective metrics with subjective reports
- Fatigue onset detection
- Exhaustion point identification
- Sensitivity validation

## 🎨 UI Features

- **Real-time Visualization**: Live signal plots with PyQtGraph
- **Multi-view Tabs**: 
  - Signal View: Raw and preprocessed signals
  - Analysis: Action detection with color coding
  - Fatigue Monitor: MDF/MPF trends and fatigue index
  - Results: Comprehensive text report
- **Futuristic Theme**: Dark background with cyan/green accents
- **Interactive Controls**: Load data, generate samples, configure analysis
- **Export Results**: Save analysis reports in text format

## 🔬 Scientific Background

This application implements methods from EMG signal processing literature:

1. **Preprocessing**: Bandpass 20-450 Hz recommended for surface EMG
2. **Activation Features**: RMS and MAV are gold standards for activation level
3. **Fatigue Features**: MDF and MPF show consistent decline with fatigue
4. **Action Detection**: Envelope-based detection with adaptive thresholding
5. **Quality Assessment**: Amplitude-based classification correlates with movement range

## 🛠️ Development

### Adding New Analysis Methods

```python
from emg_prostudio.plugins import AnalysisPlugin

class MyCustomAnalysis(AnalysisPlugin):
    def __init__(self):
        super().__init__("MyCustomAnalysis", "1.0.0")
    
    def analyze(self, emg_signal, **kwargs):
        # Your analysis code here
        return {'results': 'data'}
    
    def get_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': 'My custom analysis method'
        }

# Register the plugin
from emg_prostudio.plugins import PluginManager
manager = PluginManager()
manager.register(MyCustomAnalysis())
```

## 📊 Data Loading

### CSV Files (Multi-Channel Support)

Load multi-channel EMG data from CSV files:

```python
from emg_prostudio.utils import DataLoader

# Load CSV with time column (auto-detect sampling rate)
signal = DataLoader.load_csv(
    'emg_data.csv',
    time_column='Time',
    channel_columns=['Biceps', 'Triceps', 'Forearm']
)

# Load CSV without time column (specify sampling rate)
signal = DataLoader.load_csv(
    'emg_data.csv',
    sampling_rate=1000.0  # Hz
)

# Load CSV without header
signal = DataLoader.load_csv(
    'emg_data.csv',
    sampling_rate=1000.0,
    header=None  # No column names
)
```

**Supported CSV formats:**
- Files with/without headers
- Time column for automatic sampling rate calculation
- Multi-channel data in separate columns
- Custom delimiters (comma, tab, etc.)
- Selective channel loading

See `examples/csv_loading_example.py` for complete examples.

### Other Formats

```python
# NumPy
signal = DataLoader.load_numpy('data.npz')

# HDF5
signal = DataLoader.load_hdf5('data.h5')

# JSON
signal = DataLoader.load_json('data.json')
```

## 🔌 Model Interface

The plugin system provides easy integration points for inference models:

```python
from emg_prostudio.plugins import AnalysisPlugin

class MyModelPlugin(AnalysisPlugin):
    """Custom model inference plugin."""
    
    def __init__(self):
        super().__init__("MyModel", "1.0.0")
        # Load your model here
        # self.model = load_model('path/to/model')
    
    def analyze(self, emg_signal, **kwargs):
        """Run inference on EMG signal."""
        # Preprocess if needed
        # features = extract_features(emg_signal)
        # predictions = self.model.predict(features)
        
        return {
            'predictions': [],  # Your model outputs
            'confidence': 0.95,
            # Add any other results
        }
    
    def get_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': 'My custom model for EMG analysis',
            'model_type': 'deep_learning',  # or 'traditional', 'hybrid'
        }

# Register and use
from emg_prostudio.plugins import PluginManager
manager = PluginManager()
manager.register(MyModelPlugin())

results = manager.analyze('MyModel', emg_signal)
```

The `AnalysisPlugin` base class provides a simple interface for integrating any model or analysis method.

## 📝 Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- PyQt5 >= 5.15.0
- PyQtGraph >= 0.13.0
- Matplotlib >= 3.4.0
- Pandas >= 1.3.0
- Optional: PyTorch >= 2.0.0 (for deep learning features)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📊 Loading Multi-Channel CSV Files

EMG_PROSTUDIO has comprehensive support for loading multi-channel EMG data from CSV files:

```python
from emg_prostudio.utils import DataLoader

# Load CSV with automatic sampling rate detection
signal = DataLoader.load_csv(
    'emg_data.csv',
    time_column='Time',  # Auto-calculate sampling rate from time
    channel_columns=['Biceps', 'Triceps', 'Forearm']  # Select specific channels
)

# Load CSV without time column (specify sampling rate)
signal = DataLoader.load_csv(
    'emg_data.csv',
    sampling_rate=1000.0  # Specify sampling rate in Hz
)

# Load CSV without headers
signal = DataLoader.load_csv(
    'emg_data.csv',
    sampling_rate=1000.0,
    header=None  # Auto-generate channel names
)
```

**Features:**
- Automatic sampling rate calculation from time column
- Multi-channel support (load all or specific channels)
- Flexible format handling (with/without headers, custom delimiters)
- Compatible with various CSV formats from EMG hardware

**Example CSV formats supported:**
```csv
# Format 1: With time and named channels
Time,Biceps,Triceps,Forearm
0.000,0.123,0.234,0.345
0.001,0.126,0.237,0.348
...

# Format 2: Without time column
Biceps,Triceps,Forearm
0.123,0.234,0.345
0.126,0.237,0.348
...

# Format 3: No headers (numeric data only)
0.123,0.234,0.345
0.126,0.237,0.348
...
```

See `examples/csv_loading_example.py` for complete working examples.

## 🔌 Model Integration Interface

The plugin system provides a clean interface for integrating custom models:

```python
from emg_prostudio.plugins import AnalysisPlugin

class CustomModelPlugin(AnalysisPlugin):
    """Plugin for your custom EMG analysis model."""
    
    def __init__(self):
        super().__init__("CustomModel", "1.0.0")
        # Initialize your model here
        # self.model = load_your_model()
    
    def analyze(self, emg_signal, **kwargs):
        """Run analysis/inference on EMG signal."""
        # Your analysis code here
        # results = self.model.predict(emg_signal.data)
        
        return {
            'predictions': [],      # Your model outputs
            'confidence': 0.95,     # Confidence scores
            # Add any other results
        }
    
    def get_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': 'Custom model for EMG analysis',
            'model_type': 'deep_learning'  # or 'traditional', 'hybrid'
        }
```

**Using the plugin:**
```python
from emg_prostudio.plugins import PluginManager

# Register your plugin
manager = PluginManager()
manager.register(CustomModelPlugin())

# Run analysis
results = manager.analyze('CustomModel', emg_signal)
```

The interface is designed to be:
- **Simple**: Minimal code needed to integrate
- **Flexible**: Works with any model architecture
- **Extensible**: Easy to add new features

## 👥 Authors

- **PRIMOCOSMOS** - Initial work

## 🙏 Acknowledgments

- EMG signal processing methods based on published literature
- UI design inspired by futuristic/sci-fi aesthetics
- PyQtGraph for efficient real-time plotting

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## ❓ Troubleshooting

### Import Errors

If you get errors like `ModuleNotFoundError: No module named 'numpy'` or `ModuleNotFoundError: No module named 'emg_prostudio'`:

**Solution**: Install the package and its dependencies first:
```bash
cd EMG_PROSTUDIO
pip install -e .
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

### Specific Import Issues

If specific imports fail:
```python
from emg_prostudio.core.signal import EMGSignal  # Requires: numpy, scipy
from emg_prostudio.preprocessing.filters import EMGPreprocessor  # Requires: numpy, scipy
from emg_prostudio.features.extractor import FeatureExtractor  # Requires: numpy, scipy
from emg_prostudio.analysis.action_detector import ActionDetector  # Requires: numpy, scipy
from emg_prostudio.analysis.fatigue_monitor import FatigueMonitor  # Requires: numpy, scipy
from emg_prostudio.plugins import AnalysisPlugin  # No additional deps
```

**Minimal dependencies for core functionality**:
```bash
pip install numpy scipy pandas h5py
```

### GUI Not Starting

If GUI doesn't start, ensure PyQt5 is installed:
```bash
pip install PyQt5 pyqtgraph
```

---

**⚡ EMG PROSTUDIO - Professional EMG Analysis Made Easy ⚡**

