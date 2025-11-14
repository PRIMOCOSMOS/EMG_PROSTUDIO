# Quick Start Guide

## Installation

**⚠️ IMPORTANT**: You must install dependencies BEFORE you can import the package!

### Quick Installation (Recommended)

#### Using the Installation Script

**Linux/Mac:**
```bash
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO
./install.sh
```

**Windows:**
```cmd
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO
install.bat
```

#### Manual Installation

```bash
# 1. Clone the repository
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO

# 2. Install the package (this installs all dependencies automatically)
pip install -e .
```

### Alternative: Manual Dependency Installation

If you prefer to install dependencies separately:

```bash
# Clone the repository
git clone https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO

# For core functionality only (no GUI):
pip install -r requirements-core.txt

# For full functionality including GUI:
pip install -r requirements.txt
```

### Verify Installation

After installation, verify imports work:
```bash
python -c "from emg_prostudio import EMGSignal; print('✓ Installation successful!')"
```

## First Steps

### 1. Run the GUI Application

```bash
python main.py
```

This launches the futuristic EMG_PROSTUDIO interface where you can:
- Load EMG data files
- Generate sample data for testing
- Preprocess signals
- Analyze muscle activation and fatigue
- View real-time visualizations

### 2. Try the Command-Line Example

```bash
python examples/basic_analysis.py
```

This runs a complete analysis workflow and displays results in the terminal.

### 3. Use Programmatically

```python
from emg_prostudio.utils import generate_sample_emg
from emg_prostudio import EMGPreprocessor
from emg_prostudio.analysis import ActionDetector

# Generate test data
signal = generate_sample_emg(duration=30, n_repetitions=10)

# Preprocess
preprocessor = EMGPreprocessor(signal.sampling_rate)
clean_signal = preprocessor.preprocess_pipeline(signal)

# Detect actions
detector = ActionDetector(signal.sampling_rate)
actions = detector.detect_actions(clean_signal)
actions = detector.classify_amplitude(actions)

print(f"Detected {len(actions)} actions")
```

## GUI Features

### Signal View Tab
- **Top plot**: Raw EMG signal
- **Bottom plot**: Preprocessed signal after filtering
- Real-time visualization with zoom and pan

### Analysis Tab
- Action detection with color-coded regions:
  - 🟢 Green = Full range movement
  - 🟠 Orange = Half range movement  
  - 🔴 Red = Invalid movement
- Summary statistics

### Fatigue Monitor Tab
- MDF/MPF frequency trends
- Fatigue index progression (0-1 scale)
- Trend analysis

### Results Tab
- Comprehensive text report
- Movement quality assessment
- Fatigue analysis
- Recommendations

## Key Features

### 🎨 Futuristic UI Theme
- Dark background (#0a0e27)
- Cyan/blue accents (#00d4ff)
- Green highlights (#00ff9f)
- Sci-fi inspired design

### 📊 Analysis Capabilities
1. **Preprocessing**: Remove noise, artifacts, baseline drift
2. **Action Recognition**: Automatic counting and classification
3. **Muscle Activation**: RMS, MAV, iEMG features
4. **Fatigue Monitoring**: MDF, MPF tracking with trend analysis
5. **Quality Assessment**: Movement amplitude evaluation

### 🔌 Extensibility
- Plugin architecture for new methods
- Support for traditional signal processing
- Lightweight deep learning integration
- Easy to add custom analysis modules

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [examples/](examples/) directory
- Check out the [tests/](tests/) for usage patterns
- Try loading your own EMG data files

## Troubleshooting

**ImportError**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

**GUI doesn't start**: Ensure PyQt5 is properly installed
```bash
pip install PyQt5 pyqtgraph
```

**No module named 'emg_prostudio'**: Install the package
```bash
pip install -e .
```

## Support

For issues or questions, please open an issue on GitHub.
