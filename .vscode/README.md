# VSCode Development Setup

This directory contains VSCode-specific configuration for developing EMG_PROSTUDIO.

## Quick Start

1. **Open the project in VSCode**
   ```bash
   code /path/to/EMG_PROSTUDIO
   ```

2. **Install dependencies** (one-time setup)
   ```bash
   pip install -r requirements.txt
   ```
   Or for core dependencies only:
   ```bash
   pip install -r requirements-core.txt
   ```

3. **Start developing!** - Imports will now work without `pip install -e .`

## What's Configured

### `settings.json`
- **Python path configuration**: Automatically adds the workspace folder to Python path
- **Testing**: Pytest integration enabled
- **Auto-complete**: Enhanced auto-completion for project modules
- **File exclusions**: Hides `__pycache__`, `.pyc`, and other build artifacts

### `launch.json`
Pre-configured debug configurations:

1. **Python: Current File** - Run/debug the currently open Python file
2. **Python: Main GUI** - Launch the EMG_PROSTUDIO GUI (main.py)
3. **Python: Example Script** - Run the example analysis script
4. **Python: Pytest** - Run all tests with debugging

All configurations automatically set `PYTHONPATH` to include the workspace folder.

## Usage

### Running Code

Simply press `F5` or use the Debug panel to:
- Run the current file
- Launch the GUI application
- Run example scripts
- Execute tests

### Imports

All imports will work directly:
```python
from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.preprocessing.filters import EMGPreprocessor
from emg_prostudio.features.extractor import FeatureExtractor
from emg_prostudio.analysis.action_detector import ActionDetector
from emg_prostudio.analysis.fatigue_monitor import FatigueMonitor
```

No need to run `pip install -e .` for development!

### Testing

- Use the Testing panel in VSCode to discover and run tests
- Or press `F5` and select "Python: Pytest" configuration
- Or run from terminal: `pytest tests/`

## Troubleshooting

### Imports still not working?

1. **Reload VSCode**: Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and run "Developer: Reload Window"
2. **Check Python interpreter**: Make sure the correct Python interpreter is selected (bottom-left of VSCode)
3. **Verify dependencies**: Ensure you've installed the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### IntelliSense not working?

1. Check that Pylance or Python extension is installed
2. Reload the window: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check Python interpreter selection in bottom-left corner

### Tests not discovered?

1. Open Testing panel (beaker icon in left sidebar)
2. Click "Configure Python Tests"
3. Select "pytest"
4. Select "tests" as the directory

## Additional Tips

- **Terminal**: Use VSCode's integrated terminal - it automatically sets up the environment
- **Git**: Built-in Git integration works seamlessly
- **Debugging**: Set breakpoints by clicking left of line numbers, then press `F5`
- **Auto-completion**: Press `Ctrl+Space` to trigger IntelliSense
