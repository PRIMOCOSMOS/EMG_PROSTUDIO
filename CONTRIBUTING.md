# Contributing to EMG_PROSTUDIO

Thank you for your interest in contributing to EMG_PROSTUDIO! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/EMG_PROSTUDIO.git
cd EMG_PROSTUDIO
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

4. **Install in development mode**
```bash
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=emg_prostudio --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v
```

## How to Contribute

### Reporting Bugs

Please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant code snippets or error messages

### Suggesting Enhancements

Open an issue describing:
- The enhancement or feature
- Use case and benefits
- Possible implementation approach
- Any alternatives considered

### Pull Requests

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
- Follow the code style guide
- Add tests for new functionality
- Update documentation as needed

3. **Run tests**
```bash
pytest
```

4. **Commit your changes**
```bash
git add .
git commit -m "Description of changes"
```

5. **Push and create PR**
```bash
git push origin feature/your-feature-name
```

## Code Style Guide

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Include parameter types and descriptions
- Provide usage examples for complex functions

### Example

```python
def process_emg(
    signal: EMGSignal,
    filter_type: str = 'bandpass',
    **kwargs
) -> EMGSignal:
    """
    Process EMG signal with specified filter.
    
    Args:
        signal: Input EMG signal to process
        filter_type: Type of filter to apply ('bandpass', 'highpass', 'notch')
        **kwargs: Additional filter parameters
        
    Returns:
        Processed EMG signal
        
    Raises:
        ValueError: If filter_type is not recognized
        
    Example:
        >>> signal = EMGSignal(data, sampling_rate=1000)
        >>> processed = process_emg(signal, filter_type='bandpass')
    """
    # Implementation
    pass
```

## Project Structure

```
EMG_PROSTUDIO/
├── emg_prostudio/          # Main package
│   ├── core/               # Core functionality
│   ├── preprocessing/      # Signal preprocessing
│   ├── features/           # Feature extraction
│   ├── analysis/           # Analysis algorithms
│   ├── plugins/            # Plugin system
│   ├── ui/                 # User interface
│   └── utils/              # Utilities
├── tests/                  # Unit tests
├── examples/               # Example scripts
└── docs/                   # Documentation
```

## Adding New Features

### 1. Adding a New Analysis Method

Create a plugin by inheriting from `AnalysisPlugin`:

```python
from emg_prostudio.plugins import AnalysisPlugin
from emg_prostudio.core.signal import EMGSignal

class MyAnalysisMethod(AnalysisPlugin):
    def __init__(self):
        super().__init__("MyMethod", "1.0.0")
    
    def analyze(self, emg_signal: EMGSignal, **kwargs):
        # Your analysis implementation
        results = {
            'feature1': value1,
            'feature2': value2
        }
        return results
    
    def get_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'description': 'My custom analysis method'
        }
```

### 2. Adding a New Feature Extractor

Add methods to existing feature classes or create new ones:

```python
from emg_prostudio.features.activation import ActivationFeatures

class MyFeatures:
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
    
    def my_custom_feature(self, data: np.ndarray) -> float:
        """
        Compute custom feature.
        
        Args:
            data: EMG signal data
            
        Returns:
            Feature value
        """
        # Your implementation
        return feature_value
```

### 3. Adding Tests

Create test files in `tests/` directory:

```python
import pytest
import numpy as np
from emg_prostudio import EMGSignal

def test_my_feature():
    """Test my custom feature."""
    data = np.random.randn(1000, 1)
    signal = EMGSignal(data, sampling_rate=1000.0)
    
    # Test your feature
    result = my_feature(signal)
    
    # Assertions
    assert result > 0
    assert isinstance(result, float)
```

## Documentation Guidelines

### Code Comments

- Comment complex algorithms
- Explain non-obvious logic
- Reference papers/sources when applicable

### README Updates

When adding features, update:
- Feature list
- Installation instructions (if new dependencies)
- Usage examples
- API documentation

### Changelog

Document changes in your PR description:
- New features
- Bug fixes
- Breaking changes
- Deprecations

## Review Process

1. **Automated Checks**: Tests must pass
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address review comments
4. **Merge**: Once approved, PR will be merged

## Questions?

- Open an issue for questions
- Check existing issues and documentation
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions make EMG_PROSTUDIO better for everyone. We appreciate your time and effort!
