#!/bin/bash
# Installation script for EMG_PROSTUDIO

echo "======================================"
echo "EMG_PROSTUDIO Installation Script"
echo "======================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed"
    echo "Please install Python 3.8 or higher first"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "✓ Using Python: $($PYTHON_CMD --version)"
echo ""

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "❌ Error: pip is not installed"
    echo "Please install pip first"
    exit 1
fi

echo "Select installation type:"
echo "1) Full installation (recommended - includes GUI)"
echo "2) Core only (no GUI, for programmatic use)"
echo "3) Development installation (includes testing tools)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "📦 Installing EMG_PROSTUDIO with all dependencies..."
        $PYTHON_CMD -m pip install -e .
        ;;
    2)
        echo ""
        echo "📦 Installing core dependencies only..."
        $PYTHON_CMD -m pip install -r requirements-core.txt
        ;;
    3)
        echo ""
        echo "📦 Installing EMG_PROSTUDIO with development dependencies..."
        $PYTHON_CMD -m pip install -e ".[dev]"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Testing installation..."
echo "======================================"

# Test import
if $PYTHON_CMD -c "from emg_prostudio import EMGSignal; print('✓ Core imports working')" 2>/dev/null; then
    echo "✓ Installation successful!"
    echo ""
    echo "You can now use EMG_PROSTUDIO:"
    echo "  - Run GUI: python main.py"
    echo "  - Run example: python examples/basic_analysis.py"
    echo "  - Run tests: pytest tests/"
else
    echo "⚠️  Warning: Some imports may not work"
    echo "   Make sure all dependencies are installed:"
    echo "   pip install -r requirements.txt"
fi

echo ""
echo "======================================"
