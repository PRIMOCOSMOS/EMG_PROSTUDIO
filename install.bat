@echo off
REM Installation script for EMG_PROSTUDIO (Windows)

echo ======================================
echo EMG_PROSTUDIO Installation Script
echo ======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher first
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if pip is available
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: pip is not installed
    echo Please install pip first
    pause
    exit /b 1
)

echo Select installation type:
echo 1) Full installation (recommended - includes GUI)
echo 2) Core only (no GUI, for programmatic use)
echo 3) Development installation (includes testing tools)
echo.
set /p choice="Enter choice [1-3]: "

if "%choice%"=="1" (
    echo.
    echo Installing EMG_PROSTUDIO with all dependencies...
    python -m pip install -e .
) else if "%choice%"=="2" (
    echo.
    echo Installing core dependencies only...
    python -m pip install -r requirements-core.txt
) else if "%choice%"=="3" (
    echo.
    echo Installing EMG_PROSTUDIO with development dependencies...
    python -m pip install -e .[dev]
) else (
    echo Invalid choice
    pause
    exit /b 1
)

echo.
echo ======================================
echo Testing installation...
echo ======================================

REM Test import
python -c "from emg_prostudio import EMGSignal; print('OK: Core imports working')" 2>nul
if %errorlevel% equ 0 (
    echo OK: Installation successful!
    echo.
    echo You can now use EMG_PROSTUDIO:
    echo   - Run GUI: python main.py
    echo   - Run example: python examples\basic_analysis.py
    echo   - Run tests: pytest tests\
) else (
    echo Warning: Some imports may not work
    echo Make sure all dependencies are installed:
    echo pip install -r requirements.txt
)

echo.
echo ======================================
pause
