#!/usr/bin/env python3
"""Main entry point for EMG PROSTUDIO application."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emg_prostudio.ui.main_window import main

if __name__ == '__main__':
    main()
