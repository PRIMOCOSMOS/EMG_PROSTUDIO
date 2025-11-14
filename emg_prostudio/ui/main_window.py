"""Main GUI application with futuristic/sci-fi theme.

This module provides a modern, visually appealing interface for EMG analysis
with real-time visualization and interactive controls.
"""

import sys
import numpy as np
from typing import Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QGroupBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QMessageBox, QProgressBar, QTextEdit,
    QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont
import pyqtgraph as pg

from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.preprocessing.filters import EMGPreprocessor
from emg_prostudio.features.extractor import FeatureExtractor
from emg_prostudio.analysis.action_detector import ActionDetector
from emg_prostudio.analysis.fatigue_monitor import FatigueMonitor
from emg_prostudio.utils.sample_data import generate_sample_emg
from emg_prostudio.utils.data_io import DataLoader, DataSaver
from emg_prostudio.plugins import PluginManager
from emg_prostudio.plugins.traditional import TraditionalSignalProcessing


class EMGProStudioGUI(QMainWindow):
    """Main application window with futuristic theme."""
    
    def __init__(self):
        """Initialize the GUI."""
        super().__init__()
        
        self.emg_signal: Optional[EMGSignal] = None
        self.preprocessed_signal: Optional[EMGSignal] = None
        self.analysis_results: Optional[dict] = None
        
        # Initialize plugin system
        self.plugin_manager = PluginManager()
        self.plugin_manager.register(TraditionalSignalProcessing())
        
        self.init_ui()
        self.apply_futuristic_theme()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("EMG PROSTUDIO - Professional EMG Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("⚡ EMG PROSTUDIO ⚡")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 24, QFont.Bold)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Create tabs for different views
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_signal_view(), "📊 Signal View")
        self.tabs.addTab(self.create_analysis_view(), "🔬 Analysis")
        self.tabs.addTab(self.create_fatigue_view(), "💪 Fatigue Monitor")
        self.tabs.addTab(self.create_results_view(), "📈 Results")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready | Load data or generate sample to begin")
    
    def create_control_panel(self) -> QWidget:
        """Create control panel with buttons."""
        panel = QGroupBox("Control Panel")
        layout = QHBoxLayout()
        
        # Load data button
        btn_load = QPushButton("📁 Load Data")
        btn_load.clicked.connect(self.load_data)
        layout.addWidget(btn_load)
        
        # Generate sample button
        btn_sample = QPushButton("🎲 Generate Sample")
        btn_sample.clicked.connect(self.generate_sample)
        layout.addWidget(btn_sample)
        
        # Preprocess button
        btn_preprocess = QPushButton("⚙️ Preprocess")
        btn_preprocess.clicked.connect(self.preprocess_signal)
        layout.addWidget(btn_preprocess)
        
        # Analyze button
        btn_analyze = QPushButton("🔍 Analyze")
        btn_analyze.clicked.connect(self.analyze_signal)
        layout.addWidget(btn_analyze)
        
        # Save results button
        btn_save = QPushButton("💾 Save Results")
        btn_save.clicked.connect(self.save_results)
        layout.addWidget(btn_save)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_signal_view(self) -> QWidget:
        """Create signal visualization view."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Signal plot
        self.signal_plot = pg.PlotWidget(title="EMG Signal")
        self.signal_plot.setLabel('left', 'Amplitude', units='mV')
        self.signal_plot.setLabel('bottom', 'Time', units='s')
        self.signal_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.signal_plot)
        
        # Preprocessed signal plot
        self.preprocessed_plot = pg.PlotWidget(title="Preprocessed Signal")
        self.preprocessed_plot.setLabel('left', 'Amplitude', units='mV')
        self.preprocessed_plot.setLabel('bottom', 'Time', units='s')
        self.preprocessed_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.preprocessed_plot)
        
        widget.setLayout(layout)
        return widget
    
    def create_analysis_view(self) -> QWidget:
        """Create analysis view for action detection."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Action detection plot
        self.action_plot = pg.PlotWidget(title="Action Detection & Classification")
        self.action_plot.setLabel('left', 'Amplitude', units='mV')
        self.action_plot.setLabel('bottom', 'Time', units='s')
        self.action_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.action_plot)
        
        # Results summary
        self.action_summary = QTextEdit()
        self.action_summary.setMaximumHeight(150)
        self.action_summary.setReadOnly(True)
        layout.addWidget(self.action_summary)
        
        widget.setLayout(layout)
        return widget
    
    def create_fatigue_view(self) -> QWidget:
        """Create fatigue monitoring view."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Fatigue trajectory plot
        self.fatigue_plot = pg.PlotWidget(title="Fatigue Progression")
        self.fatigue_plot.setLabel('left', 'Feature Value')
        self.fatigue_plot.setLabel('bottom', 'Time', units='s')
        self.fatigue_plot.addLegend()
        self.fatigue_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.fatigue_plot)
        
        # Fatigue index plot
        self.fatigue_index_plot = pg.PlotWidget(title="Fatigue Index (0=Fresh, 1=Exhausted)")
        self.fatigue_index_plot.setLabel('left', 'Fatigue Index')
        self.fatigue_index_plot.setLabel('bottom', 'Time', units='s')
        self.fatigue_index_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.fatigue_index_plot)
        
        widget.setLayout(layout)
        return widget
    
    def create_results_view(self) -> QWidget:
        """Create results summary view."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        widget.setLayout(layout)
        return widget
    
    def apply_futuristic_theme(self):
        """Apply futuristic/sci-fi theme to the application."""
        # Dark theme with cyan/blue accents
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0a0e27;
                color: #00ff9f;
            }
            
            QGroupBox {
                border: 2px solid #00d4ff;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00d4ff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QPushButton {
                background-color: #1a2332;
                color: #00ff9f;
                border: 2px solid #00d4ff;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            
            QPushButton:hover {
                background-color: #2a3342;
                border-color: #00ff9f;
            }
            
            QPushButton:pressed {
                background-color: #00d4ff;
                color: #0a0e27;
            }
            
            QLabel {
                color: #00ff9f;
            }
            
            QTabWidget::pane {
                border: 2px solid #00d4ff;
                border-radius: 5px;
            }
            
            QTabBar::tab {
                background-color: #1a2332;
                color: #00ff9f;
                border: 2px solid #00d4ff;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            
            QTabBar::tab:selected {
                background-color: #0a0e27;
                border-bottom: 2px solid #0a0e27;
            }
            
            QTabBar::tab:hover {
                background-color: #2a3342;
            }
            
            QTextEdit {
                background-color: #0f1420;
                color: #00ff9f;
                border: 1px solid #00d4ff;
                border-radius: 5px;
                padding: 5px;
                font-family: 'Courier New';
            }
            
            QStatusBar {
                background-color: #1a2332;
                color: #00d4ff;
                border-top: 2px solid #00d4ff;
            }
        """)
        
        # Configure PyQtGraph to match theme
        pg.setConfigOption('background', '#0a0e27')
        pg.setConfigOption('foreground', '#00ff9f')
    
    def load_data(self):
        """Load EMG data from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load EMG Data",
            "",
            "All Files (*.npy *.npz *.h5 *.hdf5 *.json);;NumPy Files (*.npy *.npz);;HDF5 Files (*.h5 *.hdf5);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                loader = DataLoader()
                if file_path.endswith('.npy') or file_path.endswith('.npz'):
                    self.emg_signal = loader.load_numpy(file_path)
                elif file_path.endswith(('.h5', '.hdf5')):
                    self.emg_signal = loader.load_hdf5(file_path)
                elif file_path.endswith('.json'):
                    self.emg_signal = loader.load_json(file_path)
                
                self.plot_signal()
                self.statusBar().showMessage(f"Loaded: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
    
    def generate_sample(self):
        """Generate sample EMG data."""
        self.emg_signal = generate_sample_emg(
            duration=30.0,
            sampling_rate=1000.0,
            n_repetitions=10,
            n_channels=2,
            fatigue_effect=True,
            movement_quality_variation=True
        )
        
        self.plot_signal()
        self.statusBar().showMessage("Generated sample EMG data (30s, 10 reps)")
    
    def preprocess_signal(self):
        """Preprocess the EMG signal."""
        if self.emg_signal is None:
            QMessageBox.warning(self, "Warning", "Please load or generate data first!")
            return
        
        try:
            preprocessor = EMGPreprocessor(self.emg_signal.sampling_rate)
            self.preprocessed_signal = preprocessor.preprocess_pipeline(
                self.emg_signal,
                remove_powerline=True,
                powerline_freq=50.0,
                bandpass=True,
                lowcut=20.0,
                highcut=450.0
            )
            
            self.plot_preprocessed()
            self.statusBar().showMessage("Preprocessing complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preprocessing failed:\n{str(e)}")
    
    def analyze_signal(self):
        """Perform comprehensive analysis."""
        if self.preprocessed_signal is None:
            QMessageBox.warning(self, "Warning", "Please preprocess the signal first!")
            return
        
        try:
            # Use traditional signal processing plugin
            self.analysis_results = self.plugin_manager.analyze(
                'TraditionalSignalProcessing',
                self.preprocessed_signal
            )
            
            self.plot_analysis()
            self.plot_fatigue()
            self.display_results()
            self.statusBar().showMessage("Analysis complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{str(e)}")
    
    def plot_signal(self):
        """Plot raw EMG signal."""
        if self.emg_signal is None:
            return
        
        self.signal_plot.clear()
        time = self.emg_signal.time_vector
        
        colors = ['#00ff9f', '#00d4ff', '#ff00ff', '#ffff00']
        for i in range(self.emg_signal.n_channels):
            channel_data = self.emg_signal.get_channel(i)
            self.signal_plot.plot(
                time, channel_data,
                pen=pg.mkPen(color=colors[i % len(colors)], width=2),
                name=self.emg_signal.channels[i]
            )
    
    def plot_preprocessed(self):
        """Plot preprocessed signal."""
        if self.preprocessed_signal is None:
            return
        
        self.preprocessed_plot.clear()
        time = self.preprocessed_signal.time_vector
        
        colors = ['#00ff9f', '#00d4ff']
        for i in range(self.preprocessed_signal.n_channels):
            channel_data = self.preprocessed_signal.get_channel(i)
            self.preprocessed_plot.plot(
                time, channel_data,
                pen=pg.mkPen(color=colors[i % len(colors)], width=2),
                name=self.preprocessed_signal.channels[i]
            )
    
    def plot_analysis(self):
        """Plot action detection results."""
        if self.analysis_results is None:
            return
        
        self.action_plot.clear()
        
        # Plot signal
        signal = self.analysis_results['preprocessed_signal']
        time = signal.time_vector
        data = signal.get_channel(0)
        self.action_plot.plot(time, data, pen=pg.mkPen(color='#00ff9f', width=1.5))
        
        # Mark detected actions
        actions = self.analysis_results['actions']
        for action in actions:
            amp_class = action.get('amplitude_class', 'unknown')
            color_map = {
                'full': '#00ff00',      # Green
                'half': '#ffaa00',      # Orange
                'invalid': '#ff0000'    # Red
            }
            color = color_map.get(amp_class, '#ffffff')
            
            # Draw region
            region = pg.LinearRegionItem(
                [action['start_time'], action['end_time']],
                brush=pg.mkBrush(color=color, alpha=50),
                movable=False
            )
            self.action_plot.addItem(region)
        
        # Display summary
        counts = self.analysis_results['action_counts']
        summary = f"""
        ═══════════════════════════════════════
        ACTION DETECTION SUMMARY
        ═══════════════════════════════════════
        Total Actions:    {counts['total']}
        Full Range:       {counts['full']} ✓
        Half Range:       {counts['half']} ⚠
        Invalid:          {counts['invalid']} ✗
        Valid Total:      {counts['valid']}
        
        Quality Rate:     {counts['valid']/counts['total']*100:.1f}%
        ═══════════════════════════════════════
        """
        self.action_summary.setPlainText(summary)
    
    def plot_fatigue(self):
        """Plot fatigue monitoring results."""
        if self.analysis_results is None:
            return
        
        trajectory = self.analysis_results['fatigue_trajectory']
        
        # Plot fatigue indicators
        self.fatigue_plot.clear()
        time = trajectory['time']
        
        # MDF (decreases with fatigue)
        self.fatigue_plot.plot(
            time, trajectory['mdf'],
            pen=pg.mkPen(color='#00ff9f', width=2),
            name='Median Frequency'
        )
        
        # MPF
        self.fatigue_plot.plot(
            time, trajectory['mpf'],
            pen=pg.mkPen(color='#00d4ff', width=2),
            name='Mean Power Frequency'
        )
        
        # Plot fatigue index
        self.fatigue_index_plot.clear()
        fatigue_index = self.analysis_results['fatigue_analysis'].get('fatigue_index', 
                        np.zeros_like(time))
        
        # Create filled plot
        self.fatigue_index_plot.plot(
            time, fatigue_index,
            pen=pg.mkPen(color='#ff00ff', width=3),
            fillLevel=0,
            brush=pg.mkBrush(color=(255, 0, 255, 100))
        )
        
        # Add horizontal line at 0.5 (moderate fatigue)
        self.fatigue_index_plot.addLine(y=0.5, pen=pg.mkPen(color='#ffaa00', 
                                        width=2, style=Qt.DashLine))
    
    def display_results(self):
        """Display comprehensive results."""
        if self.analysis_results is None:
            return
        
        fatigue_analysis = self.analysis_results['fatigue_analysis']
        action_counts = self.analysis_results['action_counts']
        
        results_text = f"""
╔════════════════════════════════════════════════════════════╗
║               EMG ANALYSIS RESULTS                          ║
╚════════════════════════════════════════════════════════════╝

▶ ACTION ANALYSIS
  ────────────────────────────────────────────────────────────
  Total Repetitions:        {action_counts['total']}
  Full Range Movements:     {action_counts['full']} ({action_counts['full']/action_counts['total']*100:.1f}%)
  Half Range Movements:     {action_counts['half']} ({action_counts['half']/action_counts['total']*100:.1f}%)
  Invalid Movements:        {action_counts['invalid']} ({action_counts['invalid']/action_counts['total']*100:.1f}%)
  
  Overall Quality Score:    {action_counts['valid']/action_counts['total']*100:.1f}%

▶ FATIGUE ANALYSIS
  ────────────────────────────────────────────────────────────
  Exercise Duration:        {fatigue_analysis.get('duration', 0):.1f} seconds
  
  MDF Trend:
    • Decline Rate:         {fatigue_analysis['mdf_trend']['decline_rate']:.2f}% per second
    • R²:                   {fatigue_analysis['mdf_trend']['r_squared']:.3f}
  
  MPF Trend:
    • Decline Rate:         {fatigue_analysis['mpf_trend']['decline_rate']:.2f}% per second
    • R²:                   {fatigue_analysis['mpf_trend']['r_squared']:.3f}
  
  RMS Trend (Compensation):
    • Increase Rate:        {fatigue_analysis['rms_trend']['increase_rate']:.2f}% per second
    • R²:                   {fatigue_analysis['rms_trend']['r_squared']:.3f}
  
  Fatigue Indicators:
    • Max Fatigue Index:    {fatigue_analysis.get('max_fatigue_index', 0):.2f}
    • Final Fatigue Index:  {fatigue_analysis.get('final_fatigue_index', 0):.2f}

▶ INTERPRETATION
  ────────────────────────────────────────────────────────────
  Movement Quality:         {"Excellent" if action_counts['valid']/action_counts['total'] > 0.8 else "Good" if action_counts['valid']/action_counts['total'] > 0.6 else "Needs Improvement"}
  Fatigue Level:            {"Minimal" if fatigue_analysis.get('final_fatigue_index', 0) < 0.3 else "Moderate" if fatigue_analysis.get('final_fatigue_index', 0) < 0.7 else "High"}
  
╔════════════════════════════════════════════════════════════╗
║  RECOMMENDATION: Monitor movement quality as fatigue       ║
║  increases. Consider rest when quality score drops below   ║
║  60% or fatigue index exceeds 0.7.                        ║
╚════════════════════════════════════════════════════════════╝
        """
        
        self.results_text.setPlainText(results_text)
    
    def save_results(self):
        """Save analysis results."""
        if self.analysis_results is None:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.results_text.toPlainText())
                self.statusBar().showMessage(f"Results saved: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = EMGProStudioGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
