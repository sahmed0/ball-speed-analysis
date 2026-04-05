import os
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QFormLayout, 
    QCheckBox, QMessageBox, QProgressBar, QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from ..core.fitter import PowerLawFitter, FitterConfig
from .mpl_widget import MplWidget

class FitterThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config

    def run(self):
        try:
            fitter = PowerLawFitter(self.config)
            results = fitter.run_analysis(self.df)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class FitterTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.layout = QHBoxLayout(self)

        # Left Panel: Controls
        self.controls = QWidget()
        self.controls.setFixedWidth(300)
        self.form_layout = QFormLayout(self.controls)

        self.csv_path_le = QLineEdit()
        self.browse_btn = QPushButton("Load CSV Data")
        self.browse_btn.clicked.connect(self.on_browse)

        self.max_segments_sb = QSpinBox()
        self.max_segments_sb.setRange(1, 10)
        self.max_segments_sb.setValue(2)

        self.use_robust_cb = QCheckBox("Use Robust Huber Regression")
        self.use_weighted_cb = QCheckBox("Use Weighted Errors (1/σ)")
        self.use_weighted_cb.setChecked(True)
        
        self.ball_size_le = QLineEdit("Ball Size (mm)")

        self.start_btn = QPushButton("Run Power-Law Analysis")
        self.start_btn.clicked.connect(self.on_start)
        self.start_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; height: 40px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()

        self.form_layout.addRow(self.browse_btn)
        self.form_layout.addRow("CSV File:", self.csv_path_le)
        self.form_layout.addRow("Max Segments:", self.max_segments_sb)
        self.form_layout.addRow(self.use_robust_cb)
        self.form_layout.addRow(self.use_weighted_cb)
        self.form_layout.addRow("Ball Label:", self.ball_size_le)
        self.form_layout.addRow(self.start_btn)
        self.form_layout.addRow(self.progress_bar)

        # Right Panel: Plot
        self.plot_widget = MplWidget()

        self.layout.addWidget(self.controls)
        self.layout.addWidget(self.plot_widget)

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV Data", "", "CSV Files (*.csv)")
        if path:
            self.csv_path_le.setText(path)

    def on_start(self):
        path = self.csv_path_le.text()
        if not os.path.isfile(path):
            QMessageBox.critical(self, "Error", "Please select a valid CSV file.")
            return

        try:
            df = pd.read_csv(path)
            # Basic validation
            if "speed" not in df.columns or "lambda" not in df.columns or "pressure" not in df.columns:
                QMessageBox.critical(self, "Error", "CSV must contain 'speed' or 'lambda' and 'pressure' columns.")
                return
            
            config = FitterConfig(
                max_segments=self.max_segments_sb.value(),
                use_robust=self.use_robust_cb.isChecked(),
                use_weighted=self.use_weighted_cb.isChecked(),
                ball_size=self.ball_size_le.text()
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
            return

        self.start_btn.setEnabled(False)
        self.progress_bar.show()
        
        self.thread = FitterThread(df, config)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_finished(self, results):
        self.start_btn.setEnabled(True)
        self.progress_bar.hide()
        self.plot_results(results)

    def on_error(self, message):
        self.start_btn.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Error", f"Analysis failed: {message}")

    def plot_results(self, res):
        self.plot_widget.clear()
        ax = self.plot_widget.ax
        
        # Plot all data
        ax.scatter(res['logV'], res['logP'], c='lightgray', s=20, label="All Data", alpha=0.5)
        
        # Plot pre-processed working data
        ax.scatter(res['x_working'], res['y_working'], c='blue', s=25, label="Analyzed Data")
        
        # Plot outliers
        if len(res['outliers_idx']) > 0:
            ax.scatter(res['logV'][res['outliers_idx']], res['logP'][res['outliers_idx']], 
                       facecolors='none', edgecolors='red', s=50, label="Outliers")

        # Plot fitted segments
        breaks = res['breaks']
        slopes = res['slopes']
        intercepts = res['intercepts']
        metrics = res['segment_metrics']
        
        for i in range(len(breaks) - 1):
            x0, x1 = breaks[i], breaks[i+1]
            xs = np.linspace(x0, x1, 100)
            ys = slopes[i] * xs + intercepts[i]
            
            m = metrics[i]
            label = f"Seg {i+1}: α={slopes[i]:.3f}"
            if not np.isnan(m['chi2_red']):
                label += f" (χ²ν={m['chi2_red']:.1f})"
            else:
                label += f" (R²={m['r2']:.3f})"
                
            ax.plot(xs, ys, lw=3, label=label)
            
            # Annotate
            x_mid = 0.5 * (x0 + x1)
            y_mid = slopes[i] * x_mid + intercepts[i]
            y_label = y_mid * 1.05
            ax.text(x_mid, y_label, f"{slopes[i]:.3f}", fontweight='bold', ha='center', va='bottom')

        ax.set_xlabel("log10(Speed [m/s])")
        ax.set_ylabel("log10(Pressure [Pa])")
        ax.set_title(f"Power-Law Fit: {self.ball_size_le.text()} ({res['optimal_k']} segments)")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        self.plot_widget.canvas.draw()
