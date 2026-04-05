from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from .tracking_tab import TrackingTab
from .fitter_tab import FitterTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Elastohydrodynamics Motion Analyser")
        self.resize(1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.tracking_tab = TrackingTab()
        self.fitter_tab = FitterTab()

        self.tabs.addTab(self.tracking_tab, "Motion Analysis")
        self.tabs.addTab(self.fitter_tab, "Power Law Analysis")

        self.layout.addWidget(self.tabs)
