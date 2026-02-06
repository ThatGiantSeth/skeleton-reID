import sys
import numpy as np
import cv2
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel)

class CameraFeed(QLabel):
    
    def __init__(self):
        super().__init__()
    
    def update_frame(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaledToWidth(self.width(), Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

class MainWindow(QMainWindow):
    camera_frame = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Skeleton Re-identification")
        self.setFixedSize(850,820)
        
        self.camera_frame.connect(self.update_camera);
        
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        camera_layout = QVBoxLayout()
        
        results_layout = QVBoxLayout();
        results_title = QLabel('Results')
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setFont(QFont('Arial', 14))
        results_layout.addWidget(results_title)
        
        stats_layout = QVBoxLayout();
        stats_title = QLabel('Performance')
        stats_title.setAlignment(Qt.AlignCenter)
        stats_title.setFont(QFont('Arial', 14))
        stats_layout.addWidget(stats_title)
        
        top_layout.addLayout(results_layout)
        top_layout.addLayout(stats_layout)

        feed_title = QLabel('Feed')
        feed_title.setAlignment(Qt.AlignLeft)
        feed_title.setFont(QFont('Arial', 14))
        camera_layout.addWidget(feed_title)
        
        self.feed = CameraFeed()
        self.feed.setMinimumSize(480, 360)
        self.feed.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.feed)

        self.connection_stat = QLabel('No server connection.')
        self.connection_stat.setAlignment(Qt.AlignCenter)
        self.connection_stat.setFont(QFont('Arial', 12))
        layout.addLayout(top_layout)
        layout.addLayout(camera_layout)
        layout.addWidget(self.connection_stat)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
    def update_camera(self, frame: np.ndarray):
        self.feed.update_frame(frame)
    
    def update_ui(self, frame: np.ndarray):
        self.camera_frame.emit(frame)
        
    def update_connection_info(self, ip = None, port: int = None):
        if ip is None:
            self.connection_stat.setText('No server connection.')
        else:
            self.connection_stat.setText(f'Connected to server at: {ip}:{port}')
        
def runUI():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
    
    
if __name__ == "__main__":
    runUI()