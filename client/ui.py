import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel)
temp_ip = "192.168.168.2"
temp_port = 5555

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Skeleton Re-identification")
        self.setGeometry(100,100,850,820)
        
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
        
        # replace with actual camera feed
        temp_feed = QPixmap('temp_feed.jpg');
        feed = QLabel()
        feed.setPixmap(temp_feed)
        feed.setMinimumSize(480, 360)
        feed.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(feed)

        main_title = QLabel(f'Connected to server at: {temp_ip}:{temp_port}')
        main_title.setAlignment(Qt.AlignCenter)
        main_title.setFont(QFont('Arial', 12))
        layout.addLayout(top_layout)
        layout.addLayout(camera_layout)
        layout.addWidget(main_title)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        
        
        


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()