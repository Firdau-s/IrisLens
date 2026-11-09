import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QPainter

from main import MainWindow

def capture_screens():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    win = MainWindow()
    win.resize(1920, 1080)
    win.show()
    
    # Process events to lay out properly
    app.processEvents()
    
    # 1. Halaman 1: Splash Screen
    win.stacked_widget.setCurrentIndex(0)
    app.processEvents()
    pix0 = win.grab()
    pix0.save("rendered_screen_1.png")
    print("Saved rendered_screen_1.png")
    
    # 2. Halaman 2: Registration Screen
    win.stacked_widget.setCurrentIndex(1)
    app.processEvents()
    pix1 = win.grab()
    pix1.save("rendered_screen_2.png")
    print("Saved rendered_screen_2.png")
    
    # 3. Halaman 3: Examination Screen
    win.stacked_widget.setCurrentIndex(2)
    win.exam_screen.update_video_frame()
    app.processEvents()
    pix2 = win.grab()
    pix2.save("rendered_screen_3.png")
    print("Saved rendered_screen_3.png")
    
    win.close()
    app.quit()

if __name__ == "__main__":
    capture_screens()

