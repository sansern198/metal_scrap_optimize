import sys
import cv2
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QTabWidget, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QImage


# ---------------- UI ส่วนเดิม ----------------
class MeasurementTab(QWidget):
    def __init__(self):
        super().__init__()
        self.label_camera = QLabel("Camera Preview")
        self.label_camera.setAlignment(Qt.AlignCenter)
        self.label_camera.setMinimumSize(1280, 720)
        self.label_camera.setStyleSheet("background-color: #222; color: white;")

        self.btn_measure = QPushButton("Measure")
        self.btn_measure.setFont(QFont("Arial", 16))

        self.info_bar = QFrame()
        self.info_bar.setFrameShape(QFrame.StyledPanel)

        self.label_area = QLabel("Area: - mm²")
        self.label_width = QLabel("Width: - mm")
        self.label_height = QLabel("Height: - mm")
        for lbl in [self.label_area, self.label_width, self.label_height]:
            lbl.setFont(QFont("Arial", 16))
            lbl.setStyleSheet("padding: 12px;")

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.label_area)
        info_layout.addWidget(self.label_width)
        info_layout.addWidget(self.label_height)
        info_layout.addStretch()
        self.info_bar.setLayout(info_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.label_camera)
        layout.addWidget(self.btn_measure)
        layout.addWidget(self.info_bar)
        self.setLayout(layout)


class DatabaseTab(QWidget):
    def __init__(self):
        super().__init__()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Area (mm²)", "Width (mm)", "Height (mm)"]
        )
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metal Measurement System")
        self.resize(1100, 750)

        self.tabs = QTabWidget()
        self.measurement_tab = MeasurementTab()
        self.database_tab = DatabaseTab()

        self.tabs.addTab(self.measurement_tab, "Measurement Screen")
        self.tabs.addTab(self.database_tab, "Database")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        # NOTE: จะเชื่อมใหม่ใน App() ด้านล่าง
        self.measurement_tab.btn_measure.clicked.connect(self.add_dummy_data)

    def add_dummy_data(self):
        """dummy เดิม"""
        table = self.database_tab.table
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem("25000.0"))
        table.setItem(row, 1, QTableWidgetItem("120.0"))
        table.setItem(row, 2, QTableWidgetItem("200.0"))
        self.measurement_tab.label_area.setText("Area: 25000.0 mm²")
        self.measurement_tab.label_width.setText("Width: 120.0 mm")
        self.measurement_tab.label_height.setText("Height: 200.0 mm")
        self.measurement_tab.label_camera.setPixmap(QPixmap(1280, 720))


# ---------------- Worker Thread ----------------
class MeasurementWorker(QThread):
    result_ready = pyqtSignal(QPixmap, float, float, float)

    def run(self):
        # เปิดกล้อง (ตัวอย่างใช้ webcam ปกติ)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return

        # ----- ตัวอย่างประมวลผลง่าย ๆ -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area, width, height = 0.0, 0.0, 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            width, height = w, h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # แปลงภาพไป QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(1280, 720, Qt.KeepAspectRatio)

        self.result_ready.emit(pixmap, area, width, height)


# ---------------- เชื่อม UI กับ Worker ----------------
class App(MainWindow):
    def __init__(self):
        super().__init__()
        self.worker = MeasurementWorker()

        # ตัด dummy ออก → เชื่อมใหม่
        self.measurement_tab.btn_measure.clicked.disconnect()
        self.measurement_tab.btn_measure.clicked.connect(self.start_measure)

        self.worker.result_ready.connect(self.update_ui)

    def start_measure(self):
        self.worker.start()

    def update_ui(self, pixmap, area, width, height):
        # update preview
        self.measurement_tab.label_camera.setPixmap(pixmap)

        # update info bar
        self.measurement_tab.label_area.setText(f"Area: {round(area,1)} mm²")
        self.measurement_tab.label_width.setText(f"Width: {round(width,1)} mm")
        self.measurement_tab.label_height.setText(f"Height: {round(height,1)} mm")

        # update database
        table = self.database_tab.table
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(str(round(area,1))))
        table.setItem(row, 1, QTableWidgetItem(str(round(width,1))))
        table.setItem(row, 2, QTableWidgetItem(str(round(height,1))))


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())
