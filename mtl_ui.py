# -*- coding: utf-8 -*-
"""
mtl_ui.py
- ส่วน UI (PyQt5), โหลดภาพ/กล้อง, logging, บันทึกผล
- เรียกใช้ mtl_processing.Processor
"""

import cv2
import sys, os, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Union, Optional

import mtl_processing as proc  # โมดูลประมวลผล

# ---------------------- Paths & Config ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
running_number_path = os.path.join(script_dir, 'running_number.txt')
calibration_data_path = os.path.join(script_dir, 'calibration_data/calibration_data_7x5_3040p.npz')
img_output_path = os.path.join(script_dir, 'output'); os.makedirs(img_output_path, exist_ok=True)
log_path = os.path.join(script_dir, 'app.log')

PIPELINE = (
    "souphttpsrc location=http://172.16.165.250:4747/video ! "
    "multipartdemux ! jpegdec ! videoconvert ! appsink"
)

TARGET_W, TARGET_H = 4032, 3040
def resize_to_target(img, w=TARGET_W, h=TARGET_H): return proc.resize_to_target(img, w, h)

USE_IMAGE = True
IMAGE_SOURCE = os.path.join(script_dir, "img/imgmtl2.jpg")

blue_Lower = (90, 120, 120)
blue_Upper = (150, 255, 255)
ROTATE_ANGLE_DEG = 2.4
CROP_SLICE = (slice(500, 2590), slice(50, 4000))
PIXEL_MM_RATIO_W= 0.545
PIXEL_MM_RATIO_H = 0.5476

# ---------------------- Logging ----------------------
logger = logging.getLogger('MTL_MEASUREMENT')
logger.setLevel(logging.DEBUG)

logger.propagate = False

for h in list(logger.handlers):
    logger.removeHandler(h)

handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, mode='a')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(console_handler)

import re

def _s_index(name: str) -> int:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 0

def format_S_for_log(s_list):
    if not s_list:
        return ""
    s_sorted = sorted(s_list, key=lambda d: _s_index(d.get("name", "")))
    parts = [
        f"{d.get('name','S')}(W={float(d.get('W_mm',0)):.1f},"
        f"H={float(d.get('H_mm',0)):.1f},"
        f"A={round(float(d.get('A_mm2',0)))})"
        for d in s_sorted
    ]
    return "[" + " | ".join(parts) + "]"

# --- API integration (UI -> FastAPI) ---
import requests, json
API_URL = os.getenv("API_URL", "http://localhost:8000")

def post_result_to_api(
    output_img_path: str,
    run_no: Union[int, str],
    width_mm: float,
    height_mm: float,
    area_mm2: float,
    s_map_or_list=None,
    source_filename: str = "",
    output_filename: str = "",
    extras: Optional[dict] = None,
    timeout_s: float = 15.0,
    logger=None,
):
    fname = os.path.basename(output_img_path)
    mime = "image/jpeg" if fname.lower().endswith((".jpg", ".jpeg")) else "image/png"

    width_mm  = round(float(width_mm), 2)
    height_mm = round(float(height_mm), 2)
    area_mm2  = float(area_mm2)  # จะปัดก็ได้ถ้าต้องการ round(area_mm2, 2)

    data = {
        "run_no": str(run_no),
        "width_mm": f"{width_mm:.2f}",
        "height_mm": f"{height_mm:.2f}",
        "area_mm2": str(area_mm2),
        "source_filename": source_filename or "",
        "output_filename": output_filename or fname,
        "s_json": json.dumps(s_map_or_list or []),
        "meta_json": json.dumps({}),
    }
    if extras:
        data.update(extras)  # เช่น type_name, class_color

    url = f"{API_URL}/measurements/ingest"
    with open(output_img_path, "rb") as fp:
        files = {"file": (fname, fp, mime)}
        r = requests.post(url, data=data, files=files, timeout=timeout_s)

    if r.status_code >= 400:
        raise RuntimeError(f"API ingest error {r.status_code}: {r.text}")

    if logger:
        logger.info(f"[API] ingest ok -> {r.json()}")

    return r.json()

def get_next_running_number(filename=running_number_path):
    try:
        with open(filename, "r") as f:
            running_number = int(f.read().strip()) + 1
    except Exception:
        running_number = 1
    if running_number > 9999: running_number = 1
    try:
        with open(filename, "w") as f:
            f.write(str(running_number))
    except Exception as e:
        logger.error(f'Failed to write running number: {e}')
    return format(running_number, '04d')

def bgr_to_qimage(bgr) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888).copy()

BASE_W, BASE_H = 1280, 720
def clamp(v, lo, hi): return hi if v > hi else lo if v < lo else v
def pt(base, s): return max(7, int(round(base * s)))
def monospace(size=12, bold=False):
    f = QtGui.QFont("Courier New"); f.setPointSize(size); f.setBold(bold); return f
def sans(size=12, bold=False):
    f = QtGui.QFont("Montserrat"); f.setPointSize(size); f.setBold(bold); return f

# ---------------------- Widgets ----------------------
class StatusLight(QtWidgets.QLabel):
    def __init__(self, color="#666666", size=12, parent=None):
        super().__init__(parent); self._size = size
        self.setFixedSize(size, size); self.setColor(color)
    def setColor(self, color):
        self.setStyleSheet(f"background:{color}; border-radius:{self._size//2}px;")

class ResizableImageLabel(QtWidgets.QLabel):
    def __init__(self, placeholder=""):
        super().__init__(); self._qimg = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background:#000; border:2px solid #404040; border-radius:6px; color:#888;")
        if placeholder: self.setText(placeholder)
    def setImage(self, qimg: QtGui.QImage):
        self._qimg = qimg; self._updatePixmap()
    def clearImage(self, text=""):
        self._qimg = None; self.setPixmap(QtGui.QPixmap()); self.setText(text)
    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        self._updatePixmap(); super().resizeEvent(e)
    def _updatePixmap(self):
        if self._qimg is None: return
        scaled = self._qimg.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(QtGui.QPixmap.fromImage(scaled)); self.setText("")

class KPIBox(QtWidgets.QFrame):
    def __init__(self, label, unit, value_color, value_pt=28, header_pt=12):
        super().__init__(); self.setObjectName("kpiBox")
        self.setStyleSheet("QFrame#kpiBox { background:#1f1f1f; border:1px solid #404040; border-radius:10px; }")
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(12,10,12,10); lay.setSpacing(6)
        header = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setFont(sans(header_pt, True)); lbl.setStyleSheet("color:#eaeaea;")
        unit_lbl = QtWidgets.QLabel(unit); unit_lbl.setFont(sans(header_pt-2))
        unit_lbl.setStyleSheet("color:#aaa; background:#333; padding:1px 6px; border-radius:5px;")
        header.addWidget(lbl); header.addStretch(1); header.addWidget(unit_lbl)
        lay.addLayout(header)
        self.value_label = QtWidgets.QLabel("000.00")
        self.value_label.setFont(monospace(value_pt, True))
        self.value_label.setAlignment(QtCore.Qt.AlignCenter)
        self.value_label.setStyleSheet(f"color:{value_color};")
        lay.addWidget(self.value_label)
    def setValue(self, text): self.value_label.setText(text)

class LivePanel(QtWidgets.QFrame):
    def __init__(self, left_w, mid_h, s):
        super().__init__(); self.setObjectName("livePanel")
        self.setStyleSheet("QFrame#livePanel { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2a2a2a, stop:1 #333333);"
                           "border:1px solid #404040; border-radius:10px; }")
        self.setFixedWidth(left_w)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(14,14,14,14); lay.setSpacing(8)
        title = QtWidgets.QLabel("LIVE CAMERA FEED"); title.setFont(sans(pt(12, s), True)); title.setStyleSheet("color:#fff;")
        lay.addWidget(title)
        self.image_label = ResizableImageLabel("Connecting camera..."); lay.addWidget(self.image_label, 1)
    def set_qimage(self, qimg): self.image_label.setImage(qimg)

class RightPanel(QtWidgets.QFrame):
    def __init__(self, right_w, s):
        super().__init__(); self.setObjectName("rightPanel")
        self.setStyleSheet("QFrame#rightPanel{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #2a2a2a,stop:1 #333);"
                           "border:1px solid #404040;border-radius:10px;}")
        self.setFixedWidth(right_w)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(14,14,14,14); lay.setSpacing(10)

        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("MEASUREMENT DATA & LOG"); title.setFont(sans(pt(12, s), True)); title.setStyleSheet("color:#fff;")
        head.addWidget(title); head.addStretch(1); lay.addLayout(head)

        top = QtWidgets.QHBoxLayout(); top.setSpacing(10)
        meas_w = int(right_w * 0.48)
        meas_container = QtWidgets.QWidget(); meas_container.setFixedWidth(meas_w)
        meas_col = QtWidgets.QVBoxLayout(meas_container); meas_col.setContentsMargins(0,0,0,0); meas_col.setSpacing(8)

        self.kpi_width  = KPIBox("WIDTH",  "MM",  "#00ff88", value_pt=pt(20, s), header_pt=pt(10, s))
        self.kpi_height = KPIBox("HEIGHT", "MM",  "#00ff88", value_pt=pt(20, s), header_pt=pt(10, s))
        self.kpi_area   = KPIBox("AREA",   "MM²", "#00aaff", value_pt=pt(20, s), header_pt=pt(10, s))
        meas_col.addWidget(self.kpi_width); meas_col.addWidget(self.kpi_height); meas_col.addWidget(self.kpi_area)
        top.addWidget(meas_container, 0)

        log_container = QtWidgets.QWidget(); log_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        log_col = QtWidgets.QVBoxLayout(log_container); log_col.setContentsMargins(0,0,0,0); log_col.setSpacing(6)
        log_label = QtWidgets.QLabel("DATA LOG"); log_label.setFont(sans(pt(12, s), True)); log_label.setStyleSheet("color:#fff;")
        log_col.addWidget(log_label)

        self.log_table = QtWidgets.QTableWidget(0,3)
        self.log_table.setHorizontalHeaderLabels(["Time","Data","Status"])
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.horizontalHeader().setStretchLastSection(False)
        self.log_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.log_table.setColumnWidth(0, int(70*s))
        self.log_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self.log_table.setColumnWidth(2, int(64*s))
        self.log_table.setStyleSheet(
            "QTableWidget { background:#1f1f1f; color:#ddd; gridline-color:#333; border:1px solid #404040; }"
            "QHeaderView::section { background:#2a2a2a; color:#fff; border:1px solid #404040; }"
            "QTableWidget::item:selected { background:#2f3a2f; color:#e6ffe6; }"
            "QTableWidget::item { padding:4px; }"
        )
        self.log_table.setMinimumHeight(int(200*s))
        log_col.addWidget(self.log_table, 1)

        self.class_pill = QtWidgets.QLabel("Type: —")
        self.class_pill.setFont(sans(pt(11, s), True))
        self.class_pill.setAlignment(QtCore.Qt.AlignCenter)
        self.class_pill.setStyleSheet("color:#fff; background:#444; border:1px solid #666; border-radius:8px; padding:6px 10px;")
        log_col.addWidget(self.class_pill, 0)

        top.addWidget(log_container, 1)
        lay.addLayout(top, 1)

        sub = QtWidgets.QHBoxLayout()
        sub_title = QtWidgets.QLabel("PROCESSED OUTPUT"); sub_title.setFont(sans(pt(12, s), True)); sub_title.setStyleSheet("color:#fff;")
        sub.addWidget(sub_title); sub.addStretch(1); lay.addLayout(sub)
        self.proc_preview = ResizableImageLabel("PROCESSED OUTPUT PREVIEW")
        self.proc_preview.setMinimumHeight(int(180*s))
        lay.addWidget(self.proc_preview, 2)

    def set_class(self, name, color_hex):
        self.class_pill.setText(f"Type: {name}")
        self.class_pill.setStyleSheet(f"color:#fff; background:{color_hex}CC; border:1px solid {color_hex}; border-radius:8px; padding:6px 10px;")

# ---------------------- Camera Thread ----------------------
class CameraWorker(QtCore.QThread):
    frame_qimage = QtCore.pyqtSignal(QtGui.QImage)
    camera_online = QtCore.pyqtSignal(bool)
    def __init__(self, pipeline: str, parent=None):
        super().__init__(parent); self.pipeline = pipeline; self.cap = None; self.running = False; self.latest_frame = None
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        ok = self.cap.isOpened(); self.camera_online.emit(ok)
        if not ok:
            logger.error("Unable to open the camera using GStreamer pipeline."); return
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_online.emit(False); break
            frame = resize_to_target(frame); self.latest_frame = frame.copy()
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888).copy()
            self.frame_qimage.emit(qimg)
        self.release()
    def stop(self): self.running = False
    def release(self):
        try:
            if self.cap is not None: self.cap.release()
        except Exception: pass

# ---------------------- Main Window ----------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Measurement System")
        screen = QtWidgets.QApplication.primaryScreen(); geo = screen.availableGeometry()
        self.screen_w, self.screen_h = geo.width(), geo.height(); self.move(geo.topLeft())
        self.s = clamp(min(self.screen_w/BASE_W, self.screen_h/BASE_H), 0.75, 2.5)
        self.setFixedSize(self.screen_w, self.screen_h); self.statusBar().setSizeGripEnabled(False)
        self.outer_m = int(12*self.s); self.spacing = int(12*self.s)

        # Processor
        self.processor = proc.Processor(
            calibration_data_path,
            blue_lower=blue_Lower, blue_upper=blue_Upper,
            rotate_angle_deg=ROTATE_ANGLE_DEG, crop_slice=CROP_SLICE,
            pixel_mm_ratio_w=PIXEL_MM_RATIO_W, pixel_mm_ratio_h=PIXEL_MM_RATIO_H,
            logger=logger
        )

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(self.outer_m,self.outer_m,self.outer_m,self.outer_m); root.setSpacing(self.spacing)

        header = self._build_header(); root.addWidget(header)
        mid_h = self.screen_h - 2*self.outer_m - header.height() - int(64*self.s) - self.spacing
        total_w = self.screen_w - 2*self.outer_m
        left_w  = int(total_w * 0.66) - self.spacing//2
        right_w = total_w - left_w - self.spacing

        middle = QtWidgets.QHBoxLayout(); middle.setSpacing(self.spacing)
        self.live = LivePanel(left_w, mid_h, self.s); middle.addWidget(self.live)
        self.right = RightPanel(right_w, self.s);     middle.addWidget(self.right)
        root.addLayout(middle, 1)

        bottom = self._build_bottom_bar(); root.addWidget(bottom)
        self._apply_dark_palette()

        self.image_bgr = None; self.is_processing = False

        if USE_IMAGE:
            if not os.path.exists(IMAGE_SOURCE):
                QtWidgets.QMessageBox.critical(self, "Image Mode", f"Image not found:\n{IMAGE_SOURCE}"); logger.error(f"Image not found: {IMAGE_SOURCE}")
            else:
                self.image_bgr = cv2.imread(IMAGE_SOURCE)
                if self.image_bgr is None:
                    QtWidgets.QMessageBox.critical(self, "Image Mode", f"Failed to read image:\n{IMAGE_SOURCE}"); logger.error(f"Failed to read image: {IMAGE_SOURCE}")
                else:
                    self.image_bgr = resize_to_target(self.image_bgr)
                    self.image_timer = QtCore.QTimer(self, interval=250, timeout=self._update_image_feed)
                    self.image_timer.start(); self._on_camera_online(True)
        else:
            self.cam = CameraWorker(PIPELINE, self)
            self.cam.frame_qimage.connect(self.live.set_qimage)
            self.cam.camera_online.connect(self._on_camera_online)
            self.cam.start()

        self.cooldown_ms = 600
        self.measure_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("F1"), self)
        self.measure_shortcut.activated.connect(self._handle_measure)

    # ---------- UI helpers ----------
    def _apply_dark_palette(self):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#1a1a1a"))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1f1f1f"))
        pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#2c2c2c"))
        pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        self.setPalette(pal)

    def _build_header(self):
        h = int(76*self.s)
        w = QtWidgets.QFrame(); w.setObjectName("header")
        w.setStyleSheet("QFrame#header{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #2c2c2c,stop:1 #3a3a3a);border:1px solid #404040;border-radius:10px;}")
        w.setFixedHeight(h)
        l = QtWidgets.QHBoxLayout(w); l.setContentsMargins(int(14*self.s),int(10*self.s),int(14*self.s),int(10*self.s))
        title = QtWidgets.QLabel("MEASUREMENT SYSTEM"); title.setFont(sans(pt(15,self.s), True)); title.setStyleSheet("color:#fff;")
        l.addWidget(title); l.addStretch(1)
        self.datetime_lbl = QtWidgets.QLabel("--:--:-- • ----/--/--"); self.datetime_lbl.setFont(monospace(pt(11,self.s)))
        self.datetime_lbl.setStyleSheet("background:#404040;border:1px solid #606060;color:#fff;padding:6px 10px;border-radius:6px;")
        l.addWidget(self.datetime_lbl, 0, QtCore.Qt.AlignRight)
        self.header_dot = StatusLight("#00ff00", max(8,int(10*self.s))); l.addWidget(self.header_dot, 0, QtCore.Qt.AlignRight)
        self.header_text = QtWidgets.QLabel("SYSTEM READY"); self.header_text.setFont(sans(pt(10,self.s))); self.header_text.setStyleSheet("color:#00ff00;")
        l.addWidget(self.header_text, 0, QtCore.Qt.AlignRight)
        self.clock = QtCore.QTimer(self, interval=1000, timeout=self._update_clock); self.clock.start()
        return w

    def _build_bottom_bar(self):
        h = int(64*self.s)
        bar = QtWidgets.QFrame(); bar.setObjectName("bottomBar")
        bar.setStyleSheet(
            "QFrame#bottomBar{background:#121212;border:1px solid #2a2a2a;border-radius:10px;}"
            "QPushButton#measureBtn{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #28a745,stop:1 #20833a);"
            "color:#fff;border:2px solid #28a745;border-radius:10px;padding:10px 18px;font-weight:600;letter-spacing:0.5px;}")
        bar.setFixedHeight(h)
        l = QtWidgets.QHBoxLayout(bar); l.setContentsMargins(int(12*self.s),int(8*self.s),int(12*self.s),int(8*self.s))
        l.addStretch(1)
        self.btn_measure = QtWidgets.QPushButton("MEASURE"); self.btn_measure.setObjectName("measureBtn")
        self.btn_measure.setFont(sans(pt(13,self.s), True)); self.btn_measure.setMinimumWidth(int(220*self.s))
        l.addWidget(self.btn_measure, 0, QtCore.Qt.AlignCenter)
        l.addStretch(1)
        self.btn_measure.clicked.connect(self._handle_measure)
        return bar

    # ---------- Timers & Status ----------
    def _on_camera_online(self, ok: bool):
        self.header_dot.setColor("#00ff00" if ok else "#ff3333")
        if ok: self.live.image_label.clearImage("")
    def _update_clock(self):
        self.datetime_lbl.setText(datetime.now().strftime("%H:%M:%S • %Y-%m-%d"))
    def _update_image_feed(self):
        if self.image_bgr is None: return
        frame = self.image_bgr.copy()
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        self.live.set_qimage(bgr_to_qimage(frame))

    # ---------- Measure ----------
    def _lock_measure_ui(self):
        self.is_processing = True
        self.btn_measure.setEnabled(False)
        self.btn_measure.setText("MEASURING…")
        if hasattr(self, "measure_shortcut"): self.measure_shortcut.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.header_text.setText("PROCESSING"); self.header_text.setStyleSheet("color:#ffaa00;")

    def _unlock_measure_ui(self):
        def _enable():
            self.is_processing = False
            self.btn_measure.setEnabled(True)
            self.btn_measure.setText("MEASURE")
            if hasattr(self, "measure_shortcut"): self.measure_shortcut.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.header_text.setText("SYSTEM READY"); self.header_text.setStyleSheet("color:#00ff00;")
        QtCore.QTimer.singleShot(self.cooldown_ms, _enable)

    def _handle_measure(self):
        if self.is_processing:
            return
        self._lock_measure_ui()
        try:
            # --- 1) เตรียมภาพ input ---
            if USE_IMAGE:
                if self.image_bgr is None:
                    QtWidgets.QMessageBox.warning(self, "Image Mode", "No image loaded.")
                    results, output_bgr = None, None
                else:
                    results, output_bgr = self.processor.run_measurement(self.image_bgr.copy())
                    current_src_name = getattr(self, "current_source_filename", "")  # ถ้ามีการเก็บชื่อไฟล์ภาพไว้
            else:
                if not hasattr(self, "cam") or self.cam.latest_frame is None:
                    QtWidgets.QMessageBox.warning(self, "Camera", "No camera frame available yet.")
                    results, output_bgr = None, None
                else:
                    frame = self.cam.latest_frame.copy()
                    results, output_bgr = self.processor.run_measurement(frame)
                    current_src_name = "camera_frame"

            # --- 2) อัปเดต UI + เซฟ output + ส่ง API ---
            if results and output_bgr is not None:
                # 2.1 ค่าหลัก
                w_raw = float(results["width_mm"])
                h_raw = float(results["height_mm"])
                a_raw = float(results["area_mm2"])
                type_name = results.get("type", "-")
                cls_color = results.get("class_color", "#00AEEF")

                w = round(w_raw, 2)
                h = round(h_raw, 2)
                a = round(a_raw, 2)

                self.right.kpi_width.setValue(f"{w:.2f}")
                self.right.kpi_height.setValue(f"{h:.2f}")
                self.right.kpi_area.setValue(f"{a:.2f}")
                self.right.set_class(type_name, cls_color)

                # 2.2 เลขรันนิ่ง + เซฟรูป output (แนะนำใช้ .jpg ให้เข้ากับ API)
                run = int(get_next_running_number())
                out_name = f"MTL_{run:06d}.jpg"
                out_path = os.path.join(img_output_path, out_name)
                cv2.imwrite(out_path, output_bgr)

                # 2.3 เตรียม S1..SN จาก rects_mm ถ้ามี
                # rects_mm: List[Tuple[W_mm, H_mm, A_mm2]]
                rects = results.get("rects_mm", [])
                s_payload = None
                s_info = ""
                if rects:
                    s_payload = [
                        {
                            "name": f"S{i}",
                            "W_mm": round(float(rwmm), 1),
                            "H_mm": round(float(rhmm), 1),
                            "A_mm2": round(float(ramm2), 0),
                        }
                        for i, (rwmm, rhmm, ramm2) in enumerate(rects, start=1)
                    ]
                    s_info = "  " + format_S_for_log(s_payload)

                # 2.4 log UI
                log_msg = (
                    f"MEASURE OK run={run} "
                    f"W={w:.2f}mm H={h:.2f}mm A={a:.2f}mm2 "
                    f"Type={type_name}{s_info}"
                )
                logger.info(log_msg)

                # 2.5 prepend ลงตาราง log ด้านขวา
                t = datetime.now().strftime("%H:%M:%S")
                # เอาข้อความย่อ ไม่รวม prefix run=
                short_msg = log_msg.replace(f"MEASURE OK run={run} ", "")
                self._prepend_log_row([t, short_msg, "✓"])

                # 2.6 แสดงภาพผลลัพธ์ใน preview
                self._set_proc_preview(output_bgr)

                # 2.7 ส่งผลเข้า API (รูป output + ตัวเลข + S1..SN)
                try:
                    type_name = results.get("type", "-")
                    api_json = post_result_to_api(
                        output_img_path=out_path,
                        run_no=run,
                        width_mm=w,
                        height_mm=h,
                        area_mm2=a,
                        s_map_or_list=s_payload,
                        source_filename=current_src_name,
                        output_filename=out_name,
                        extras={"type_name": type_name},
                        logger=logger
                    )
                    # จะเอาค่า id/run_no/outfile จาก api_json ไปโชว์เพิ่มก็ได้
                    # logger.info(f"[API] DB row id={api_json.get('id')} saved")
                except Exception as api_err:
                    logger.exception(f"Send to API failed: {api_err}")
                    # ไม่ให้ UI ค้าง/ล้ม แม้ API มีปัญหา

            else:
                t = datetime.now().strftime("%H:%M:%S")
                self._prepend_log_row([t, "No object detected", "⚠"])
                self._set_proc_preview(None)
                self.right.set_class("—", "#444444")
                logger.warning("MEASURE NO_OBJECT")

        except Exception as e:
            logger.exception("Measurement error")
            QtWidgets.QMessageBox.critical(self, "Error", f"Measurement failed:\n{e}")
        finally:
            self._unlock_measure_ui()

    # ---------- small helpers ----------
    def _set_proc_preview(self, bgr):
        if bgr is None: self.right.proc_preview.clearImage("NO PROCESSED IMAGE"); return
        self.right.proc_preview.setImage(bgr_to_qimage(bgr))

    def _prepend_log_row(self, items):
        table = self.right.log_table
        table.insertRow(0)
        for col, text in enumerate(items):
            it = QtWidgets.QTableWidgetItem(text)
            it.setFont(monospace(pt(10,self.s)))
            if col == 2:
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                it.setForeground(QtGui.QBrush(QtGui.QColor("#00ff88" if text == "✓" else "#ffaa00")))
            table.setItem(0, col, it)
        while table.rowCount() > 10:
            table.removeRow(table.rowCount() - 1)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            if not USE_IMAGE and hasattr(self, "cam") and self.cam and self.cam.isRunning():
                self.cam.stop(); self.cam.wait(1500)
            if USE_IMAGE and hasattr(self, "image_timer"):
                self.image_timer.stop()
        except Exception: pass
        return super().closeEvent(event)

# ---------------------- Entry ----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MTL Industrial Measurement System")
    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
