# -*- coding: utf-8 -*-
"""
MTL Industrial Measurement System (PyQt5 + OpenCV/GStreamer)
- Window size = screen available area (per device), user cannot resize
- Auto-scale fonts, paddings, KPI box heights based on screen scale (base 1280x720)
- Left (≈ 66%): Live camera (kept 16:9, resizable inside)
- Right (≈ 34%): Measurement (vertical) + Data Log, Processed Output below (kept 16:9)
"""

import sys, os, logging
import numpy as np
import cv2
from logging.handlers import RotatingFileHandler
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets

# ---------------------- Configuration & Paths ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
running_number_path = os.path.join(script_dir, 'running_number.txt')
calibration_data_path = os.path.join(script_dir, 'calibration_data/calibration_data_7x5_3040p.npz')
img_output_path = os.path.join(script_dir, 'output')
os.makedirs(img_output_path, exist_ok=True)
log_path = os.path.join(script_dir, 'app.log')

# Camera Pipeline
PIPELINE = (
    "souphttpsrc location=http://172.16.165.250:4747/video ! "
    "multipartdemux ! jpegdec ! videoconvert ! appsink"
)

# Measurement params
blue_Lower = (90, 10, 50)
blue_Upper = (150, 255, 255)
ROTATE_ANGLE_DEG = 2.4
CROP_SLICE = (slice(340, 2420), slice(100, 4000))   # [y1:y2, x1:x2]
PIXEL_MM_RATIO_W = 0.53
PIXEL_MM_RATIO_H = 0.53

# ---------------------- Logging ----------------------
logger = logging.getLogger('MTL_MEASUREMENT')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, mode='a')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(handler)

# ---------------------- Utilities ----------------------
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def get_next_running_number(filename=running_number_path):
    try:
        with open(filename, "r") as f:
            running_number = int(f.read().strip()) + 1
    except Exception:
        running_number = 1
    if running_number > 9999:
        running_number = 1
    try:
        with open(filename, "w") as f:
            f.write(str(running_number))
    except Exception as e:
        logger.error(f'Failed to write running number: {e}')
    return format(running_number, '04d')

def load_calibration_data(path):
    try:
        data = np.load(path)
        mtx, dist = data['mtx'], data['dist']
        logger.debug(f'Calibration loaded from {path}')
        return mtx, dist
    except Exception as e:
        logger.error(f'Failed to load calibration data from {path}: {e}')
        return None, None

def cal_dim(contours, hierarchy, px2mm_x, px2mm_y):
    total_area_px = 0
    max_rect = None
    max_cnt = None
    if hierarchy is None:
        return None
    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if hierarchy[0][i][3] != -1:
            total_area_px -= area_px
        else:
            total_area_px += area_px
            if max_cnt is None or area_px > cv2.contourArea(max_cnt):
                max_cnt = cnt
                max_rect = cv2.minAreaRect(cnt)
    if max_rect is None or max_cnt is None:
        return None
    area_mm2 = total_area_px * px2mm_x * px2mm_y
    (cx, cy), (w, h), angle = max_rect
    width_mm = w * px2mm_x
    height_mm = h * px2mm_y
    return {"area_mm2": area_mm2, "width_mm": width_mm, "height_mm": height_mm,
            "angle": angle, "contour": max_cnt}

def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888).copy()

# ---------------------- Scaling helpers ----------------------
BASE_W, BASE_H = 1280, 720  # design baseline

def clamp(v, lo, hi): return hi if v > hi else lo if v < lo else v

def pt(base, s):  # scale font points
    return max(7, int(round(base * s)))

# ---------------------- UI helper widgets ----------------------
def monospace(size=12, bold=False):
    f = QtGui.QFont("Courier New"); f.setPointSize(size); f.setBold(bold); return f

def sans(size=12, bold=False):
    f = QtGui.QFont("Montserrat"); f.setPointSize(size); f.setBold(bold); return f

class StatusLight(QtWidgets.QLabel):
    def __init__(self, color="#666666", size=12, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setColor(color)
    def setColor(self, color):
        self.setStyleSheet(f"background:{color}; border-radius:{self._size//2}px;")

class ResizableImageLabel(QtWidgets.QLabel):
    """รักษาอัตราส่วนภาพและขยายเต็มพื้นที่ที่มี"""
    def __init__(self, placeholder=""):
        super().__init__()
        self._qimg = None
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
        super().__init__()
        self.setObjectName("kpiBox")
        self.setStyleSheet("QFrame#kpiBox { background:#1f1f1f; border:1px solid #404040; border-radius:10px; }")
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(12,10,12,10); lay.setSpacing(6)
        header = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setFont(sans(header_pt, True)); lbl.setStyleSheet("color:#eaeaea;")
        unit_lbl = QtWidgets.QLabel(unit); unit_lbl.setFont(sans(header_pt-2)); unit_lbl.setStyleSheet(
            "color:#aaa; background:#333; padding:1px 6px; border-radius:5px;")
        header.addWidget(lbl); header.addStretch(1); header.addWidget(unit_lbl)
        lay.addLayout(header)
        self.value_label = QtWidgets.QLabel("000.00")
        self.value_label.setFont(monospace(value_pt, True))
        self.value_label.setAlignment(QtCore.Qt.AlignCenter)
        self.value_label.setStyleSheet(f"color:{value_color};")
        lay.addWidget(self.value_label)
    def setValue(self, text): self.value_label.setText(text)

# ---------------------- Panels ----------------------
class LivePanel(QtWidgets.QFrame):
    def __init__(self, left_w, mid_h, s):
        super().__init__()
        self.setObjectName("livePanel")
        self.setStyleSheet("QFrame#livePanel { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2a2a2a, stop:1 #333333);"
                           "border:1px solid #404040; border-radius:10px; }")
        self.setFixedWidth(left_w)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(14,14,14,14); lay.setSpacing(8)
        title = QtWidgets.QLabel("LIVE CAMERA FEED"); title.setFont(sans(pt(12, s), True)); title.setStyleSheet("color:#fff;")
        lay.addWidget(title)
        self.image_label = ResizableImageLabel("Connecting camera...")
        lay.addWidget(self.image_label, 1)
        row = QtWidgets.QHBoxLayout()
        def item(name, color):
            w = QtWidgets.QWidget(); l = QtWidgets.QHBoxLayout(w); l.setContentsMargins(0,0,0,0)
            dot = StatusLight(color, max(8, int(10*s))); lbl = QtWidgets.QLabel(name); lbl.setFont(sans(pt(10, s))); lbl.setStyleSheet("color:#ccc;")
            l.addWidget(dot); l.addWidget(lbl); return w, dot
        w1, self.power_dot = item("POWER", "#00ff00")
        w2, self.camera_dot = item("CAMERA", "#ff3333")
        w3, self.proc_dot   = item("PROCESSING", "#666")
        w4, self.net_dot    = item("NETWORK", "#00ff00")
        row.addWidget(w1); row.addWidget(w2); row.addWidget(w3); row.addWidget(w4); row.addStretch(1); lay.addLayout(row)
    def set_qimage(self, qimg): self.image_label.setImage(qimg)

class RightPanel(QtWidgets.QFrame):
    def __init__(self, right_w, s):
        super().__init__()
        self.setObjectName("rightPanel")
        self.setStyleSheet("QFrame#rightPanel{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #2a2a2a,stop:1 #333);"
                           "border:1px solid #404040;border-radius:10px;}")
        self.setFixedWidth(right_w)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(14,14,14,14); lay.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("MEASUREMENT DATA & LOG"); title.setFont(sans(pt(12, s), True)); title.setStyleSheet("color:#fff;")
        head.addWidget(title); head.addStretch(1); lay.addLayout(head)

        # top row: measurement + log
        top = QtWidgets.QHBoxLayout(); top.setSpacing(10)
        meas_w = int(right_w * 0.48)
        meas_container = QtWidgets.QWidget(); meas_container.setFixedWidth(meas_w)
        meas_col = QtWidgets.QVBoxLayout(meas_container); meas_col.setContentsMargins(0,0,0,0); meas_col.setSpacing(8)

        self.kpi_width  = KPIBox("WIDTH",  "MM",  "#00ff88", value_pt=pt(26, s), header_pt=pt(12, s))
        self.kpi_height = KPIBox("HEIGHT", "MM",  "#00ff88", value_pt=pt(26, s), header_pt=pt(12, s))
        self.kpi_area   = KPIBox("AREA",   "MM²", "#00aaff", value_pt=pt(26, s), header_pt=pt(12, s))
        meas_col.addWidget(self.kpi_width)
        meas_col.addWidget(self.kpi_height)
        meas_col.addWidget(self.kpi_area)
        top.addWidget(meas_container, 0)

        log_container = QtWidgets.QWidget()
        log_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        log_col = QtWidgets.QVBoxLayout(log_container); log_col.setContentsMargins(0,0,0,0); log_col.setSpacing(6)
        log_label = QtWidgets.QLabel("DATA LOG"); log_label.setFont(sans(pt(12, s), True)); log_label.setStyleSheet("color:#fff;")
        log_col.addWidget(log_label)

        self.log_table = QtWidgets.QTableWidget(0,3)
        self.log_table.setHorizontalHeaderLabels(["Time","Data","Status"])
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
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
        top.addWidget(log_container, 1)

        lay.addLayout(top, 1)

        # processed output
        sub = QtWidgets.QHBoxLayout()
        sub_title = QtWidgets.QLabel("PROCESSED OUTPUT"); sub_title.setFont(sans(pt(12, s), True)); sub_title.setStyleSheet("color:#fff;")
        sub.addWidget(sub_title); sub.addStretch(1); lay.addLayout(sub)
        self.proc_preview = ResizableImageLabel("PROCESSED OUTPUT PREVIEW")
        self.proc_preview.setMinimumHeight(int(180*s))
        lay.addWidget(self.proc_preview, 2)

# ---------------------- Camera Thread ----------------------
class CameraWorker(QtCore.QThread):
    frame_qimage = QtCore.pyqtSignal(QtGui.QImage)
    camera_online = QtCore.pyqtSignal(bool)
    def __init__(self, pipeline: str, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.cap = None
        self.running = False
        self.latest_frame = None
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        online = self.cap.isOpened()
        self.camera_online.emit(online)
        if not online:
            logger.error("Unable to open the camera using GStreamer pipeline.")
            return
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_online.emit(False); break
            self.latest_frame = frame.copy()
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888).copy()
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
        super().__init__()
        self.setWindowTitle("MTL Industrial Measurement System")

        # --- Detect screen & lock to that size ---
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.availableGeometry()
        self.screen_w, self.screen_h = geo.width(), geo.height()
        self.move(geo.topLeft())
        # scale factor (clamped for extreme monitors)
        self.s = clamp(min(self.screen_w/BASE_W, self.screen_h/BASE_H), 0.75, 2.5)

        # lock size to screen area (no manual resize)
        self.setFixedSize(self.screen_w, self.screen_h)
        self.statusBar().setSizeGripEnabled(False)

        # layout margins/spacings scaled
        self.outer_m = int(12*self.s)
        self.spacing = int(12*self.s)

        self.mtx, self.dist = load_calibration_data(calibration_data_path)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(self.outer_m,self.outer_m,self.outer_m,self.outer_m); root.setSpacing(self.spacing)

        header = self.build_header(); root.addWidget(header)

        # compute mid sizes & column widths
        mid_h = self.screen_h - 2*self.outer_m - header.height() - int(64*self.s) - self.spacing
        total_w = self.screen_w - 2*self.outer_m
        left_w  = int(total_w * 0.66) - self.spacing//2
        right_w = total_w - left_w - self.spacing

        # ---- Middle: two fixed panels (no splitter) ----
        middle = QtWidgets.QHBoxLayout(); middle.setSpacing(self.spacing)
        self.live = LivePanel(left_w, mid_h, self.s); middle.addWidget(self.live)
        self.right = RightPanel(right_w, self.s);     middle.addWidget(self.right)
        root.addLayout(middle, 1)

        bottom = self.build_bottom_bar(); root.addWidget(bottom)

        self.apply_dark_palette()

        # camera thread
        self.cam = CameraWorker(PIPELINE, self)
        self.cam.frame_qimage.connect(self.live.set_qimage)
        self.cam.camera_online.connect(self.on_camera_online)
        self.cam.start()

        QtWidgets.QShortcut(QtGui.QKeySequence("F1"), self, activated=self.handle_measure)
        self.is_processing = False

    # ---- Header ----
    def build_header(self):
        h = int(76*self.s)
        w = QtWidgets.QFrame(); w.setObjectName("header")
        w.setStyleSheet("QFrame#header{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #2c2c2c,stop:1 #3a3a3a);"
                        "border:1px solid #404040;border-radius:10px;}")
        w.setFixedHeight(h)
        l = QtWidgets.QHBoxLayout(w); l.setContentsMargins(int(14*self.s),int(10*self.s),int(14*self.s),int(10*self.s))
        logo = QtWidgets.QLabel("MTL"); logo.setAlignment(QtCore.Qt.AlignCenter)
        logo.setFixedSize(int(48*self.s),int(48*self.s))
        logo.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #ff6b35,stop:1 #f7931e);"
                           "color:white;border-radius:8px;font-weight:700;")
        f = logo.font(); f.setPointSize(pt(20, self.s)); logo.setFont(f)

        title_box = QtWidgets.QVBoxLayout()
        t = QtWidgets.QLabel("MEASUREMENT SYSTEM"); t.setFont(sans(pt(15,self.s), True)); t.setStyleSheet("color:#fff;")
        st = QtWidgets.QLabel("Industrial Grade Precision Control"); st.setFont(sans(pt(9,self.s))); st.setStyleSheet("color:#ccc;")
        title_box.addWidget(t); title_box.addWidget(st)
        left = QtWidgets.QHBoxLayout(); left.addWidget(logo); left.addLayout(title_box)
        l.addLayout(left); l.addStretch(1)

        self.datetime_lbl = QtWidgets.QLabel("--:--:-- • ----/--/--"); self.datetime_lbl.setFont(monospace(pt(11,self.s)))
        self.datetime_lbl.setStyleSheet("background:#404040;border:1px solid #606060;color:#fff;padding:6px 10px;border-radius:6px;")
        l.addWidget(self.datetime_lbl, 0, QtCore.Qt.AlignRight)
        self.header_dot = StatusLight("#00ff00", max(8,int(10*self.s))); l.addWidget(self.header_dot, 0, QtCore.Qt.AlignRight)
        self.header_text = QtWidgets.QLabel("SYSTEM READY"); self.header_text.setFont(sans(pt(10,self.s))); self.header_text.setStyleSheet("color:#00ff00;")
        l.addWidget(self.header_text, 0, QtCore.Qt.AlignRight)
        self.clock = QtCore.QTimer(self, interval=1000, timeout=self.update_clock); self.clock.start()
        return w

    # ---- Bottom ----
    def build_bottom_bar(self):
        h = int(64*self.s)
        bar = QtWidgets.QFrame(); bar.setObjectName("bottomBar")
        bar.setStyleSheet(
            "QFrame#bottomBar{background:#121212;border:1px solid #2a2a2a;border-radius:10px;}"
            "QPushButton#measureBtn{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #28a745,stop:1 #20833a);"
            "color:#fff;border:2px solid #28a745;border-radius:10px;padding:10px 18px;font-weight:600;letter-spacing:0.5px;}")
        bar.setFixedHeight(h)
        l = QtWidgets.QHBoxLayout(bar); l.setContentsMargins(int(12*self.s),int(8*self.s),int(12*self.s),int(8*self.s))
        l.addStretch(1)
        self.btn_measure = QtWidgets.QPushButton("MEASURE (F1)"); self.btn_measure.setObjectName("measureBtn")
        self.btn_measure.setFont(sans(pt(13,self.s), True)); self.btn_measure.setMinimumWidth(int(220*self.s))
        l.addWidget(self.btn_measure, 0, QtCore.Qt.AlignCenter)
        l.addStretch(1)
        self.btn_measure.clicked.connect(self.handle_measure)
        return bar

    def apply_dark_palette(self):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#1a1a1a"))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1f1f1f"))
        pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#2c2c2c"))
        pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        self.setPalette(pal)

    # ---- Status/clock ----
    def on_camera_online(self, ok: bool):
        self.live.power_dot.setColor("#00ff00")
        self.live.camera_dot.setColor("#00ff00" if ok else "#ff3333")
        if ok: self.live.image_label.clearImage("")
    def update_clock(self):
        self.datetime_lbl.setText(datetime.now().strftime("%H:%M:%S • %Y-%m-%d"))

    # ---- Measurement ----
    def handle_measure(self):
        if self.is_processing: return
        if not hasattr(self.cam, "latest_frame") or self.cam.latest_frame is None:
            QtWidgets.QMessageBox.warning(self, "Camera", "No camera frame available yet."); return
        self.is_processing = True; self.live.proc_dot.setColor("#ffd700")
        try:
            results, output_bgr = self.run_measurement(self.cam.latest_frame.copy())
        except Exception as e:
            logger.exception("Measurement error")
            QtWidgets.QMessageBox.critical(self, "Error", f"Measurement failed:\n{e}")
            self.live.proc_dot.setColor("#666666"); self.is_processing = False; return

        if results:
            w = results["width_mm"]; h = results["height_mm"]; a = results["area_mm2"]
            self.right.kpi_width.setValue(f"{w:.2f}")
            self.right.kpi_height.setValue(f"{h:.2f}")
            self.right.kpi_area.setValue(f"{a:.2f}")
            run = get_next_running_number()
            out_path = os.path.join(img_output_path, f"mtal{run}.png"); cv2.imwrite(out_path, output_bgr)
            t = datetime.now().strftime("%H:%M:%S")
            data = f"W:{w:.2f}  H:{h:.2f}  A:{a:.2f}"
            self.prepend_log_row([t, data, "✓"])
            self.set_proc_preview(output_bgr)
        else:
            t = datetime.now().strftime("%H:%M:%S"); self.prepend_log_row([t, "No object detected", "⚠"]); self.set_proc_preview(None)

        self.live.proc_dot.setColor("#666666"); self.is_processing = False

    def set_proc_preview(self, bgr):
        if bgr is None: self.right.proc_preview.clearImage("NO PROCESSED IMAGE"); return
        self.right.proc_preview.setImage(bgr_to_qimage(bgr))

    def run_measurement(self, frame_bgr):
        frame_HD = frame_bgr
        if self.mtx is not None and self.dist is not None:
            frame_HD = cv2.undistort(frame_HD, self.mtx, self.dist, None, self.mtx)
        frame_HD = rotate_image(frame_HD, ROTATE_ANGLE_DEG)
        yslice, xslice = CROP_SLICE
        frame_HD = frame_HD[yslice, xslice].copy()
        output = frame_HD.copy()

        gray = cv2.cvtColor(frame_HD, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, self.kernel)
        frame_no_glare = cv2.inpaint(frame_HD, closed, 3, cv2.INPAINT_TELEA)

        hsv = cv2.cvtColor(frame_no_glare, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, blue_Lower, blue_Upper)
        blue_mask = cv2.bitwise_not(blue_mask)

        if cv2.countNonZero(blue_mask) <= 200000:
            return None, output

        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 0, 0), 1)

        measurements = cal_dim(contours, hierarchy, PIXEL_MM_RATIO_W, PIXEL_MM_RATIO_H)
        if not measurements: return None, output

        object_width = measurements["width_mm"]
        object_height = measurements["height_mm"]
        object_area = measurements["area_mm2"]
        cnt = measurements["contour"]

        M = cv2.moments(cnt)
        if M["m00"] != 0: cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        else: cX, cY = 20, 20

        cv2.drawContours(output, [cnt], 0, (0, 255, 0), 2)
        cv2.putText(output, f"Area:  {object_area:.1f} mm2", (cX - 120, cY - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(output, f"Width: {object_width:.1f} mm", (cX - 120, cY - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(output, f"Height:{object_height:.1f} mm", (cX - 120, cY + 10),  cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        return {"width_mm": object_width, "height_mm": object_height, "area_mm2": object_area}, output

    def prepend_log_row(self, items):
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
            if hasattr(self, "cam") and self.cam and self.cam.isRunning():
                self.cam.stop(); self.cam.wait(1500)
        except Exception:
            pass
        return super().closeEvent(event)

# ---------------------- Entry ----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MTL Industrial Measurement System")
    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
