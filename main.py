import cv2
import sys, os, logging
import math
import numpy as np
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
# PIPELINE = (
#     "rtspsrc location=rtsp://127.0.0.1:8554/test latency=100 protocols=tcp "
#     "! rtpjpegdepay ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
# )
PIPELINE = (
    "souphttpsrc location=http://172.16.165.250:4747/video ! "
    "multipartdemux ! jpegdec ! videoconvert ! appsink"
)

TARGET_W, TARGET_H = 4032, 3040

def resize_to_target(img, w=TARGET_W, h=TARGET_H):
    ih, iw = img.shape[:2]
    interp = cv2.INTER_AREA if (iw > w or ih > h) else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)

# ---------------------- Image/Camera switch ----------------------
USE_IMAGE = True
IMAGE_SOURCE = os.path.join(script_dir, "img/imgno.jpg")

# Measurement parameters
blue_Lower = (90, 120, 120)
blue_Upper = (150, 255, 255)
ROTATE_ANGLE_DEG = 2.4
CROP_SLICE = (slice(500, 2590), slice(50, 4000))
PIXEL_MM_RATIO_W = 0.5479
PIXEL_MM_RATIO_H = 0.5454

# ---------------------- Logging ----------------------
logger = logging.getLogger('MTL_MEASUREMENT')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, mode='a')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(handler)

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

def cal_dim(approx):
    dist_list = []
    for i in range(len(approx)):
        for j in range(i+1, len(approx)):
            dist = math.hypot(approx[i][0][0] - approx[j][0][0], approx[i][0][1] - approx[j][0][1])
            dist_list.append(dist)
    dist_list = sorted(dist_list)[:4]
    if len(dist_list) < 4:
        return 0, 0
    height = int((dist_list[0] + dist_list[1]) / 2)
    width  = int((dist_list[2] + dist_list[3]) / 2)
    return width, height

def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888).copy()

BASE_W, BASE_H = 1280, 720
def clamp(v, lo, hi): return hi if v > hi else lo if v < lo else v
def pt(base, s): return max(7, int(round(base * s)))
def monospace(size=12, bold=False):
    f = QtGui.QFont("Courier New"); f.setPointSize(size); f.setBold(bold); return f
def sans(size=12, bold=False):
    f = QtGui.QFont("Montserrat"); f.setPointSize(size); f.setBold(bold); return f

# ---------------------- Shape utils ----------------------
def get_external_and_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, []
    hierarchy = hierarchy[0]
    ext_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not ext_indices:
        return None, []
    largest_idx = max(ext_indices, key=lambda i: cv2.contourArea(contours[i]))
    ext_cnt = contours[largest_idx]
    hole_indices = [i for i, h in enumerate(hierarchy) if h[3] == largest_idx]
    holes = [contours[i] for i in hole_indices if cv2.contourArea(contours[i]) > 50]
    return ext_cnt, holes

def classify_shape(cnt, approx, holes, area_px):
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    rect_area = float(rw * rh) if rw > 0 and rh > 0 else 0.0

    area_px = float(area_px)
    solidity = (area_px / hull_area) if hull_area > 0 else 0.0
    extent   = (area_px / rect_area) if rect_area > 0 else 0.0
    v        = len(approx) if approx is not None else 0
    concavity = 1.0 - ((area_px / hull_area) if hull_area > 0 else 0.0)

    SOL_MIN  = 0.95
    EXT_MIN  = 0.90
    CONC_MAX = 0.05

    if holes and len(holes) >= 1:
        return "Asymmetrical", "00d084"
    
    is_rect_like = (v in (4, 5)) and (solidity >= SOL_MIN) and (extent >= EXT_MIN) and (concavity <= CONC_MAX)

    if is_rect_like:
        return "Symmetrical", "#00d084"
    else:
        return "Asymmetrical", "#00d084"


# ---------------------- Widgets ----------------------
class StatusLight(QtWidgets.QLabel):
    def __init__(self, color="#666666", size=12, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setColor(color)
    def setColor(self, color):
        self.setStyleSheet(f"background:{color}; border-radius:{self._size//2}px;")

class ResizableImageLabel(QtWidgets.QLabel):
    """Fix size"""
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

        self.kpi_width  = KPIBox("WIDTH",  "MM",  "#00ff88", value_pt=pt(20, s), header_pt=pt(10, s))
        self.kpi_height = KPIBox("HEIGHT", "MM",  "#00ff88", value_pt=pt(20, s), header_pt=pt(10, s))
        self.kpi_area   = KPIBox("AREA",   "MM²", "#00aaff", value_pt=pt(20, s), header_pt=pt(10, s))
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

        self.class_pill = QtWidgets.QLabel("CLASS: —")
        self.class_pill.setFont(sans(pt(11, s), True))
        self.class_pill.setAlignment(QtCore.Qt.AlignCenter)
        self.class_pill.setStyleSheet(
            "color:#fff; background:#444; border:1px solid #666; border-radius:8px; padding:6px 10px;")
        log_col.addWidget(self.class_pill, 0)

        top.addWidget(log_container, 1)
        lay.addLayout(top, 1)

        # processed output
        sub = QtWidgets.QHBoxLayout()
        sub_title = QtWidgets.QLabel("PROCESSED OUTPUT"); sub_title.setFont(sans(pt(12, s), True)); sub_title.setStyleSheet("color:#fff;")
        sub.addWidget(sub_title); sub.addStretch(1); lay.addLayout(sub)
        self.proc_preview = ResizableImageLabel("PROCESSED OUTPUT PREVIEW")
        self.proc_preview.setMinimumHeight(int(180*s))
        lay.addWidget(self.proc_preview, 2)

    def set_class(self, name, color_hex):
        self.class_pill.setText(f"CLASS: {name}")
        self.class_pill.setStyleSheet(
            f"color:#fff; background:{color_hex}CC; border:1px solid {color_hex}; border-radius:8px; padding:6px 10px;"
        )

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
            frame = resize_to_target(frame)
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
        self.setWindowTitle("Measurement System")

        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.availableGeometry()
        self.screen_w, self.screen_h = geo.width(), geo.height()
        self.move(geo.topLeft())
        self.s = clamp(min(self.screen_w/BASE_W, self.screen_h/BASE_H), 0.75, 2.5)

        self.setFixedSize(self.screen_w, self.screen_h)
        self.statusBar().setSizeGripEnabled(False)

        self.outer_m = int(12*self.s)
        self.spacing = int(12*self.s)

        self.mtx, self.dist = load_calibration_data(calibration_data_path)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(self.outer_m,self.outer_m,self.outer_m,self.outer_m); root.setSpacing(self.spacing)

        header = self.build_header(); root.addWidget(header)

        mid_h = self.screen_h - 2*self.outer_m - header.height() - int(64*self.s) - self.spacing
        total_w = self.screen_w - 2*self.outer_m
        left_w  = int(total_w * 0.66) - self.spacing//2
        right_w = total_w - left_w - self.spacing

        middle = QtWidgets.QHBoxLayout(); middle.setSpacing(self.spacing)
        self.live = LivePanel(left_w, mid_h, self.s); middle.addWidget(self.live)
        self.right = RightPanel(right_w, self.s);     middle.addWidget(self.right)
        root.addLayout(middle, 1)

        bottom = self.build_bottom_bar(); root.addWidget(bottom)
        self.apply_dark_palette()

        # ---------------------- Image or Camera ----------------------
        self.image_bgr = None
        self.is_processing = False

        if USE_IMAGE:
            if not os.path.exists(IMAGE_SOURCE):
                QtWidgets.QMessageBox.critical(self, "Image Mode", f"Image not found:\n{IMAGE_SOURCE}")
                logger.error(f"Image not found: {IMAGE_SOURCE}")
            else:
                self.image_bgr = cv2.imread(IMAGE_SOURCE)
                if self.image_bgr is None:
                    QtWidgets.QMessageBox.critical(self, "Image Mode", f"Failed to read image:\n{IMAGE_SOURCE}")
                    logger.error(f"Failed to read image: {IMAGE_SOURCE}")
                else:
                    self.image_bgr = resize_to_target(self.image_bgr)
                    self.image_timer = QtCore.QTimer(self, interval=250, timeout=self.update_image_feed)
                    self.image_timer.start()
                    self.on_camera_online(True)

        else:
            self.cam = CameraWorker(PIPELINE, self)
            self.cam.frame_qimage.connect(self.live.set_qimage)
            self.cam.camera_online.connect(self.on_camera_online)
            self.cam.start()

        QtWidgets.QShortcut(QtGui.QKeySequence("F1"), self, activated=self.handle_measure)

    def update_image_feed(self):
        if self.image_bgr is None:
            return
        frame = self.image_bgr.copy()
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        self.live.set_qimage(bgr_to_qimage(frame))

    # ---- Header ----
    def build_header(self):
        h = int(76*self.s)
        w = QtWidgets.QFrame(); w.setObjectName("header")
        w.setStyleSheet("QFrame#header{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #2c2c2c,stop:1 #3a3a3a);"
                        "border:1px solid #404040;border-radius:10px;}")
        w.setFixedHeight(h)
        l = QtWidgets.QHBoxLayout(w); l.setContentsMargins(int(14*self.s),int(10*self.s),int(14*self.s),int(10*self.s))

        title_box = QtWidgets.QVBoxLayout()
        t = QtWidgets.QLabel("MEASUREMENT SYSTEM"); t.setFont(sans(pt(15,self.s), True)); t.setStyleSheet("color:#fff;")
        title_box.addWidget(t);
        l.addLayout(title_box)
        l.addStretch(1)

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
        self.header_dot.setColor("#00ff00" if ok else "#ff3333")
        if ok: self.live.image_label.clearImage("")
    def update_clock(self):
        self.datetime_lbl.setText(datetime.now().strftime("%H:%M:%S • %Y-%m-%d"))

    # ---- Measurement ----
    def handle_measure(self):
        if self.is_processing: return
        self.is_processing = True
        try:
            if USE_IMAGE:
                if self.image_bgr is None:
                    QtWidgets.QMessageBox.warning(self, "Image Mode", "No image loaded.")
                    results, output_bgr = None, None
                else:
                    results, output_bgr = self.run_measurement_image(self.image_bgr.copy())
            else:
                if not hasattr(self, "cam") or self.cam.latest_frame is None:
                    QtWidgets.QMessageBox.warning(self, "Camera", "No camera frame available yet.")
                    results, output_bgr = None, None
                else:
                    results, output_bgr = self.run_measurement(self.cam.latest_frame.copy())
        except Exception as e:
            logger.exception("Measurement error")
            QtWidgets.QMessageBox.critical(self, "Error", f"Measurement failed:\n{e}")
            self.is_processing = False; return

        if results and output_bgr is not None:
            w = results["width_mm"]; h = results["height_mm"]; a = results["area_mm2"]
            cls_name = results["class_name"]; cls_color = results["class_color"]
            self.right.kpi_width.setValue(f"{w:.2f}")
            self.right.kpi_height.setValue(f"{h:.2f}")
            self.right.kpi_area.setValue(f"{a:.2f}")
            self.right.set_class(cls_name, cls_color)

            run = get_next_running_number()
            out_path = os.path.join(img_output_path, f"mtal{run}.png")
            cv2.imwrite(out_path, output_bgr)

            logger.info(f"MEASURE OK run={run} W={w:.2f}mm H={h:.2f}mm A={a:.2f}mm2 Class={cls_name}")

            t = datetime.now().strftime("%H:%M:%S")
            data = f"W:{w:.2f}  H:{h:.2f}  A:{a:.2f}  Class:{cls_name}"
            self.prepend_log_row([t, data, "✓"])
            self.set_proc_preview(output_bgr)
        else:
            t = datetime.now().strftime("%H:%M:%S")
            self.prepend_log_row([t, "No object detected", "⚠"])
            self.set_proc_preview(None)
            self.right.set_class("—", "#444444")
            logger.warning("MEASURE NO_OBJECT")

        self.is_processing = False

    # ---------------------- ฟังก์ชันประมวลผลฝั่งรูปภาพ ----------------------
    def run_measurement_image(self, image_bgr: np.ndarray):
        return self.run_measurement(image_bgr)

    def set_proc_preview(self, bgr):
        if bgr is None: self.right.proc_preview.clearImage("NO PROCESSED IMAGE"); return
        self.right.proc_preview.setImage(bgr_to_qimage(bgr))

    def run_measurement(self, frame_bgr):
        frame_HD = frame_bgr

        # 1) undistort + rotate + crop
        if self.mtx is not None and self.dist is not None:
            frame_HD = cv2.undistort(frame_HD, self.mtx, self.dist, None, self.mtx)
        frame_HD = rotate_image(frame_HD, ROTATE_ANGLE_DEG)
        yslice, xslice = CROP_SLICE
        frame_HD = frame_HD[yslice, xslice].copy()
        output = frame_HD.copy()

        # 2) glare removal
        gray = cv2.cvtColor(frame_HD, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, self.kernel)
        frame_no_glare = cv2.inpaint(frame_HD, closed, 3, cv2.INPAINT_TELEA)

        # 3) background masking
        hsv = cv2.cvtColor(frame_no_glare, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, blue_Lower, blue_Upper)
        obj_mask = cv2.bitwise_not(blue_mask)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        if cv2.countNonZero(obj_mask) <= 10000:
            return None, output

        # 4) external contour + holes
        ext_cnt, hole_cnts = get_external_and_holes(obj_mask)
        if ext_cnt is None:
            return None, output

        # 5) Area cal
        outer_area_px = cv2.contourArea(ext_cnt)
        holes_area_px = sum(cv2.contourArea(hc) for hc in hole_cnts)
        net_area_px = max(0.0, outer_area_px - holes_area_px)

        if net_area_px < 1000:
            return None, output

        # 6) arcLength + minAreaRect
        epsilon = 0.01 * cv2.arcLength(ext_cnt, True)
        approx = cv2.approxPolyDP(ext_cnt, epsilon, True)
        rect = cv2.minAreaRect(ext_cnt)  # ((cx,cy),(w,h),theta)
        (w_px, h_px) = rect[1]

        if w_px < h_px:
            w_px, h_px = h_px, w_px

        # 7) ratio
        width_mm  = float(w_px) * float(PIXEL_MM_RATIO_W)
        height_mm = float(h_px) * float(PIXEL_MM_RATIO_H)
        area_mm2  = float(net_area_px) * float(PIXEL_MM_RATIO_W) * float(PIXEL_MM_RATIO_H)

        # 8) class
        class_name, class_color = classify_shape(ext_cnt, approx, hole_cnts, net_area_px)

        # 9) drawContours
        cv2.drawContours(output, [ext_cnt], -1, (0, 255, 255), 2)
        # holes
        for hc in hole_cnts:
            cv2.drawContours(output, [hc], -1, (0, 0, 255), 2)

        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(output, [box], True, (0, 0, 0), 2)

        M = cv2.moments(ext_cnt)
        if M["m00"] != 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        else:
            cX, cY = 20, 20

        cv2.putText(output, f"Area:  {area_mm2:.1f} mm2", (cX - 140, cY - 90), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)
        cv2.putText(output, f"Width: {width_mm:.1f} mm",  (cX - 140, cY - 20), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)
        cv2.putText(output, f"Height:{height_mm:.1f} mm", (cX - 140, cY + 40), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)
        cv2.putText(output, f"Class: {class_name}",       (cX - 140, cY + 100), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)

        measurements = {
            "width_mm":   width_mm,
            "height_mm":  height_mm,
            "area_mm2":   area_mm2,
            "contour":    ext_cnt,
            "holes":      hole_cnts,
            "approx":     approx,
            "class_name": class_name,
            "class_color": class_color
        }
        return measurements, output

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
            if not USE_IMAGE and hasattr(self, "cam") and self.cam and self.cam.isRunning():
                self.cam.stop(); self.cam.wait(1500)
            if USE_IMAGE and hasattr(self, "image_timer"):
                self.image_timer.stop()
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
