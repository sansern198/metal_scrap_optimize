# -*- coding: utf-8 -*-
"""
mtl_core.py
- Core module for camera handling and measurement pipeline
- Designed to be imported by UI or other apps
"""

import os
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# ---------------------- Paths & Config ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNING_NUMBER_PATH = os.path.join(SCRIPT_DIR, 'running_number.txt')
CALIB_PATH = os.path.join(SCRIPT_DIR, 'calibration_data/calibration_data_7x5_3040p.npz')
IMG_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

# Camera pipeline (edit to your need)
PIPELINE = (
    "souphttpsrc location=http://192.168.1.106:4747/video ! "
    "multipartdemux ! jpegdec ! videoconvert ! appsink"
)

# HSV mask range (blue background)
BLUE_LOWER = (90, 10, 50)
BLUE_UPPER = (150, 255, 255)

# Pixel to mm ratio
PIXEL_MM_RATIO_W = 0.5
PIXEL_MM_RATIO_H = 0.5

# Rotate + crop as your original code
ROTATE_ANGLE_DEG = 2.4
CROP_SLICE = (slice(340, 2420), slice(100, 4000))  # y, x

# Non-zero threshold for object present
NONZERO_THRESHOLD = 500_000

# ---------------------- Logger ----------------------
LOG_PATH = os.path.join(SCRIPT_DIR, 'app.log')
logger = logging.getLogger('MTL_MEASUREMENT')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=10*1024*1024, backupCount=5, mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)


# ---------------------- Utils ----------------------
def rotate_image(image, angle_deg: float):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def get_next_running_number(filename=RUNNING_NUMBER_PATH):
    try:
        with open(filename, "r") as f:
            n = int(f.read().strip()) + 1
    except Exception:
        n = 1
    if n > 9999:
        n = 1
    try:
        with open(filename, "w") as f:
            f.write(str(n))
    except Exception as e:
        logger.error(f"Failed to write running number: {e}")
    return format(n, '04d')

def load_calibration(path=CALIB_PATH):
    try:
        data = np.load(path)
        mtx, dist = data['mtx'], data['dist']
        logger.debug(f'Calibration loaded from {path}')
        return mtx, dist
    except Exception as e:
        logger.error(f'Failed to load calibration data: {e}')
        return None, None


def cal_dim(contours, hierarchy, px2mm_x, px2mm_y):
    """Compute area with holes and bounding min-rect (largest outer contour)."""
    if hierarchy is None:
        return None

    total_area_px = 0
    max_rect, max_cnt = None, None

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
    (_, _), (w, h), angle = max_rect
    width_mm = w * px2mm_x
    height_mm = h * px2mm_y

    return {
        "area_mm2": area_mm2,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "angle": angle,
        "contour": max_cnt
    }


# ---------------------- Camera Manager ----------------------
class CameraManager:
    """Thin wrapper around OpenCV VideoCapture for GStreamer pipeline."""
    def __init__(self, pipeline: str = PIPELINE):
        self.pipeline = pipeline
        self.cap = None
        self.online = False

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.online = self.cap.isOpened()
        if not self.online:
            logger.error("Unable to open the camera using GStreamer pipeline.")
        return self.online

    def read(self):
        """Return BGR frame or None."""
        if not self.cap:
            return None
        ok, frame = self.cap.read()
        if not ok:
            self.online = False
            return None
        return frame

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.online = False


# ---------------------- Measurement Pipeline ----------------------
class MeasurementPipeline:
    def __init__(self, mtx=None, dist=None):
        self.mtx = mtx
        self.dist = dist
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def process_frame(self, frame_bgr):
        """
        Returns (result_dict or None, output_bgr)
        result_dict keys: width_mm, height_mm, area_mm2, angle
        """
        frame_HD = frame_bgr

        # 1) Undistort (if calibration available)
        if self.mtx is not None and self.dist is not None:
            frame_HD = cv2.undistort(frame_HD, self.mtx, self.dist, None, self.mtx)

        # 2) Rotate
        frame_HD = rotate_image(frame_HD, ROTATE_ANGLE_DEG)

        # 3) Crop
        yslice, xslice = CROP_SLICE
        frame_HD = frame_HD[yslice, xslice].copy()

        # Keep a copy for drawing
        output = frame_HD.copy()

        # 4) Glare removal via threshold + close + inpaint
        gray = cv2.cvtColor(frame_HD, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, self.kernel)
        frame_no_glare = cv2.inpaint(frame_HD, closed, 3, cv2.INPAINT_TELEA)

        # 5) HSV mask (blue background inverted)
        hsv = cv2.cvtColor(frame_no_glare, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        blue_mask = cv2.bitwise_not(blue_mask)

        nonZero_blue = cv2.countNonZero(blue_mask)
        logger.debug(f'Object check nonZero_blue: {nonZero_blue}')
        if nonZero_blue <= NONZERO_THRESHOLD:
            return None, output

        # 6) Contours & dimensions
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 0, 0), 1)

        meas = cal_dim(contours, hierarchy, PIXEL_MM_RATIO_W, PIXEL_MM_RATIO_H)
        if not meas:
            return None, output

        w_mm = meas["width_mm"]
        h_mm = meas["height_mm"]
        a_mm2 = meas["area_mm2"]
        ang = meas["angle"]
        cnt = meas["contour"]

        # 7) Annotate
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        else:
            cX, cY = 20, 20

        cv2.drawContours(output, [cnt], 0, (0, 255, 0), 2)
        cv2.putText(output, f"Area: {a_mm2:.1f} mm2", (cX - 120, cY - 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(output, f"Width: {w_mm:.1f} mm", (cX - 120, cY - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(output, f"Height:{h_mm:.1f} mm", (cX - 120, cY + 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        return {
            "width_mm": w_mm,
            "height_mm": h_mm,
            "area_mm2": a_mm2,
            "angle": ang
        }, output

    def save_output(self, image_bgr):
        run = get_next_running_number()
        out_path = os.path.join(IMG_OUTPUT_DIR, f"mtal{run}.png")
        cv2.imwrite(out_path, image_bgr)
        return out_path


# ---------------------- Factory helpers ----------------------
def create_default_pipeline():
    mtx, dist = load_calibration(CALIB_PATH)
    return MeasurementPipeline(mtx, dist)

def create_camera(pipeline: str = PIPELINE):
    cam = CameraManager(pipeline)
    cam.start()
    return cam
