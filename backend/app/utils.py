import os
import threading

# หา path ของโปรเจค (โฟลเดอร์ที่ไฟล์นี้อยู่)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(BASE_DIR, "work")

# ใช้ environment variable ถ้ามี, ถ้าไม่มีให้ใช้ path ในโฟลเดอร์โปรเจค
LOG_FILE = os.getenv("LOG_FILE", os.path.join(WORK_DIR, "app.log"))
IMG_DIR = os.getenv("IMG_DIR", os.path.join(WORK_DIR, "img"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(WORK_DIR, "output"))
RUNNING_NO_FILE = os.getenv("RUNNING_NO_FILE", os.path.join(WORK_DIR, "running_number.txt"))
CALIB_DIR = os.getenv("CALIB_DIR", os.path.join(WORK_DIR, "calibration_data"))

_running_no_lock = threading.Lock()

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def read_running_number() -> int:
    if not os.path.exists(RUNNING_NO_FILE):
        with open(RUNNING_NO_FILE, "w", encoding="utf-8") as f:
            f.write("1")
        return 1
    with open(RUNNING_NO_FILE, "r", encoding="utf-8") as f:
        return int(f.read().strip() or "1")

def get_next_running_number() -> int:
    with _running_no_lock:
        current = read_running_number()
        nxt = current + 1
        with open(RUNNING_NO_FILE, "w", encoding="utf-8") as f:
            f.write(str(nxt))
        return current
    
# try:
#     import mtl_gpio as _gpio
#     _HAS_GPIO = True
# except Exception:
#     _HAS_GPIO = False
#     _gpio = None

# def signal_api_fail(duration_sec: float = 2.0):
#     """
#     เรียกฟังก์ชันนี้เมื่อ 'ส่ง API ไม่ผ่าน' เพื่อให้ไฟเหลืองติดชั่วครู่
#     ใช้ pulse (ไม่ค้าง) และไม่พังบนเครื่องที่ไม่มี Jetson.GPIO
#     """
#     if _HAS_GPIO and hasattr(_gpio, "api_fail"):
#         try:
#             _gpio.api_fail()  # ภายในจะ pulse ไฟเหลือง ~2s
#         except Exception:
#             pass  # กันพังทุกกรณี
#     # ถ้าไม่มี GPIO ก็เงียบ ๆ ไป (หรือจะ print/log ก็ได้)
#     # ตัวอย่าง: print("[WARN] API failed (yellow light pulse).")

# def handle_api_response(ok: bool, duration_sec: float = 2.0):
#     """
#     Helper สำหรับโค้ดที่เรียก API:
#         ok=True  -> ไม่ทำอะไร
#         ok=False -> pulse ไฟเหลือง
#     """
#     if not ok:
#         signal_api_fail(duration_sec)

# # (ทางเลือก) ใช้เป็น context manager หากอยากให้ 'ข้อผิดพลาดระดับ Exception'
# # ระหว่างส่ง API ก็ pulse ไฟเหลืองให้ด้วยอัตโนมัติ
# from contextlib import contextmanager

# @contextmanager
# def api_fail_yellow_guard(duration_sec: float = 2.0):
#     try:
#         yield
#     except Exception:
#         signal_api_fail(duration_sec)
#         raise
