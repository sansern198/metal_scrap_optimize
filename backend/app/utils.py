import os
import threading

LOG_FILE = os.getenv("LOG_FILE", "/app/work/app.log")
IMG_DIR = os.getenv("IMG_DIR", "/app/work/img")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/work/output")
RUNNING_NO_FILE = os.getenv("RUNNING_NO_FILE", "/app/work/running_number.txt")
CALIB_DIR = os.getenv("CALIB_DIR", "/app/work/calibration_data")

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
