import threading
import time

try:
    import Jetson.GPIO as GPIO
    _gpio_ok = True
except Exception:
    _gpio_ok = False
    class _DummyGPIO:
        BOARD = BCM = OUT = HIGH = LOW = None
        def setmode(self, *a, **k): pass
        def setwarnings(self, *a, **k): pass
        def setup(self, *a, **k): pass
        def output(self, *a, **k): pass
        def cleanup(self, *a, **k): pass
    GPIO = _DummyGPIO()

# ======= ตั้งค่าพิน =======
# เขียว=11, เหลือง=13, แดง=15
GREEN_PIN  = 11
YELLOW_PIN = 13
RED_PIN    = 15

MODE = GPIO.BOARD
ACTIVE_HIGH = True  # ถ้ารีเลย์/โมดูลใช้ active-low ให้เปลี่ยนเป็น False

_initialized = False
_lock = threading.Lock()

def _lvl(on: bool):
    if not _gpio_ok:
        return None
    if ACTIVE_HIGH:
        return GPIO.HIGH if on else GPIO.LOW
    else:
        return GPIO.LOW if on else GPIO.HIGH

def init():
    global _initialized
    if _initialized:
        return
    if _gpio_ok:
        GPIO.setmode(MODE)
        GPIO.setwarnings(False)
        for p in (GREEN_PIN, YELLOW_PIN, RED_PIN):
            GPIO.setup(p, GPIO.OUT, initial=_lvl(False))
    _initialized = True
    set_idle()  # สถานะพร้อมทำงาน (ไฟเขียวติด)

def ensure_init():
    if not _initialized:
        init()

def cleanup():
    if _gpio_ok:
        GPIO.cleanup()

def _set(pin, on: bool):
    ensure_init()
    if _gpio_ok:
        GPIO.output(pin, _lvl(on))

def set_green(on: bool):
    _set(GREEN_PIN, on)

def set_yellow(on: bool):
    _set(YELLOW_PIN, on)

def set_red(on: bool):
    _set(RED_PIN, on)

def set_idle():
    with _lock:
        set_green(True)
        set_yellow(False)
        set_red(False)

def measure_start():
    with _lock:
        set_green(False)
        set_yellow(False)
        set_red(False)

def measure_success():
    set_idle()

def measure_fail():
    with _lock:
        set_green(False)
        set_yellow(False)
        set_red(True)

def pulse_yellow(duration_sec: float = 2.0):
    """ไฟเหลืองติดชั่วครู่เพื่อแจ้งเตือน (เช่น API fail)"""
    def _worker():
        with _lock:
            set_yellow(True)
        time.sleep(max(0.1, duration_sec))
        with _lock:
            set_yellow(False)
    threading.Thread(target=_worker, daemon=True).start()

def api_fail():
    """ใช้เมื่อส่ง API ไม่ผ่าน"""
    pulse_yellow(2.0)

def api_fail_latched():
    """แสดงสถานะ API ล้มเหลว: เขียว=OFF, แดง=OFF, เหลือง=ON (ค้าง)"""
    with _lock:
        set_green(False)
        set_red(False)
        set_yellow(True)

