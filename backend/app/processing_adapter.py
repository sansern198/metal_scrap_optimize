import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any
from .utils import IMG_DIR, OUTPUT_DIR

# จุดเชื่อมต่อกับ mtl_processing.py ของคุณ
# คุณควรมีฟังก์ชันประมาณนี้ในไฟล์เดิม:
#   def run_measurement(image_bgr) -> Tuple[Dict[str, Any], np.ndarray]:
#       return results_dict, output_bgr
#
# โดยที่ results_dict ต้องอย่างน้อย: width_mm, height_mm, area_mm2
# และ output_bgr คือภาพ BGR ที่วาด overlay แล้ว (หรือจะส่งกลับภาพดิบก็ได้)
#
# ถ้าในโปรเจกต์จริงชื่อฟังก์ชันไม่ตรง ให้เปลี่ยนที่นี่จุดเดียว
import sys
sys.path.append("/app/work")  # ให้ import mtl_processing.py จากรูทโปรเจกต์
import mtl_processing as proc  # ใช้ของเดิม

def run_from_file(filename: str, rotate_deg: float = 0.0) -> Tuple[Dict[str, Any], "np.ndarray"]:
    src_path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Image not found: {src_path}")

    img = cv2.imread(src_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {src_path}")

    # เผื่อคุณต้องการหมุนก่อนเข้าพายป์ไลน์
    if abs(rotate_deg) > 1e-6:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), rotate_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # เรียกพายป์ไลน์จริง
    # โปรดแก้ชื่อฟังก์ชันให้ตรงกับของคุณ ถ้าไม่ใช่ run_measurement
    results, out_bgr = proc.run_measurement(img)  # ต้องมีใน mtl_processing.py

    # ตรวจค่าอย่างน้อย 3 ฟิลด์
    for k in ("width_mm", "height_mm", "area_mm2"):
        if k not in results:
            raise RuntimeError(f"Processing result missing field '{k}'")

    return results, out_bgr
