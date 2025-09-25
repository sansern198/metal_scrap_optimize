# backend/app/main.py
import os
import json
from fastapi import FastAPI, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session

from .database import init_db, get_session, engine
from .models import Measurement
from .schemas import MeasurementOut
from .utils import OUTPUT_DIR

# ---------- FastAPI app ----------
app = FastAPI(title="MTL API", version="1.0.0")

@app.on_event("startup")
def on_startup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}

# โหลดไฟล์ output ผ่าน API
@app.get("/output/{filename}")
def get_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Output not found")
    return FileResponse(path)

# ---------- Admin (sqladmin) ----------
from sqladmin import Admin, ModelView
from markupsafe import Markup
import re

def format_output_filename(m):
    if not m.output_filename:
        return ""
    return Markup(f'<a href="/output/{m.output_filename}" target="_blank">{m.output_filename}</a>')

def _s_index(name: str) -> int:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 0

from markupsafe import Markup

def format_meta_S_short(m):
    """ใช้ใน list view: จำกัดความยาว"""
    s_list = []
    if m.meta and isinstance(m.meta, dict):
        s_list = m.meta.get("S", [])
    if not s_list:
        return ""

    parts = []
    for d in s_list:
        parts.append(
            f"{d.get('name','S')}(W={float(d.get('W_mm',0)):.1f},"
            f"H={float(d.get('H_mm',0)):.1f},"
            f"A={int(round(float(d.get('A_mm2',0))) )})"
        )
    text = "[" + " | ".join(parts) + "]"

    max_len = 100
    if len(text) > max_len:
        return Markup(text[:max_len] + "…")
    return Markup(text)


def format_meta_S_full(m):
    """ใช้ใน detail view: แสดงเต็ม"""
    s_list = []
    if m.meta and isinstance(m.meta, dict):
        s_list = m.meta.get("S", [])
    if not s_list:
        return ""

    parts = []
    for d in s_list:
        parts.append(
            f"{d.get('name','S')}(W={float(d.get('W_mm',0)):.1f},"
            f"H={float(d.get('H_mm',0)):.1f},"
            f"A={int(round(float(d.get('A_mm2',0))) )})"
        )
    return Markup("[" + " | ".join(parts) + "]")

from markupsafe import Markup

def format_output_filename(m):
    if not m.output_filename:
        return ""
    return Markup(f'<a href="/output/{m.output_filename}" target="_blank">{m.output_filename}</a>')

class MeasurementAdmin(ModelView, model=Measurement):
    column_list = [
        Measurement.id,
        Measurement.run_no,
        Measurement.output_filename,   # ต้องมีใน list
        Measurement.width_mm,
        Measurement.height_mm,
        Measurement.area_mm2,
        Measurement.type_name,
        Measurement.meta,
        Measurement.created_at,
    ]

    # List view
    column_formatters = {
        Measurement.meta: lambda m, a: format_meta_S_short(m),
        Measurement.output_filename: lambda m, a: format_output_filename(m),
    }

    # Detail view
    column_formatters_detail = {
        Measurement.meta: lambda m, a: format_meta_S_full(m),
        Measurement.output_filename: lambda m, a: format_output_filename(m),
    }

    form_excluded_columns = ["output_filename", "meta"] 

# ต้องสร้าง Admin หลังจากประกาศ class แล้วค่อย add_view
admin = Admin(app, engine, base_url="/admin")
admin.add_view(MeasurementAdmin)

# ---------- Ingest endpoint ----------
@app.post("/measurements/ingest", response_model=MeasurementOut)
def ingest_measurement(
    session: Session = Depends(get_session),
    file: UploadFile = File(...),                 # รูป output (มี overlay แล้ว)
    run_no: int = Form(...),                      # เลขรันนิ่งจาก UI
    width_mm: float = Form(...),
    height_mm: float = Form(...),
    area_mm2: float = Form(...),
    source_filename: str = Form(""),
    output_filename: str = Form(""),
    type_name: str = Form(""),                    # << รับ type จาก UI
    s_json: str = Form(None),                     # JSON ของ S1..SN (list/dict)
    meta_json: str = Form(None),                  # JSON อื่น ๆ เพิ่มเติม (optional)
):
    # 1) บันทึกไฟล์รูป output
    out_name = output_filename or f"MTL_{run_no:06d}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    content = file.file.read()
    with open(out_path, "wb") as f:
        f.write(content)

    # 2) รวม meta (ใส่ค่า S1..SN ไว้ใน meta["S"])
    meta = {}
    if meta_json:
        try:
            meta.update(json.loads(meta_json))
        except Exception:
            pass
    if s_json:
        try:
            meta["S"] = json.loads(s_json)
        except Exception:
            pass

    def _round_s_list(s_list):
        out = []
        for d in s_list:
            out.append({
                "name": d.get("name","S"),
                "W_mm": round(float(d.get("W_mm",0)), 1),
                "H_mm": round(float(d.get("H_mm",0)), 1),
                "A_mm2": round(float(d.get("A_mm2",0)), 0),
            })
        return out

    if s_json:
        try:
            s_data = json.loads(s_json)
            if isinstance(s_data, list):
                meta["S"] = _round_s_list(s_data)
            else:
                meta["S"] = s_data
        except Exception:
            pass

    # 3) ปัดทศนิยม (width/height ให้ 2 ตำแหน่ง)
    width_mm = round(float(width_mm), 2)
    height_mm = round(float(height_mm), 2)
    # area จะเก็บเต็ม หรือจะปัดก็ได้:
    # area_mm2 = round(float(area_mm2), 2)
    area_mm2 = float(area_mm2)

    # 4) บันทึกฐานข้อมูล
    m = Measurement(
        run_no=int(run_no),
        source_filename=source_filename or "",
        output_filename=out_name,
        width_mm=width_mm,
        height_mm=height_mm,
        area_mm2=area_mm2,
        type_name=type_name or None,   # << ใช้ค่าที่รับมา
        meta=meta or None,
    )
    session.add(m)
    session.commit()
    session.refresh(m)

    # 5) ตอบกลับ
    return MeasurementOut(
        id=m.id,
        run_no=m.run_no,
        source_filename=m.source_filename,
        output_filename=m.output_filename,
        width_mm=m.width_mm,
        height_mm=m.height_mm,
        area_mm2=m.area_mm2,
        type_name=m.type_name,   # << ส่งกลับด้วย
        meta=m.meta,
    )
