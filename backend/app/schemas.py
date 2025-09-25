# backend/app/schemas.py
from typing import Optional
from pydantic import BaseModel

class CreateFromFileReq(BaseModel):
    filename: str
    rotate_deg: float = 0.0
    save_overlay: bool = True

class MeasurementOut(BaseModel):
    id: int
    run_no: int
    source_filename: str
    output_filename: str
    width_mm: float
    height_mm: float
    area_mm2: float
    type_name: Optional[str] = None
    meta: Optional[dict] = None
