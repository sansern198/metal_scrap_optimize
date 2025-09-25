from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB  # ใช้กับ Postgres

class Measurement(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_no: int
    source_filename: str
    output_filename: str
    width_mm: float
    height_mm: float
    area_mm2: float
    type_name: Optional[str] = Field(default=None, index=True)
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=lambda: datetime.now().replace(microsecond=0))

