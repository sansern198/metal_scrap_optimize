import os
from sqlmodel import SQLModel, create_engine, Session

DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://mtl:mtlpass@localhost:5432/mtl")
engine = create_engine(DB_URL, echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
