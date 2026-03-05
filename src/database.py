from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./titanic.db")

# SQLite needs check_same_thread=False for FastAPI
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # ← required for SQLite
)

# ORM for interaction with database
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Create Base class for declaration of table
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from src.models import PredictionRecord
    Base.metadata.create_all(bind=engine)