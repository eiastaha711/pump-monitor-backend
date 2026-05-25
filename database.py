"""
database.py — SQLite setup using SQLAlchemy
All pump readings and fault events are stored here permanently.
"""

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# SQLite file — created automatically on first run
DATABASE_URL = "sqlite:///./pump_monitor.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Table 1: every reading from the ESP32 ──────────────────────────────────
class Reading(Base):
    __tablename__ = "readings"

    id          = Column(Integer, primary_key=True, index=True)
    pump_id     = Column(String, index=True)           # e.g. "pump_01"
    timestamp   = Column(DateTime, default=datetime.utcnow, index=True)
    status      = Column(String)                       # "healthy" / "warning" / "danger"
    fault_type  = Column(String, nullable=True)        # e.g. "filter_fault", None if healthy
    mic_rms     = Column(Float, nullable=True)         # raw feature values (optional)
    acc_rms     = Column(Float, nullable=True)
    health_score = Column(Float, nullable=True)        # 0.0 – 1.0 score from model


# ── Table 2: fault events (status changes only) ────────────────────────────
class FaultEvent(Base):
    __tablename__ = "fault_events"

    id          = Column(Integer, primary_key=True, index=True)
    pump_id     = Column(String, index=True)
    timestamp   = Column(DateTime, default=datetime.utcnow, index=True)
    status      = Column(String)                       # "warning" / "danger" / "healthy"
    description = Column(String)                       # human-readable description


# ── Table 3: pump registry ─────────────────────────────────────────────────
class Pump(Base):
    __tablename__ = "pumps"

    id          = Column(String, primary_key=True)     # "pump_01"
    name        = Column(String)                       # "Pump 01"
    location    = Column(String, nullable=True)        # optional label


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
