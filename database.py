"""
Database configuration and ORM models for the pump monitoring system.
This module manages persistent storage for pump readings, fault events, and registered pumps using SQLite and SQLAlchemy.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///./pump_monitor.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class Reading(Base):
    __tablename__ = "readings"

    id = Column(Integer, primary_key=True, index=True)
    pump_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String)
    fault_type = Column(String, nullable=True)
    mic_rms = Column(Float, nullable=True)
    acc_rms = Column(Float, nullable=True)
    acc_x_rms = Column(Float, nullable=True)
    acc_y_rms = Column(Float, nullable=True)
    acc_z_rms = Column(Float, nullable=True)
    health_score = Column(Float, nullable=True)


class FaultEvent(Base):
    __tablename__ = "fault_events"

    id = Column(Integer, primary_key=True, index=True)
    pump_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String)
    description = Column(String)


class Pump(Base):
    __tablename__ = "pumps"

    id = Column(String, primary_key=True)
    name = Column(String)
    location = Column(String, nullable=True)


def create_tables():
    """
    Creates all database tables defined by the SQLAlchemy models.
    Existing tables are preserved, while any missing tables are created automatically.
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Provides a database session for each FastAPI request.
    The session is automatically closed after the request has finished processing.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
