"""
main.py — FastAPI backend for Pump Health Monitor

INSTALL DEPENDENCIES:
    pip install fastapi uvicorn sqlalchemy

RUN:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

ENDPOINTS:
    POST /ingest                     ← ESP32 sends data here
    GET  /pumps                      ← list all pumps + current status
    GET  /pumps/{pump_id}/history    ← 7-day hourly timeline
    GET  /pumps/{pump_id}/events     ← recent fault events
    GET  /pumps/{pump_id}/signal     ← last N seconds of raw signal values
    GET  /docs                       ← interactive API docs (auto-generated)
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

from database import get_db, create_tables, Reading, FaultEvent, Pump
from model import load_model, predict

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Pump Health Monitor", version="1.0")

# Allow the HTML frontend to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()
    load_model()
    # Register default pumps if they don't exist yet
    db = next(get_db())
    for pid, name in [("pump_01", "Pump 01"), ("pump_02", "Pump 02"), ("pump_03", "Pump 03")]:
        if not db.query(Pump).filter(Pump.id == pid).first():
            db.add(Pump(id=pid, name=name))
    db.commit()
    db.close()
    print("[startup] Tables ready, model loaded, pumps registered.")


# ── Ingest schema (what ESP32 sends) ───────────────────────────────────────
class SensorPayload(BaseModel):
    pump_id:    str            # "pump_01"
    mic_rms:    float          # microphone RMS amplitude
    mic_peak:   Optional[float] = None
    mic_crest:  Optional[float] = None
    mic_kurtosis: Optional[float] = None
    acc_rms:    float          # accelerometer RMS
    acc_peak:   Optional[float] = None
    acc_crest:  Optional[float] = None
    acc_kurtosis: Optional[float] = None
    mic_fft_dominant: Optional[float] = None
    acc_fft_dominant: Optional[float] = None


# ── POST /ingest — called by ESP32 every second ────────────────────────────
@app.post("/ingest")
def ingest(payload: SensorPayload, db: Session = Depends(get_db)):
    """
    Receives sensor data from ESP32, runs ML prediction, stores result.
    ESP32 sends a JSON body matching SensorPayload.
    """
    features = payload.dict()
    result = predict(features)

    # Save reading
    reading = Reading(
        pump_id      = payload.pump_id,
        status       = result["status"],
        fault_type   = result["label"] if result["label"] != "healthy" else None,
        mic_rms      = payload.mic_rms,
        acc_rms      = payload.acc_rms,
        health_score = result["health_score"],
    )
    db.add(reading)

    # Save fault event only when status CHANGES
    last = (
        db.query(Reading)
        .filter(Reading.pump_id == payload.pump_id)
        .order_by(Reading.timestamp.desc())
        .offset(1)   # second-to-last reading
        .first()
    )
    if last is None or last.status != result["status"]:
        event = FaultEvent(
            pump_id     = payload.pump_id,
            status      = result["status"],
            description = result["description"],
        )
        db.add(event)

    db.commit()
    return {"status": "ok", "prediction": result}


# ── GET /pumps — list all pumps with current status ────────────────────────
@app.get("/pumps")
def list_pumps(db: Session = Depends(get_db)):
    pumps = db.query(Pump).all()
    result = []
    for pump in pumps:
        latest = (
            db.query(Reading)
            .filter(Reading.pump_id == pump.id)
            .order_by(Reading.timestamp.desc())
            .first()
        )
        # Count fault events in last 7 days
        since = datetime.utcnow() - timedelta(days=7)
        faults_7d = (
            db.query(FaultEvent)
            .filter(FaultEvent.pump_id == pump.id, FaultEvent.timestamp >= since,
                    FaultEvent.status != "healthy")
            .count()
        )
        result.append({
            "id":       pump.id,
            "name":     pump.name,
            "status":   latest.status if latest else "healthy",
            "fault":    latest.fault_type or "No faults detected" if latest else "No data yet",
            "faults_7d": faults_7d,
        })
    return result


# ── GET /pumps/{id}/history — 7-day hourly timeline ────────────────────────
@app.get("/pumps/{pump_id}/history")
def pump_history(pump_id: str, days: int = 7, db: Session = Depends(get_db)):
    """
    Returns one row per day, each with 24 blocks (one per hour).
    Block value: "healthy" / "warning" / "danger" / "empty"
    """
    since = datetime.utcnow() - timedelta(days=days)
    readings = (
        db.query(Reading)
        .filter(Reading.pump_id == pump_id, Reading.timestamp >= since)
        .order_by(Reading.timestamp.asc())
        .all()
    )

    # Bucket readings into (day_offset, hour) slots
    # day 0 = oldest, day N-1 = today
    now = datetime.utcnow()
    day_labels = []
    for i in range(days - 1, -1, -1):
        d = now - timedelta(days=i)
        day_labels.append("Today" if i == 0 else d.strftime("%a"))

    # Build a map: (day_offset, hour) → worst status in that slot
    STATUS_RANK = {"danger": 3, "warning": 2, "healthy": 1, "empty": 0}
    slots = defaultdict(lambda: "empty")
    for r in readings:
        delta_days = (now.date() - r.timestamp.date()).days
        day_idx = days - 1 - delta_days
        if 0 <= day_idx < days:
            hour = r.timestamp.hour
            key = (day_idx, hour)
            if STATUS_RANK.get(r.status, 0) > STATUS_RANK.get(slots[key], 0):
                slots[key] = r.status

    result = []
    for day_idx, label in enumerate(day_labels):
        blocks = []
        for hour in range(24):
            # Mark future hours today as "empty"
            if label == "Today" and hour > now.hour:
                blocks.append("empty")
            else:
                blocks.append(slots.get((day_idx, hour), "empty"))
        result.append({"label": label, "blocks": blocks})

    return result


# ── GET /pumps/{id}/events — recent fault events ───────────────────────────
@app.get("/pumps/{pump_id}/events")
def pump_events(pump_id: str, limit: int = 10, db: Session = Depends(get_db)):
    events = (
        db.query(FaultEvent)
        .filter(FaultEvent.pump_id == pump_id)
        .order_by(FaultEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    result = []
    for e in events:
        # Format timestamp nicely
        now = datetime.utcnow()
        delta = now - e.timestamp
        if delta.days == 0:
            ts = f"Today, {e.timestamp.strftime('%H:%M')}"
        elif delta.days == 1:
            ts = f"Yesterday, {e.timestamp.strftime('%H:%M')}"
        else:
            ts = e.timestamp.strftime("%a %d %b, %H:%M")

        result.append({
            "status":      e.status,
            "description": e.description,
            "timestamp":   ts,
        })
    return result


# ── GET /pumps/{id}/signal — last N seconds of raw values ─────────────────
@app.get("/pumps/{pump_id}/signal")
def pump_signal(pump_id: str, seconds: int = 60, db: Session = Depends(get_db)):
    """
    Returns the last `seconds` readings — used for the live signal chart.
    """
    since = datetime.utcnow() - timedelta(seconds=seconds)
    readings = (
        db.query(Reading)
        .filter(Reading.pump_id == pump_id, Reading.timestamp >= since)
        .order_by(Reading.timestamp.asc())
        .all()
    )

    labels, mic, acc, score = [], [], [], []
    for r in readings:
        labels.append(r.timestamp.strftime("%H:%M:%S"))
        mic.append(round(r.mic_rms or 0.0, 3))
        acc.append(round(r.acc_rms or 0.0, 3))
        score.append(round(r.health_score or 0.0, 3))

    return {"labels": labels, "mic": mic, "acc": acc, "score": score}


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}
