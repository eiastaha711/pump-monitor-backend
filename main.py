"""
main.py — FastAPI backend for Pump Health Monitor

INSTALL DEPENDENCIES:
    pip install fastapi uvicorn sqlalchemy numpy

RUN:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

ENDPOINTS:
    POST /ingest                     ← ESP32 sends features here (old path)
    POST /pumps/{pump_id}/raw        ← ESP32 sends raw binary samples (WiFi path)
    POST /pumps/{pump_id}/fft        ← pump_fft_analyzer.py sends FFT here
    GET  /pumps                      ← list all pumps + current status
    GET  /pumps/{pump_id}/history    ← 7-day hourly timeline
    GET  /pumps/{pump_id}/events     ← recent fault events
    GET  /pumps/{pump_id}/signal     ← last N seconds of raw signal values
    GET  /pumps/{pump_id}/fft        ← latest FFT data (4 channels)
    GET  /docs                       ← interactive API docs (auto-generated)
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
from collections import defaultdict
import numpy as np
import struct

from database import get_db, create_tables, Reading, FaultEvent, Pump, engine
from model import load_model, predict
from features import extract_features, FEATURE_NAMES

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Pump Health Monitor", version="1.3")

# Allow the HTML frontend to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory FFT storage (latest snapshot per pump) ───────────────────────
# Stores: { "pump_01": { freq_acc: [...], mag_x: [...], ... , timestamp: "..." } }
latest_fft = {}

# ── Data collection state ─────────────────────────────────────────────────
import csv, os, pickle

class CollectionState:
    def __init__(self):
        self.label      = "healthy"      # current label being recorded
        self.collecting  = False          # whether logging is active
        self.log_dir     = "collected_data"
        self.csv_path    = None
        self.csv_writer  = None
        self.csv_file    = None
        self.frame_count = 0

    def start(self, label: str):
        self.stop()  # close any open file
        self.label = label
        self.collecting = True
        self.frame_count = 0
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f"{label}_{timestamp}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        header = ["timestamp", "pump_id", "label"] + FEATURE_NAMES
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=header)
        self.csv_writer.writeheader()
        print(f"[Collection] Started: label='{label}' → {self.csv_path}")

    def log_frame(self, pump_id: str, features: dict):
        if not self.collecting or not self.csv_writer:
            return
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "pump_id":   pump_id,
            "label":     self.label,
        }
        row.update(features)
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.frame_count += 1

    def stop(self):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print(f"[Collection] Stopped: {self.frame_count} frames saved to {self.csv_path}")
        self.collecting = False
        self.csv_writer = None
        self.csv_file = None

collection = CollectionState()

# ── Anomaly model (loaded from pump_anomaly_model.pkl if it exists) ────────
ANOMALY_MODEL_PATH = "pump_anomaly_model.pkl"
anomaly_model = None
anomaly_scaler = None
anomaly_calibration = None
fault_classifier = None
fault_clf_scaler = None


def load_anomaly_model():
    """Load the trained Isolation Forest + classifier from disk."""
    global anomaly_model, anomaly_scaler, anomaly_calibration
    global fault_classifier, fault_clf_scaler
    if os.path.exists(ANOMALY_MODEL_PATH):
        with open(ANOMALY_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        anomaly_model       = data["model"]
        anomaly_scaler      = data["scaler"]
        anomaly_calibration = data["calibration"]
        fault_classifier    = data.get("classifier")
        fault_clf_scaler    = data.get("clf_scaler")
        info = data.get("train_info", {})
        print(f"[anomaly] Loaded model: {info.get('healthy_frames', '?')} healthy frames, "
              f"{info.get('n_features', '?')} features")
        if fault_classifier is not None:
            print(f"[classifier] Loaded: classes={data.get('clf_classes', '?')}, "
                  f"accuracy={info.get('cv_accuracy', '?')}")
        else:
            print(f"[classifier] Not found in model file — using rule-based fallback")
    else:
        print(f"[anomaly] {ANOMALY_MODEL_PATH} not found — anomaly scoring disabled. "
              f"Train with: python train_model.py")


def score_anomaly(features_dict):
    """
    Score a feature vector using the trained Isolation Forest.

    Returns:
        anomaly_score (float): 0 = normal, 100 = extremely anomalous
        tier (str): "healthy" / "warning" / "fault"
    """
    if anomaly_model is None or anomaly_scaler is None:
        return None, "unknown"

    from features import FEATURE_NAMES, features_to_vector
    vec = features_to_vector(features_dict).reshape(1, -1)
    vec_scaled = anomaly_scaler.transform(vec)

    raw_score = float(anomaly_model.score_samples(vec_scaled)[0])

    # Normalize: score_max → 0, score_min → 30, below → >30
    cal = anomaly_calibration
    span = cal["score_max"] - cal["score_min"]
    if span < 1e-9:
        normalized = 0.0
    else:
        normalized = (1.0 - (raw_score - cal["score_min"]) / span) * 30.0
    normalized = max(0.0, min(100.0, normalized))

    # 3-tier classification
    if normalized >= 60:
        tier = "fault"
    elif normalized >= 30:
        tier = "warning"
    else:
        tier = "healthy"

    return round(normalized, 1), tier


# ── Fault severity thresholds (based on ISO 10816-7 / API 610) ─────────
# For each fault: which feature to check, and the risk threshold.
#   - ML detects fault → warning (early detection)
#   - Feature value >= threshold → danger (risk level per standards)
#
# Thresholds derived from:
#   ISO 10816-7 vibration severity zones for centrifugal pumps
#   ProPump Services / Power-MI reference guidelines
#
# These are calibrated during training from collected data.
# Fallback values below if no calibration exists.
FAULT_THRESHOLDS = {
    "imbalance": {
        "feature": "f1_peak_radial",          # 1X peak amplitude (g)
        "threshold": 0.04,                     # ~2.5 mm/s @ 24 Hz → ISO Zone B/C
        "unit": "g",
        "desc_warning": "Imbalance detected (early) — monitor closely",
        "desc_danger":  "Imbalance — risk level, action required",
    },
    "looseness": {
        "feature": "f05_peak_radial",          # 0.5X sub-harmonic (g)
        "threshold": 0.02,                     # sub-harmonic presence
        "secondary": "f3_peak_radial",         # 3X harmonic
        "sec_threshold": 0.02,
        "unit": "g",
        "desc_warning": "Looseness detected (early) — check mounting bolts",
        "desc_danger":  "Structural looseness — risk level, tighten immediately",
    },
    "no_water": {
        "feature": "broadband_mic",            # broadband acoustic energy
        "threshold": 0.035,                    # high broadband = cavitation/dry run
        "unit": "RMS",
        "desc_warning": "No water suspected (early) — check supply valve",
        "desc_danger":  "No water / dry run — risk level, stop pump immediately",
    },
}

# Loaded from training data (calibrated thresholds)
trained_thresholds = None


def load_thresholds_from_model():
    """Load calibrated thresholds from the trained model if available."""
    global trained_thresholds
    if os.path.exists(ANOMALY_MODEL_PATH):
        with open(ANOMALY_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        trained_thresholds = data.get("fault_thresholds")
        if trained_thresholds:
            print(f"[thresholds] Loaded calibrated thresholds from training data")
            for fault, info in trained_thresholds.items():
                print(f"  {fault}: {info['feature']} >= {info['threshold']:.4f} → danger")


def classify_fault_ml(features_dict):
    """
    Classify fault type using Random Forest, then assess severity
    by comparing the relevant feature against ISO-based thresholds.

    Flow:
      1. ML predicts: healthy / imbalance / looseness / no_water
      2. If fault → check feature value against threshold
         - Below threshold → warning (ML caught it early)
         - At/above threshold → danger (risk level per standards)

    Returns:
        label, confidence, status, description
    """
    if fault_classifier is None or fault_clf_scaler is None:
        return None, None, None, None

    from features import FEATURE_NAMES, features_to_vector
    vec = features_to_vector(features_dict).reshape(1, -1)
    vec_scaled = fault_clf_scaler.transform(vec)

    label = fault_classifier.predict(vec_scaled)[0]
    proba = fault_classifier.predict_proba(vec_scaled)[0]
    confidence = float(proba.max())

    if label == "healthy":
        return label, confidence, "healthy", "No faults detected"

    # ── Fault detected — check severity against thresholds ────────
    thresholds = trained_thresholds or FAULT_THRESHOLDS
    fault_info = thresholds.get(label, {})
    feature_name = fault_info.get("feature")

    if feature_name and feature_name in features_dict:
        value = features_dict[feature_name]
        threshold = fault_info.get("threshold", 0)

        # Also check secondary feature (for looseness: 3X)
        sec_feature = fault_info.get("secondary")
        sec_exceeds = False
        if sec_feature and sec_feature in features_dict:
            sec_val = features_dict[sec_feature]
            sec_thresh = fault_info.get("sec_threshold", 0)
            sec_exceeds = sec_val >= sec_thresh

        if value >= threshold or sec_exceeds:
            status = "danger"
            description = fault_info.get("desc_danger",
                f"{label} — risk level ({feature_name}={value:.4f} >= {threshold})")
        else:
            status = "warning"
            description = fault_info.get("desc_warning",
                f"{label} — early detection ({feature_name}={value:.4f} < {threshold})")
    else:
        # No threshold defined — use confidence
        status = "danger" if confidence > 0.8 else "warning"
        description = f"{label} detected ({confidence*100:.0f}% confidence)"

    return label, confidence, status, description


@app.on_event("startup")
def startup():
    create_tables()

    # ── Auto-migrate: add per-axis acc columns if missing (schema v2) ──
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    existing_cols = {c["name"] for c in inspector.get_columns("readings")}
    with engine.connect() as conn:
        for col in ["acc_x_rms", "acc_y_rms", "acc_z_rms"]:
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE readings ADD COLUMN {col} FLOAT"))
                print(f"[migrate] Added column readings.{col}")
        conn.commit()

    load_model()
    load_anomaly_model()
    load_thresholds_from_model()
    # Register default pumps if they don't exist yet
    db = next(get_db())
    for pid, name in [("pump_01", "Pump 01"), ("pump_02", "Pump 02"), ("pump_03", "Pump 03")]:
        if not db.query(Pump).filter(Pump.id == pid).first():
            db.add(Pump(id=pid, name=name))
    db.commit()
    db.close()
    print("[startup] Tables ready, models loaded, pumps registered.")


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


# ── FFT schema (what pump_fft_analyzer.py sends) ──────────────────────────
class FFTPayload(BaseModel):
    pump_id:   str
    freq_acc:  List[float]    # 257 bins, 0–2000 Hz
    mag_x:     List[float]    # Acc X magnitude (g)
    mag_y:     List[float]    # Acc Y magnitude (g)
    mag_z:     List[float]    # Acc Z magnitude (g)
    freq_mic:  List[float]    # 513 bins, 0–4000 Hz
    mag_mic:   List[float]    # Mic magnitude
    roll:      Optional[float] = 0.0    # sensor roll in degrees
    pitch:     Optional[float] = 0.0    # sensor pitch in degrees
    faults:    Optional[list]  = None   # [{name, conf, desc}, ...]


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


# ── POST /pumps/{id}/fft — receive FFT snapshot from analyzer ─────────────
@app.post("/pumps/{pump_id}/fft")
def post_fft(pump_id: str, payload: FFTPayload):
    """
    Receives the latest FFT data (4 channels) from pump_fft_analyzer.py.
    Stored in memory only (not DB) — overwritten each frame.
    """
    latest_fft[pump_id] = {
        "freq_acc": payload.freq_acc,
        "mag_x":    payload.mag_x,
        "mag_y":    payload.mag_y,
        "mag_z":    payload.mag_z,
        "freq_mic": payload.freq_mic,
        "mag_mic":  payload.mag_mic,
        "roll":     payload.roll,
        "pitch":    payload.pitch,
        "faults":   payload.faults or [],
        "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
    }
    return {"status": "ok"}


# ── GET /pumps/{id}/fft — serve latest FFT to frontend ────────────────────
@app.get("/pumps/{pump_id}/fft")
def get_fft(pump_id: str):
    """
    Returns the latest FFT snapshot for the given pump.
    If no data has been posted yet, returns empty arrays.
    """
    if pump_id in latest_fft:
        return latest_fft[pump_id]
    return {
        "freq_acc": [], "mag_x": [], "mag_y": [], "mag_z": [],
        "freq_mic": [], "mag_mic": [],
        "roll": 0.0, "pitch": 0.0, "faults": [],
        "timestamp": None,
        "anomaly_score": None, "anomaly_tier": None,
    }


# ── DSP constants (must match firmware v2.1) ──────────────────────────────
ACC_N       = 512
ACC_FS      = 800
ACC_SCALE   = 4.0 / 2048.0     # raw int16 → g  (±4g, 12-bit)
MIC_N       = 1024
MIC_FS      = 8000
RPM_DEFAULT = 1450
BLADES_DEFAULT = 6
PAYLOAD_SIZE = ACC_N * 3 * 2 + MIC_N * 2   # 5120 bytes


def _compute_fft_acc(signal):
    sig = signal.astype(np.float64)
    sig -= sig.mean()
    win = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * win)) / (len(sig) / 2)
    freq = np.fft.rfftfreq(len(sig), 1.0 / ACC_FS)
    return freq, mag


def _compute_fft_mic(signal):
    sig = signal.astype(np.float32)
    sig -= sig.mean()
    win = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * win)) / (MIC_N / 2)
    freq = np.fft.rfftfreq(len(sig), 1.0 / MIC_FS)
    return freq, mag


def _band_energy(freq, mag, flo, fhi):
    idx = (freq >= flo) & (freq <= fhi)
    return float(np.sqrt(np.mean(mag[idx]**2))) if idx.any() else 0.0


def _peak_near(freq, mag, target, tol=3.0):
    idx = (freq >= target - tol) & (freq <= target + tol)
    return float(mag[idx].max()) if idx.any() else 0.0


def _orient_from_acc(ax, ay, az):
    mx, my, mz = ax.mean(), ay.mean(), az.mean()
    roll  = float(np.degrees(np.arctan2(my, mz)))
    pitch = float(np.degrees(np.arctan2(-mx, np.sqrt(my**2 + mz**2))))
    return roll, pitch


def _classify_faults(freq_acc, freq_mic, mic_mag, ax_mag, ay_mag, az_mag,
                     f1, bpf):
    """
    Classify pump faults from FFT data.
    Detects 3 fault types:
      1. Imbalance         — elevated 1X peak in radial axes
      2. No enough water   — broadband mic energy + BPF sidebands (cavitation)
      3. Structural looseness — sub-harmonic (0.5X) + 3X harmonic
    """
    faults = []

    radial_e = (_band_energy(freq_acc, ax_mag, 1, 400) +
                _band_energy(freq_acc, ay_mag, 1, 400)) / 2

    # ── 1. Imbalance — dominant 1X peak in radial (X/Y) ──────────────
    p1x = max(_peak_near(freq_acc, ax_mag, f1),
              _peak_near(freq_acc, ay_mag, f1))
    if radial_e > 1e-6:
        conf = min(p1x / (radial_e + 1e-9), 1.0)
        if conf > 0.15:
            faults.append({"name": "Imbalance", "conf": round(conf, 3),
                "desc": f"1X peak {p1x:.4f} g @ {f1:.1f} Hz"})

    # ── 2. No enough water (cavitation) — broadband mic + BPF sidebands
    bb_mic = _band_energy(freq_mic, mic_mag, 200, MIC_FS // 2)
    bpf_sb = (_peak_near(freq_mic, mic_mag, bpf + f1) +
              _peak_near(freq_mic, mic_mag, bpf - f1))
    cav_conf = min(bb_mic * 0.0002 + bpf_sb * 0.0003, 1.0)
    if cav_conf > 0.05:
        faults.append({"name": "No enough water", "conf": round(cav_conf, 3),
            "desc": f"Broadband mic {bb_mic:.0f}, BPF sidebands {bpf_sb:.0f}"})

    # ── 3. Structural looseness — 0.5X sub-harmonic + 3X harmonic ─────
    sub = max(_peak_near(freq_acc, ax_mag, 0.5*f1),
              _peak_near(freq_acc, ay_mag, 0.5*f1))
    p3x = max(_peak_near(freq_acc, ax_mag, 3*f1),
              _peak_near(freq_acc, ay_mag, 3*f1))
    loose_conf = min((sub + p3x) / (p1x + 1e-9), 1.0)
    if loose_conf > 0.1:
        faults.append({"name": "Structural looseness", "conf": round(loose_conf, 3),
            "desc": f"0.5X={sub:.4f} g, 3X={p3x:.4f} g"})

    faults.sort(key=lambda x: x["conf"], reverse=True)
    return faults or [{"name": "Healthy", "conf": 1.0, "desc": "No significant fault signatures"}]


# ── POST /pumps/{id}/raw — ESP32 sends raw binary samples ─────────────────
@app.post("/pumps/{pump_id}/raw")
async def post_raw(pump_id: str, request: Request,
                   db: Session = Depends(get_db)):
    """
    Receives raw binary samples from ESP32 via WiFi.

    Binary format (5120 bytes little-endian):
      [acc_x int16 × 512][acc_y int16 × 512][acc_z int16 × 512][mic uint16 × 1024]

    Computes FFTs, orientation, fault classification server-side,
    stores results for the frontend to poll via GET /pumps/{id}/fft.
    """
    body = await request.body()
    if len(body) != PAYLOAD_SIZE:
        raise HTTPException(status_code=400,
            detail=f"Expected {PAYLOAD_SIZE} bytes, got {len(body)}")

    # Unpack binary
    offset = 0
    ax_raw = np.frombuffer(body, dtype=np.int16, count=ACC_N, offset=offset)
    offset += ACC_N * 2
    ay_raw = np.frombuffer(body, dtype=np.int16, count=ACC_N, offset=offset)
    offset += ACC_N * 2
    az_raw = np.frombuffer(body, dtype=np.int16, count=ACC_N, offset=offset)
    offset += ACC_N * 2
    mic_raw = np.frombuffer(body, dtype=np.uint16, count=MIC_N, offset=offset)

    # Scale accelerometer to g
    ax_g = ax_raw.astype(np.float64) * ACC_SCALE
    ay_g = ay_raw.astype(np.float64) * ACC_SCALE
    az_g = az_raw.astype(np.float64) * ACC_SCALE

    # Compute FFTs
    freq_acc, mag_x = _compute_fft_acc(ax_g)
    _,        mag_y = _compute_fft_acc(ay_g)
    _,        mag_z = _compute_fft_acc(az_g)
    freq_mic, mag_mic = _compute_fft_mic(mic_raw.astype(np.float32))

    # Orientation
    roll, pitch = _orient_from_acc(ax_g, ay_g, az_g)

    # Fault classification
    f1  = RPM_DEFAULT / 60
    bpf = f1 * BLADES_DEFAULT
    faults = _classify_faults(freq_acc, freq_mic, mag_mic,
                              mag_x, mag_y, mag_z, f1, bpf)

    # Store in memory for GET /pumps/{id}/fft
    latest_fft[pump_id] = {
        "freq_acc": freq_acc.tolist(),
        "mag_x":    mag_x.tolist(),
        "mag_y":    mag_y.tolist(),
        "mag_z":    mag_z.tolist(),
        "freq_mic": freq_mic.tolist(),
        "mag_mic":  mag_mic.tolist(),
        "roll":     roll,
        "pitch":    pitch,
        "faults":   faults,
        "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
    }

    # ── Extract features and run ML scoring ─────────────────────────────
    feat = extract_features(ax_g, ay_g, az_g, mic_raw)
    collection.log_frame(pump_id, feat)
    anomaly_score, anomaly_tier = score_anomaly(feat)
    ml_label, ml_conf, ml_status, ml_desc = classify_fault_ml(feat)

    # Determine status and fault — ML classifier first, then rule-based fallback
    top_fault = faults[0]
    if ml_label is not None:
        # ML-driven: classifier gives fault type + status
        status = ml_status
        fault_name = ml_label if ml_label != "healthy" else None
        health_score_val = 1.0 - ml_conf if ml_label == "healthy" else ml_conf
        fault_desc = ml_desc
    else:
        # Fallback: rule-based
        if top_fault["name"] == "Healthy":
            status = "healthy"
        elif top_fault["conf"] > 0.6:
            status = "danger"
        else:
            status = "warning"
        fault_name = top_fault["name"] if top_fault["name"] != "Healthy" else None
        health_score_val = top_fault["conf"] if top_fault["name"] != "Healthy" else 0.0
        fault_desc = top_fault["desc"]

    # Store ML results in FFT data
    if anomaly_score is not None:
        latest_fft[pump_id]["anomaly_score"] = anomaly_score
        latest_fft[pump_id]["anomaly_tier"]  = anomaly_tier
    if ml_label is not None:
        latest_fft[pump_id]["ml_label"]       = ml_label
        latest_fft[pump_id]["ml_confidence"]  = round(ml_conf, 3)
        latest_fft[pump_id]["ml_status"]      = ml_status
        latest_fft[pump_id]["ml_description"] = fault_desc

    # Store reading in DB
    acc_rms   = float(np.sqrt(np.mean(ax_g**2 + ay_g**2 + az_g**2)))
    acc_x_rms = float(np.sqrt(np.mean(ax_g**2)))
    acc_y_rms = float(np.sqrt(np.mean(ay_g**2)))
    acc_z_rms = float(np.sqrt(np.mean(az_g**2)))
    mic_rms   = float(np.sqrt(np.mean((mic_raw.astype(np.float32) - mic_raw.mean())**2)))

    reading = Reading(
        pump_id      = pump_id,
        status       = status,
        fault_type   = fault_name,
        mic_rms      = mic_rms,
        acc_rms      = acc_rms,
        acc_x_rms    = acc_x_rms,
        acc_y_rms    = acc_y_rms,
        acc_z_rms    = acc_z_rms,
        health_score = health_score_val,
    )
    db.add(reading)

    # Save fault event on status change
    last = (
        db.query(Reading)
        .filter(Reading.pump_id == pump_id)
        .order_by(Reading.timestamp.desc())
        .offset(1)
        .first()
    )
    if last is None or last.status != status:
        desc = fault_desc
        if ml_conf is not None:
            desc = f"[ML {ml_label} {ml_conf*100:.0f}%] {desc}"
        db.add(FaultEvent(
            pump_id     = pump_id,
            status      = status,
            description = desc,
        ))

    db.commit()

    return {"status": "ok", "faults": faults,
            "ml_label": ml_label, "ml_confidence": ml_conf,
            "anomaly_score": anomaly_score,
            "anomaly_tier": anomaly_tier,
            "collecting": collection.collecting,
            "collection_label": collection.label,
            "collection_frames": collection.frame_count}


# ── Collection control endpoints ──────────────────────────────────────────
class CollectionRequest(BaseModel):
    label: str   # e.g. "healthy", "imbalance_2g", "misalignment_1mm"

@app.post("/collection/start")
def collection_start(req: CollectionRequest):
    """
    Start logging engineered features to CSV.
    Call this before each data collection session with the appropriate label.

    Data collection plan (5 sections per condition, 5 min on + off each):
      healthy
      no_water
      imbalance
      looseness
    """
    collection.start(req.label)
    return {
        "status": "ok",
        "label": req.label,
        "csv_path": collection.csv_path,
        "message": f"Logging started with label '{req.label}'. "
                   f"Send sensor data via POST /pumps/{{pump_id}}/raw to record frames."
    }

@app.post("/collection/stop")
def collection_stop():
    """Stop logging and close the CSV file."""
    path = collection.csv_path
    frames = collection.frame_count
    collection.stop()
    return {
        "status": "ok",
        "frames_saved": frames,
        "csv_path": path,
    }

@app.get("/collection/status")
def collection_status():
    """Check current collection state."""
    return {
        "collecting":   collection.collecting,
        "label":        collection.label,
        "csv_path":     collection.csv_path,
        "frames_saved": collection.frame_count,
    }


# ── How long before a pump is considered offline (no data) ──────────────────
STALE_TIMEOUT = timedelta(seconds=30)


# ── GET /pumps — list all pumps with current status ────────────────────────
@app.get("/pumps")
def list_pumps(db: Session = Depends(get_db)):
    pumps = db.query(Pump).all()
    result = []
    now = datetime.utcnow()
    for pump in pumps:
        latest = (
            db.query(Reading)
            .filter(Reading.pump_id == pump.id)
            .order_by(Reading.timestamp.desc())
            .first()
        )
        # Count fault events in last 7 days
        since = now - timedelta(days=7)
        faults_7d = (
            db.query(FaultEvent)
            .filter(FaultEvent.pump_id == pump.id, FaultEvent.timestamp >= since,
                    FaultEvent.status != "healthy")
            .count()
        )

        # Check if pump is offline (no data received recently)
        # First check DB readings, then fall back to in-memory FFT data
        fft_live = latest_fft.get(pump.id)

        if latest and (now - latest.timestamp) <= STALE_TIMEOUT:
            # Fresh DB reading — use it
            status = latest.status
            fault  = latest.fault_type or "No faults detected"
        elif fft_live and fft_live.get("timestamp"):
            # No fresh DB reading, but we have live FFT data in memory
            a_tier  = fft_live.get("anomaly_tier")
            a_score = fft_live.get("anomaly_score")
            fft_faults = fft_live.get("faults", [])
            top = fft_faults[0] if fft_faults else None

            if a_tier is not None:
                # ML-driven status
                status = {"fault": "danger", "warning": "warning", "healthy": "healthy"}.get(a_tier, "healthy")
            elif top and top.get("name") and top["name"] != "Healthy":
                status = "danger" if top.get("conf", 0) > 0.6 else "warning"
            else:
                status = "healthy"

            if top and top.get("name") and top["name"] != "Healthy":
                fault = top["name"] + " detected"
                if a_score is not None:
                    fault += f" (ML score: {a_score})"
            else:
                fault = "No faults detected"
        elif latest:
            # Have old DB readings but nothing recent
            status = "offline"
            fault  = "No data — pump offline"
        else:
            status = "offline"
            fault  = "No data yet"

        result.append({
            "id":       pump.id,
            "name":     pump.name,
            "status":   status,
            "fault":    fault,
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