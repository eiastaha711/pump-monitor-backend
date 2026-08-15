"""
FastAPI backend for the Pump Health Monitor, responsible for receiving sensor data, processing pump signals, and exposing monitoring endpoints.
It coordinates FFT analysis, fault detection, data collection, model inference, and database persistence for monitored pumps.
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


app = FastAPI(title="Pump Health Monitor", version="1.3")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


latest_fft = {}


import csv, os, pickle

class CollectionState:
    def __init__(self):
        """
        Initializes the data collection state and its file-handling resources.
        The object tracks the active label, output path, writer, file handle, and number of recorded frames.
        """
        self.label      = "healthy"
        self.collecting  = False
        self.log_dir     = "collected_data"
        self.csv_path    = None
        self.csv_writer  = None
        self.csv_file    = None
        self.frame_count = 0

    def start(self, label: str):
        """
        Starts a new labeled data collection session and creates its CSV output file.
        Any previously open session is closed before the new file and CSV writer are initialized.
        """
        self.stop()
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
        """
        Writes one engineered feature frame to the active collection CSV file.
        The frame is ignored when collection is inactive; otherwise, it is flushed immediately and counted.
        """
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
        """
        Stops the current data collection session and safely closes the CSV file.
        It also clears the active writer and file references while preserving the final collection metadata.
        """
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print(f"[Collection] Stopped: {self.frame_count} frames saved to {self.csv_path}")
        self.collecting = False
        self.csv_writer = None
        self.csv_file = None

collection = CollectionState()


ANOMALY_MODEL_PATH = "pump_anomaly_model.pkl"
anomaly_model = None
anomaly_scaler = None
anomaly_calibration = None
fault_classifier = None
fault_clf_scaler = None


def load_anomaly_model():
    """
    Loads the trained anomaly detector, feature scaler, calibration data, and optional fault classifier from disk.
    If the model file is unavailable, anomaly scoring remains disabled and the application continues with fallback behavior.
    """
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
    Scores an engineered feature vector with the trained Isolation Forest and converts the raw output to a normalized anomaly score.
    It returns both the score and a healthy, warning, or fault tier based on the configured calibration range.
    """
    if anomaly_model is None or anomaly_scaler is None:
        return None, "unknown"

    from features import FEATURE_NAMES, features_to_vector
    vec = features_to_vector(features_dict).reshape(1, -1)
    vec_scaled = anomaly_scaler.transform(vec)

    raw_score = float(anomaly_model.score_samples(vec_scaled)[0])


    cal = anomaly_calibration
    span = cal["score_max"] - cal["score_min"]
    if span < 1e-9:
        normalized = 0.0
    else:
        normalized = (1.0 - (raw_score - cal["score_min"]) / span) * 30.0
    normalized = max(0.0, min(100.0, normalized))


    if normalized >= 60:
        tier = "fault"
    elif normalized >= 30:
        tier = "warning"
    else:
        tier = "healthy"

    return round(normalized, 1), tier


FAULT_THRESHOLDS = {
    "imbalance": {
        "feature": "f1_peak_radial",
        "threshold": 0.04,
        "unit": "g",
        "desc_warning": "Imbalance detected (early) — monitor closely",
        "desc_danger":  "Imbalance — risk level, action required",
    },
    "looseness": {
        "feature": "f05_peak_radial",
        "threshold": 0.02,
        "secondary": "f3_peak_radial",
        "sec_threshold": 0.02,
        "unit": "g",
        "desc_warning": "Looseness detected (early) — check mounting bolts",
        "desc_danger":  "Structural looseness — risk level, tighten immediately",
    },
    "no_water": {
        "feature": "broadband_mic",
        "threshold": 0.035,
        "unit": "RMS",
        "desc_warning": "No water suspected (early) — check supply valve",
        "desc_danger":  "No water / dry run — risk level, stop pump immediately",
    },
}


trained_thresholds = None


def load_thresholds_from_model():
    """
    Loads fault-severity thresholds that were calibrated during model training.
    When calibrated thresholds are present, they override the built-in fallback thresholds used for severity assessment.
    """
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
    Classifies the pump condition with the trained fault classifier and estimates prediction confidence.
    For detected faults, it compares the relevant features with calibrated or fallback thresholds to assign warning or danger severity.
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


    thresholds = trained_thresholds or FAULT_THRESHOLDS
    fault_info = thresholds.get(label, {})
    feature_name = fault_info.get("feature")

    if feature_name and feature_name in features_dict:
        value = features_dict[feature_name]
        threshold = fault_info.get("threshold", 0)


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

        status = "danger" if confidence > 0.8 else "warning"
        description = f"{label} detected ({confidence*100:.0f}% confidence)"

    return label, confidence, status, description


@app.on_event("startup")
def startup():
    """
    Initializes the application resources when the FastAPI service starts.
    It creates or migrates database tables, loads trained models and thresholds, and registers the default pumps when necessary.
    """
    create_tables()


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

    db = next(get_db())
    for pid, name in [("pump_01", "Pump 01"), ("pump_02", "Pump 02"), ("pump_03", "Pump 03")]:
        if not db.query(Pump).filter(Pump.id == pid).first():
            db.add(Pump(id=pid, name=name))
    db.commit()
    db.close()
    print("[startup] Tables ready, models loaded, pumps registered.")


class SensorPayload(BaseModel):
    pump_id:    str
    mic_rms:    float
    mic_peak:   Optional[float] = None
    mic_crest:  Optional[float] = None
    mic_kurtosis: Optional[float] = None
    acc_rms:    float
    acc_peak:   Optional[float] = None
    acc_crest:  Optional[float] = None
    acc_kurtosis: Optional[float] = None
    mic_fft_dominant: Optional[float] = None
    acc_fft_dominant: Optional[float] = None


class FFTPayload(BaseModel):
    pump_id:   str
    freq_acc:  List[float]
    mag_x:     List[float]
    mag_y:     List[float]
    mag_z:     List[float]
    freq_mic:  List[float]
    mag_mic:   List[float]
    roll:      Optional[float] = 0.0
    pitch:     Optional[float] = 0.0
    faults:    Optional[list]  = None


@app.post("/ingest")
def ingest(payload: SensorPayload, db: Session = Depends(get_db)):
    """
    Receives precomputed ESP32 sensor features, runs the prediction pipeline, and stores the resulting pump reading.
    A fault event is added only when the reported pump status changes from the previous reading.
    """
    features = payload.dict()
    result = predict(features)


    reading = Reading(
        pump_id      = payload.pump_id,
        status       = result["status"],
        fault_type   = result["label"] if result["label"] != "healthy" else None,
        mic_rms      = payload.mic_rms,
        acc_rms      = payload.acc_rms,
        health_score = result["health_score"],
    )
    db.add(reading)


    last = (
        db.query(Reading)
        .filter(Reading.pump_id == payload.pump_id)
        .order_by(Reading.timestamp.desc())
        .offset(1)
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


@app.post("/pumps/{pump_id}/fft")
def post_fft(pump_id: str, payload: FFTPayload):
    """
    Receives the latest FFT snapshot produced by the external analyzer for a specific pump.
    The snapshot is stored in memory for frontend access and replaces the previous snapshot for that pump.
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


@app.get("/pumps/{pump_id}/fft")
def get_fft(pump_id: str):
    """
    Returns the latest in-memory FFT snapshot for the requested pump.
    If no snapshot has been received, it returns the expected response structure with empty signal arrays and default values.
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


ACC_N       = 512
ACC_FS      = 800
ACC_SCALE   = 4.0 / 2048.0
MIC_N       = 1024
MIC_FS      = 8000
RPM_DEFAULT = 1450
BLADES_DEFAULT = 6
PAYLOAD_SIZE = ACC_N * 3 * 2 + MIC_N * 2


def _compute_fft_acc(signal):
    """
    Computes the single-sided frequency spectrum of an accelerometer signal using a Hann window.
    The signal mean is removed before the FFT, and matching frequency and magnitude arrays are returned.
    """
    sig = signal.astype(np.float64)
    sig -= sig.mean()
    win = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * win)) / (len(sig) / 2)
    freq = np.fft.rfftfreq(len(sig), 1.0 / ACC_FS)
    return freq, mag


def _compute_fft_mic(signal):
    """
    Computes the single-sided frequency spectrum of a microphone signal using a Hann window.
    It removes the DC component and returns frequency bins together with normalized spectral magnitudes.
    """
    sig = signal.astype(np.float32)
    sig -= sig.mean()
    win = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * win)) / (MIC_N / 2)
    freq = np.fft.rfftfreq(len(sig), 1.0 / MIC_FS)
    return freq, mag


def _band_energy(freq, mag, flo, fhi):
    """
    Calculates the RMS spectral magnitude inside a selected frequency band.
    It returns zero when the requested band contains no FFT bins.
    """
    idx = (freq >= flo) & (freq <= fhi)
    return float(np.sqrt(np.mean(mag[idx]**2))) if idx.any() else 0.0


def _peak_near(freq, mag, target, tol=3.0):
    """
    Finds the largest spectral magnitude within a tolerance window around a target frequency.
    It returns zero when no frequency bins fall inside the requested window.
    """
    idx = (freq >= target - tol) & (freq <= target + tol)
    return float(mag[idx].max()) if idx.any() else 0.0


def _orient_from_acc(ax, ay, az):
    """
    Estimates sensor roll and pitch from the mean acceleration on the three axes.
    The calculated orientation angles are returned in degrees.
    """
    mx, my, mz = ax.mean(), ay.mean(), az.mean()
    roll  = float(np.degrees(np.arctan2(my, mz)))
    pitch = float(np.degrees(np.arctan2(-mx, np.sqrt(my**2 + mz**2))))
    return roll, pitch


def _classify_faults(freq_acc, freq_mic, mic_mag, ax_mag, ay_mag, az_mag,
                     f1, bpf):
    """
    Applies rule-based spectral analysis to identify imbalance, insufficient-water behavior, and structural looseness.
    Detected conditions are ranked by confidence, with a healthy result returned when no significant signature is found.
    """
    faults = []

    radial_e = (_band_energy(freq_acc, ax_mag, 1, 400) +
                _band_energy(freq_acc, ay_mag, 1, 400)) / 2


    p1x = max(_peak_near(freq_acc, ax_mag, f1),
              _peak_near(freq_acc, ay_mag, f1))
    if radial_e > 1e-6:
        conf = min(p1x / (radial_e + 1e-9), 1.0)
        if conf > 0.15:
            faults.append({"name": "Imbalance", "conf": round(conf, 3),
                "desc": f"1X peak {p1x:.4f} g @ {f1:.1f} Hz"})


    bb_mic = _band_energy(freq_mic, mic_mag, 200, MIC_FS // 2)
    bpf_sb = (_peak_near(freq_mic, mic_mag, bpf + f1) +
              _peak_near(freq_mic, mic_mag, bpf - f1))
    cav_conf = min(bb_mic * 0.0002 + bpf_sb * 0.0003, 1.0)
    if cav_conf > 0.05:
        faults.append({"name": "No enough water", "conf": round(cav_conf, 3),
            "desc": f"Broadband mic {bb_mic:.0f}, BPF sidebands {bpf_sb:.0f}"})


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


@app.post("/pumps/{pump_id}/raw")
async def post_raw(pump_id: str, request: Request,
                   db: Session = Depends(get_db)):
    """
    Receives raw binary accelerometer and microphone samples from the ESP32 and processes them server-side.
    It computes spectra and features, runs fault and anomaly inference, updates live FFT data, and persists the resulting pump status.
    """
    body = await request.body()
    if len(body) != PAYLOAD_SIZE:
        raise HTTPException(status_code=400,
            detail=f"Expected {PAYLOAD_SIZE} bytes, got {len(body)}")


    offset = 0
    ax_raw = np.frombuffer(body, dtype=np.int16, count=ACC_N, offset=offset)
    offset += ACC_N * 2
    ay_raw = np.frombuffer(body, dtype=np.int16, count=ACC_N, offset=offset)
    offset += ACC_N * 2
    az_raw = np.frombuffer(body, dtype=np.int16, count=ACC_N, offset=offset)
    offset += ACC_N * 2
    mic_raw = np.frombuffer(body, dtype=np.uint16, count=MIC_N, offset=offset)


    ax_g = ax_raw.astype(np.float64) * ACC_SCALE
    ay_g = ay_raw.astype(np.float64) * ACC_SCALE
    az_g = az_raw.astype(np.float64) * ACC_SCALE


    freq_acc, mag_x = _compute_fft_acc(ax_g)
    _,        mag_y = _compute_fft_acc(ay_g)
    _,        mag_z = _compute_fft_acc(az_g)
    freq_mic, mag_mic = _compute_fft_mic(mic_raw.astype(np.float32))


    roll, pitch = _orient_from_acc(ax_g, ay_g, az_g)


    f1  = RPM_DEFAULT / 60
    bpf = f1 * BLADES_DEFAULT
    faults = _classify_faults(freq_acc, freq_mic, mag_mic,
                              mag_x, mag_y, mag_z, f1, bpf)


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


    feat = extract_features(ax_g, ay_g, az_g, mic_raw)
    collection.log_frame(pump_id, feat)
    anomaly_score, anomaly_tier = score_anomaly(feat)
    ml_label, ml_conf, ml_status, ml_desc = classify_fault_ml(feat)


    top_fault = faults[0]
    if ml_label is not None:

        status = ml_status
        fault_name = ml_label if ml_label != "healthy" else None
        health_score_val = 1.0 - ml_conf if ml_label == "healthy" else ml_conf
        fault_desc = ml_desc
    else:

        if top_fault["name"] == "Healthy":
            status = "healthy"
        elif top_fault["conf"] > 0.6:
            status = "danger"
        else:
            status = "warning"
        fault_name = top_fault["name"] if top_fault["name"] != "Healthy" else None
        health_score_val = top_fault["conf"] if top_fault["name"] != "Healthy" else 0.0
        fault_desc = top_fault["desc"]


    if anomaly_score is not None:
        latest_fft[pump_id]["anomaly_score"] = anomaly_score
        latest_fft[pump_id]["anomaly_tier"]  = anomaly_tier
    if ml_label is not None:
        latest_fft[pump_id]["ml_label"]       = ml_label
        latest_fft[pump_id]["ml_confidence"]  = round(ml_conf, 3)
        latest_fft[pump_id]["ml_status"]      = ml_status
        latest_fft[pump_id]["ml_description"] = fault_desc


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


    last = (
        db.query(Reading)
        .filter(Reading.pump_id == pump_id)
        .order_by(Reading.timestamp.desc())
        .offset(1)
        .first()
    )
    if last is None or last.status != status:
        db.add(FaultEvent(
            pump_id     = pump_id,
            status      = status,
            description = fault_desc,
        ))

    db.commit()

    return {"status": "ok", "faults": faults,
            "ml_label": ml_label, "ml_confidence": ml_conf,
            "anomaly_score": anomaly_score,
            "anomaly_tier": anomaly_tier,
            "collecting": collection.collecting,
            "collection_label": collection.label,
            "collection_frames": collection.frame_count}


class CollectionRequest(BaseModel):
    label: str

@app.post("/collection/start")
def collection_start(req: CollectionRequest):
    """
    Starts feature collection using the label supplied in the request and opens a new CSV dataset file.
    The response reports the active label and output path so the collection session can be tracked externally.
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
    """
    Stops the active feature collection session and closes its CSV file.
    It returns the output path together with the total number of frames saved during the session.
    """
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
    """
    Returns the current state of the feature collection process.
    The response includes whether collection is active, the assigned label, output path, and saved frame count.
    """
    return {
        "collecting":   collection.collecting,
        "label":        collection.label,
        "csv_path":     collection.csv_path,
        "frames_saved": collection.frame_count,
    }


STALE_TIMEOUT = timedelta(seconds=30)


@app.get("/pumps")
def list_pumps(db: Session = Depends(get_db)):
    """
    Returns all registered pumps together with their latest operational status and recent fault count.
    A pump is marked offline when neither recent database readings nor live FFT data are available within the configured timeout.
    """
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

        since = now - timedelta(days=7)
        faults_7d = (
            db.query(FaultEvent)
            .filter(FaultEvent.pump_id == pump.id, FaultEvent.timestamp >= since,
                    FaultEvent.status != "healthy")
            .count()
        )


        fft_live = latest_fft.get(pump.id)

        if latest and (now - latest.timestamp) <= STALE_TIMEOUT:

            status = latest.status
            fault  = latest.fault_type or "No faults detected"
        elif fft_live and fft_live.get("timestamp"):

            a_tier  = fft_live.get("anomaly_tier")
            a_score = fft_live.get("anomaly_score")
            fft_faults = fft_live.get("faults", [])
            top = fft_faults[0] if fft_faults else None

            if a_tier is not None:

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


@app.get("/pumps/{pump_id}/history")
def pump_history(pump_id: str, days: int = 7, db: Session = Depends(get_db)):
    """
    Builds an hourly status timeline for the requested number of recent days.
    Each hour uses the most severe recorded state for that period, while future or unavailable hours are returned as empty.
    """
    since = datetime.utcnow() - timedelta(days=days)
    readings = (
        db.query(Reading)
        .filter(Reading.pump_id == pump_id, Reading.timestamp >= since)
        .order_by(Reading.timestamp.asc())
        .all()
    )


    now = datetime.utcnow()
    day_labels = []
    for i in range(days - 1, -1, -1):
        d = now - timedelta(days=i)
        day_labels.append("Today" if i == 0 else d.strftime("%a"))


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

            if label == "Today" and hour > now.hour:
                blocks.append("empty")
            else:
                blocks.append(slots.get((day_idx, hour), "empty"))
        result.append({"label": label, "blocks": blocks})

    return result


@app.get("/pumps/{pump_id}/events")
def pump_events(pump_id: str, limit: int = 10, db: Session = Depends(get_db)):
    """
    Returns the most recent fault events recorded for a specific pump.
    Event timestamps are formatted into readable relative or calendar-based labels before being returned to the client.
    """
    events = (
        db.query(FaultEvent)
        .filter(FaultEvent.pump_id == pump_id)
        .order_by(FaultEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    result = []
    for e in events:

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


@app.get("/pumps/{pump_id}/signal")
def pump_signal(pump_id: str, seconds: int = 60, db: Session = Depends(get_db)):
    """
    Returns recent microphone, accelerometer, and health-score readings for the requested time window.
    The data is ordered chronologically and formatted for direct use by the live signal chart.
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


@app.get("/health")
def health():
    """
    Provides a lightweight health-check endpoint for the backend service.
    A successful response indicates that the API is running and able to serve requests.
    """
    return {"status": "ok"}
