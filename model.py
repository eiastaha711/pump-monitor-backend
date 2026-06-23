"""
model.py — loads your trained Random Forest and runs predictions.

HOW TO PLUG IN YOUR REAL MODEL:
  1. Train your Random Forest in a separate script and save it:
       import pickle
       with open("model.pkl", "wb") as f:
           pickle.dump(your_model, f)

  2. Put model.pkl in the same folder as this file.
  3. The rest works automatically.

FEATURE ORDER — must match what you used during training:
  [mic_rms, mic_peak, mic_crest, mic_kurtosis,
   acc_rms, acc_peak, acc_crest, acc_kurtosis,
   mic_fft_dominant, acc_fft_dominant]

  Adjust FEATURE_NAMES below if yours differ.
"""

import pickle
import numpy as np
import os

# Labels your model outputs — must match training labels exactly
LABELS = ["healthy", "no_water", "looseness", "imbalance"]

# Map model output → (status, human description)
LABEL_MAP = {
    "healthy":    ("healthy", "No faults detected"),
    "no_water":   ("danger",  "Not enough water — cavitation detected"),
    "looseness":  ("warning", "Structural looseness detected"),
    "imbalance":  ("warning", "Imbalance detected"),
}

FEATURE_NAMES = [
    "mic_rms", "mic_peak", "mic_crest", "mic_kurtosis",
    "acc_rms", "acc_peak", "acc_crest", "acc_kurtosis",
    "mic_fft_dominant", "acc_fft_dominant",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

_model = None


def load_model():
    """Load model from disk. Called once at startup."""
    global _model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print(f"[model] Loaded from {MODEL_PATH}")
    else:
        print("[model] model.pkl not found — using rule-based fallback until model is ready")
        _model = None


def predict(features: dict) -> dict:
    """
    Takes a dict of feature values, returns prediction dict.

    Args:
        features: {"mic_rms": 0.12, "acc_rms": 0.09, ...}

    Returns:
        {
          "label":       "filter_fault",
          "status":      "warning",
          "description": "Filter fault detected",
          "health_score": 0.43
        }
    """
    if _model is not None:
        # Build feature vector in the correct order
        X = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
        label = _model.predict(X)[0]

        # Health score: 0 = healthy, 1 = worst
        # Use predict_proba if available, else map label to fixed score
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(X)[0]
            healthy_idx = list(_model.classes_).index("healthy") if "healthy" in list(_model.classes_) else 0
            health_score = float(1.0 - proba[healthy_idx])
        else:
            health_score = {"healthy": 0.1, "imbalance": 0.5, "looseness": 0.6, "no_water": 0.85}.get(label, 0.5)

    else:
        # ── Rule-based fallback (until model.pkl exists) ──────────────────
        # Simple threshold on RMS values — replace with your own thresholds
        mic_rms = features.get("mic_rms", 0.0)
        acc_rms = features.get("acc_rms", 0.0)
        combined = (mic_rms + acc_rms) / 2.0

        if combined < 0.30:
            label = "healthy"
            health_score = combined / 0.30 * 0.25
        elif combined < 0.50:
            label = "imbalance"
            health_score = 0.25 + (combined - 0.30) / 0.20 * 0.25
        elif combined < 0.70:
            label = "looseness"
            health_score = 0.50 + (combined - 0.50) / 0.20 * 0.25
        else:
            label = "no_water"
            health_score = min(0.75 + (combined - 0.70) * 0.5, 1.0)

    status, description = LABEL_MAP.get(label, ("healthy", "No faults detected"))

    return {
        "label":        label,
        "status":       status,
        "description":  description,
        "health_score": round(health_score, 3),
    }
