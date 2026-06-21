"""
features.py — Engineered feature extraction for pump health ML.

Extracts 16 physics-informed features from raw accelerometer + microphone
data, matching the signatures used in classify_faults().

These features are what the Isolation Forest trains on — not raw FFT bins.
Each feature corresponds to a physical phenomenon in rotating machinery.

Usage:
    from features import extract_features
    feat = extract_features(ax_g, ay_g, az_g, mic_raw, rpm=1450, blades=6)
    # feat is a dict: {"f1_peak_radial": 0.038, "f2_peak_radial": 0.012, ...}
"""

import numpy as np

# ── DSP constants (must match firmware) ───────────────────────────────────
ACC_FS    = 800
ACC_N     = 512
MIC_FS    = 8000
MIC_N     = 1024
ACC_SCALE = 4.0 / 2048.0   # raw int16 → g  (±4g, 12-bit)

# ── Feature names in fixed order (for model input) ────────────────────────
FEATURE_NAMES = [
    "f1_peak_radial",       #  0  1X peak in X/Y (g) — imbalance indicator
    "f2_peak_radial",       #  1  2X peak in X/Y (g) — misalignment indicator
    "f3_peak_radial",       #  2  3X peak in X/Y (g) — looseness indicator
    "f05_peak_radial",      #  3  0.5X sub-harmonic (g) — looseness indicator
    "bpf_peak_acc",         #  4  BPF peak in acc X/Y (g)
    "bpf_peak_mic",         #  5  BPF peak in mic FFT
    "bpf_sideband_mic",     #  6  BPF±F1 sidebands in mic — cavitation indicator
    "radial_energy",        #  7  RMS energy 1–400 Hz in X/Y (g)
    "axial_energy",         #  8  RMS energy 1–400 Hz in Z (g)
    "axial_radial_ratio",   #  9  axial/radial energy ratio — misalignment
    "hf_energy",            # 10  RMS energy 200–400 Hz in X (g) — bearing
    "hf_lf_ratio",          # 11  HF/LF energy ratio — bearing indicator
    "broadband_mic",        # 12  RMS energy 200–4000 Hz in mic — cavitation
    "acc_rms_total",        # 13  overall acc RMS (g) — general severity
    "mic_rms",              # 14  overall mic RMS — general severity
    "roll",                 # 15  sensor roll (degrees)
    "pitch",                # 16  sensor pitch (degrees)
]


# ── Helper DSP functions ──────────────────────────────────────────────────
def _fft(signal, fs):
    """Hann-windowed FFT. Returns (freq_array, magnitude_array)."""
    sig = signal.astype(np.float64)
    sig -= sig.mean()
    win = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * win)) / (len(sig) / 2)
    freq = np.fft.rfftfreq(len(sig), 1.0 / fs)
    return freq, mag


def _band_energy(freq, mag, flo, fhi):
    """RMS of FFT magnitude in frequency band [flo, fhi]."""
    idx = (freq >= flo) & (freq <= fhi)
    return float(np.sqrt(np.mean(mag[idx]**2))) if idx.any() else 0.0


def _peak_near(freq, mag, target, tol=3.0):
    """Max FFT magnitude within ±tol Hz of target frequency."""
    idx = (freq >= target - tol) & (freq <= target + tol)
    return float(mag[idx].max()) if idx.any() else 0.0


# ── Main extraction function ─────────────────────────────────────────────
def extract_features(ax_g, ay_g, az_g, mic_raw,
                     rpm=1450, blades=6):
    """
    Extract 17 engineered features from one capture frame.

    Args:
        ax_g:     np.array (ACC_N,) accelerometer X in g
        ay_g:     np.array (ACC_N,) accelerometer Y in g
        az_g:     np.array (ACC_N,) accelerometer Z in g
        mic_raw:  np.array (MIC_N,) raw mic ADC values (uint16)
        rpm:      pump RPM (default 1450)
        blades:   impeller blade count (default 6)

    Returns:
        dict with keys matching FEATURE_NAMES
    """
    f1  = rpm / 60.0
    bpf = f1 * blades

    # Compute FFTs
    freq_acc, mag_x = _fft(ax_g, ACC_FS)
    _,        mag_y = _fft(ay_g, ACC_FS)
    _,        mag_z = _fft(az_g, ACC_FS)
    freq_mic, mag_mic = _fft(mic_raw.astype(np.float32), MIC_FS)

    # ── Shaft-related peaks (radial = max of X, Y) ───────────────────
    f1_peak  = max(_peak_near(freq_acc, mag_x, f1),
                   _peak_near(freq_acc, mag_y, f1))
    f2_peak  = max(_peak_near(freq_acc, mag_x, 2*f1),
                   _peak_near(freq_acc, mag_y, 2*f1))
    f3_peak  = max(_peak_near(freq_acc, mag_x, 3*f1),
                   _peak_near(freq_acc, mag_y, 3*f1))
    f05_peak = max(_peak_near(freq_acc, mag_x, 0.5*f1),
                   _peak_near(freq_acc, mag_y, 0.5*f1))

    # ── BPF peaks ─────────────────────────────────────────────────────
    bpf_acc  = max(_peak_near(freq_acc, mag_x, bpf),
                   _peak_near(freq_acc, mag_y, bpf))
    bpf_mic  = _peak_near(freq_mic, mag_mic, bpf)
    bpf_sb   = (_peak_near(freq_mic, mag_mic, bpf + f1) +
                _peak_near(freq_mic, mag_mic, bpf - f1))

    # ── Band energies ─────────────────────────────────────────────────
    radial_e = (_band_energy(freq_acc, mag_x, 1, 400) +
                _band_energy(freq_acc, mag_y, 1, 400)) / 2
    axial_e  = _band_energy(freq_acc, mag_z, 1, 400)
    ax_ratio = axial_e / (radial_e + 1e-9)

    hf_e     = _band_energy(freq_acc, mag_x, 200, ACC_FS // 2)
    lf_e     = _band_energy(freq_acc, mag_x, 1, 200)
    hf_lf    = hf_e / (lf_e + 1e-9)

    bb_mic   = _band_energy(freq_mic, mag_mic, 200, MIC_FS // 2)

    # ── Overall RMS ───────────────────────────────────────────────────
    acc_rms  = float(np.sqrt(np.mean(ax_g**2 + ay_g**2 + az_g**2)))
    mic_f    = mic_raw.astype(np.float32)
    mic_rms  = float(np.sqrt(np.mean((mic_f - mic_f.mean())**2)))

    # ── Orientation ───────────────────────────────────────────────────
    mx, my, mz = ax_g.mean(), ay_g.mean(), az_g.mean()
    roll  = float(np.degrees(np.arctan2(my, mz)))
    pitch = float(np.degrees(np.arctan2(-mx, np.sqrt(my**2 + mz**2))))

    return {
        "f1_peak_radial":     round(f1_peak, 6),
        "f2_peak_radial":     round(f2_peak, 6),
        "f3_peak_radial":     round(f3_peak, 6),
        "f05_peak_radial":    round(f05_peak, 6),
        "bpf_peak_acc":       round(bpf_acc, 6),
        "bpf_peak_mic":       round(bpf_mic, 6),
        "bpf_sideband_mic":   round(bpf_sb, 6),
        "radial_energy":      round(radial_e, 6),
        "axial_energy":       round(axial_e, 6),
        "axial_radial_ratio": round(ax_ratio, 6),
        "hf_energy":          round(hf_e, 6),
        "hf_lf_ratio":        round(hf_lf, 6),
        "broadband_mic":      round(bb_mic, 6),
        "acc_rms_total":      round(acc_rms, 6),
        "mic_rms":            round(mic_rms, 6),
        "roll":               round(roll, 2),
        "pitch":              round(pitch, 2),
    }


def features_to_vector(feat_dict):
    """Convert feature dict to numpy array in FEATURE_NAMES order."""
    return np.array([feat_dict[name] for name in FEATURE_NAMES])
