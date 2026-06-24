"""
train_model.py — Train Isolation Forest for pump anomaly detection.

This script:
  1. Loads CSV files from collected_data/ (or a specified directory)
  2. Splits data: healthy-only for training, everything for evaluation
  3. Trains an Isolation Forest on healthy data only
  4. Calibrates anomaly score to 0–100 scale using healthy baseline
  5. Saves the trained model + scaler to pump_anomaly_model.pkl
  6. Prints evaluation results showing how early faults are detected

Usage:
  python train_model.py
  python train_model.py --data-dir collected_data --contamination 0.02
  python train_model.py --data-dir collected_data --show-plots

The trained model is used by the backend to score incoming data
and raise warnings before fault thresholds trigger.
"""

import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Feature columns (must match features.py FEATURE_NAMES)
FEATURE_COLS = [
    "f1_peak_radial", "f2_peak_radial", "f3_peak_radial", "f05_peak_radial",
    "bpf_peak_acc", "bpf_peak_mic", "bpf_sideband_mic",
    "radial_energy", "axial_energy", "axial_radial_ratio",
    "hf_energy", "hf_lf_ratio", "broadband_mic",
    "acc_rms_total", "mic_rms",
    "roll", "pitch",
]


def load_data(data_dir):
    """Load all CSV files from data_dir into a single DataFrame."""
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[Error] No CSV files found in {data_dir}")
        print(f"        Collect data first using POST /collection/start")
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        label = df["label"].iloc[0] if "label" in df.columns else "unknown"
        print(f"  Loaded {f}: {len(df)} frames, label='{label}'")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} frames from {len(files)} files")
    return combined


def train(data_dir="collected_data", contamination=0.02, show_plots=False):
    """
    Train Isolation Forest on healthy data.

    Args:
        data_dir:      folder containing CSV files from collection
        contamination: expected fraction of outliers in healthy data
                       (accounts for noise; 0.02 = 2% is conservative)
        show_plots:    if True, show score distribution plots
    """
    print("=" * 60)
    print("  Pump Anomaly Model — Training")
    print("=" * 60)
    print(f"\nLoading data from: {data_dir}/\n")

    df = load_data(data_dir)
    if df is None:
        return

    # Check that we have feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[Error] Missing feature columns: {missing}")
        return

    # ── Split by label ────────────────────────────────────────────────
    healthy = df[df["label"] == "healthy"]
    faulty  = df[df["label"] != "healthy"]

    print(f"\nHealthy frames:  {len(healthy)}")
    print(f"Faulty frames:   {len(faulty)}")
    if len(faulty) > 0:
        fault_labels = faulty["label"].value_counts()
        for label, count in fault_labels.items():
            print(f"  - {label}: {count} frames")

    if len(healthy) < 50:
        print(f"\n[Warning] Only {len(healthy)} healthy frames.")
        print(f"          Recommend at least 200+ for reliable anomaly detection.")
        if len(healthy) < 10:
            print(f"[Error] Too few healthy frames to train. Collect more data.")
            return

    # ── Prepare feature matrices ──────────────────────────────────────
    X_healthy = healthy[FEATURE_COLS].values
    X_all     = df[FEATURE_COLS].values

    # Standardize features (important for Isolation Forest)
    scaler = StandardScaler()
    X_healthy_scaled = scaler.fit_transform(X_healthy)

    # ── Train Isolation Forest ────────────────────────────────────────
    print(f"\nTraining Isolation Forest...")
    print(f"  contamination = {contamination}")
    print(f"  features      = {len(FEATURE_COLS)}")
    print(f"  n_estimators  = 200")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_healthy_scaled)
    print("  Training complete.")

    # ── Calibrate anomaly scores ──────────────────────────────────────
    # score_samples() returns negative values (more negative = more anomalous)
    # We convert to 0–100 scale where:
    #   0   = perfectly normal
    #   100 = extremely anomalous
    healthy_scores = model.score_samples(X_healthy_scaled)
    score_min = float(healthy_scores.min())
    score_max = float(healthy_scores.max())

    # Use healthy score distribution to set thresholds
    healthy_pct = np.percentile(healthy_scores, [1, 5, 50, 95, 99])
    print(f"\nHealthy score distribution:")
    print(f"  1st percentile:  {healthy_pct[0]:.4f}")
    print(f"  5th percentile:  {healthy_pct[1]:.4f}")
    print(f"  Median:          {healthy_pct[2]:.4f}")
    print(f"  95th percentile: {healthy_pct[3]:.4f}")
    print(f"  99th percentile: {healthy_pct[4]:.4f}")

    # Calibration: map raw scores to 0–100
    # score_max (most normal) → 0, score_min (least normal in healthy) → ~30
    # Anything below score_min gets scores > 30 (into warning/fault territory)
    calibration = {
        "healthy_median": float(healthy_pct[2]),
        "healthy_p01":    float(healthy_pct[0]),
        "healthy_p99":    float(healthy_pct[4]),
        "score_min":      score_min,
        "score_max":      score_max,
    }

    # ── Evaluate on fault data ────────────────────────────────────────
    if len(faulty) > 0:
        print(f"\n{'─' * 60}")
        print(f"  Evaluation on fault data")
        print(f"{'─' * 60}")

        X_all_scaled = scaler.transform(X_all)
        all_scores = model.score_samples(X_all_scaled)

        # Normalized anomaly score: 0 = normal, 100 = very anomalous
        def normalize_score(raw_score):
            # Linear map: score_max → 0, score_min → 30
            # Below score_min → >30 (anomalous territory)
            span = score_max - score_min
            if span < 1e-9:
                return 0.0
            normalized = (1.0 - (raw_score - score_min) / span) * 30.0
            return max(0.0, min(100.0, normalized))

        df["anomaly_score"] = [normalize_score(s) for s in all_scores]

        # Per-label statistics
        for label in df["label"].unique():
            subset = df[df["label"] == label]
            scores = subset["anomaly_score"]
            print(f"\n  {label}:")
            print(f"    Mean score:  {scores.mean():.1f}")
            print(f"    Max score:   {scores.max():.1f}")
            print(f"    > 30 (warn): {(scores > 30).sum()}/{len(scores)} "
                  f"({(scores > 30).mean()*100:.0f}%)")
            print(f"    > 60 (fault):{(scores > 60).sum()}/{len(scores)} "
                  f"({(scores > 60).mean()*100:.0f}%)")

    # ── Save model ────────────────────────────────────────────────────
    # ── Train supervised classifier (Random Forest) ───────────────────
    print(f"\n{'─' * 60}")
    print(f"  Training fault classifier (Random Forest)")
    print(f"{'─' * 60}")

    X_all = df[FEATURE_COLS].values
    y_all = df["label"].values

    clf_scaler = StandardScaler()
    X_all_scaled = clf_scaler.fit_transform(X_all)

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )
    classifier.fit(X_all_scaled, y_all)

    # Cross-validation accuracy
    cv_scores = cross_val_score(classifier, X_all_scaled, y_all, cv=min(5, len(df)//4), scoring='accuracy')
    print(f"  Classes:    {list(classifier.classes_)}")
    print(f"  Accuracy:   {cv_scores.mean()*100:.1f}% (±{cv_scores.std()*100:.1f}%)")

    # Per-class accuracy
    y_pred = classifier.predict(X_all_scaled)
    for label in classifier.classes_:
        mask = y_all == label
        acc = (y_pred[mask] == label).mean() * 100
        print(f"    {label}: {acc:.0f}% ({mask.sum()} frames)")

    # ── Compute severity thresholds from data ─────────────────────────
    # For each fault: threshold = healthy 95th percentile for the key feature.
    # Below → warning (ML early detection), at/above → danger (risk).
    print(f"\n{'─' * 60}")
    print(f"  Computing severity thresholds")
    print(f"{'─' * 60}")

    fault_feature_map = {
        "imbalance": {
            "feature": "f1_peak_radial",
            "desc_warning": "Imbalance detected (early) — monitor closely",
            "desc_danger":  "Imbalance — risk level, action required",
        },
        "looseness": {
            "feature": "f05_peak_radial",
            "secondary": "f3_peak_radial",
            "desc_warning": "Looseness detected (early) — check mounting bolts",
            "desc_danger":  "Structural looseness — risk level, tighten immediately",
        },
        "no_water": {
            "feature": "broadband_mic",
            "desc_warning": "No water suspected (early) — check supply valve",
            "desc_danger":  "No water / dry run — risk level, stop pump immediately",
        },
    }

    fault_thresholds = {}
    for fault_name, info in fault_feature_map.items():
        feat_col = info["feature"]
        # Threshold = 95th percentile of healthy data for this feature
        healthy_vals = healthy[feat_col].values
        threshold = float(np.percentile(healthy_vals, 95))

        fault_thresholds[fault_name] = {
            "feature": feat_col,
            "threshold": round(threshold, 6),
            "desc_warning": info["desc_warning"],
            "desc_danger": info["desc_danger"],
        }

        # Also handle secondary feature (looseness 3X)
        if "secondary" in info:
            sec_col = info["secondary"]
            sec_thresh = float(np.percentile(healthy[sec_col].values, 95))
            fault_thresholds[fault_name]["secondary"] = sec_col
            fault_thresholds[fault_name]["sec_threshold"] = round(sec_thresh, 6)

        # Show how well it separates
        fault_vals = df[df["label"] == fault_name][feat_col].values if fault_name in df["label"].values else []
        if len(fault_vals) > 0:
            above = (fault_vals >= threshold).sum()
            print(f"  {fault_name}: {feat_col} threshold = {threshold:.4f}")
            print(f"    Fault frames above threshold: {above}/{len(fault_vals)} "
                  f"({above/len(fault_vals)*100:.0f}%) → danger")
            print(f"    Fault frames below threshold: {len(fault_vals)-above}/{len(fault_vals)} "
                  f"({(len(fault_vals)-above)/len(fault_vals)*100:.0f}%) → warning")

    model_data = {
        "model":        model,
        "scaler":       scaler,
        "feature_names": FEATURE_COLS,
        "calibration":  calibration,
        "classifier":   classifier,
        "clf_scaler":   clf_scaler,
        "clf_classes":  list(classifier.classes_),
        "fault_thresholds": fault_thresholds,
        "train_info": {
            "healthy_frames":  len(healthy),
            "total_frames":    len(df),
            "contamination":   contamination,
            "n_features":      len(FEATURE_COLS),
            "cv_accuracy":     round(cv_scores.mean(), 3),
        },
    }

    model_path = "pump_anomaly_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n{'=' * 60}")
    print(f"  Model saved to: {model_path}")
    print(f"{'=' * 60}")

    # ── Optional plots ────────────────────────────────────────────────
    if show_plots:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Score distributions per label
            ax = axes[0]
            for label in df["label"].unique():
                subset = df[df["label"] == label]
                ax.hist(subset["anomaly_score"], bins=30, alpha=0.6, label=label)
            ax.axvline(30, color='orange', ls='--', label='Warning threshold')
            ax.axvline(60, color='red', ls='--', label='Fault threshold')
            ax.set_xlabel("Anomaly Score (0–100)")
            ax.set_ylabel("Frames")
            ax.set_title("Score Distribution by Label")
            ax.legend()

            # Plot 2: Score over time
            ax = axes[1]
            colors = {"healthy": "green"}
            color_list = ["red", "orange", "purple", "brown", "pink"]
            ci = 0
            for label in df["label"].unique():
                if label != "healthy":
                    colors[label] = color_list[ci % len(color_list)]
                    ci += 1

            for label in df["label"].unique():
                subset = df[df["label"] == label]
                ax.scatter(range(len(subset)), subset["anomaly_score"],
                          s=3, alpha=0.5, color=colors.get(label, "gray"),
                          label=label)
            ax.axhline(30, color='orange', ls='--', alpha=0.7)
            ax.axhline(60, color='red', ls='--', alpha=0.7)
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Anomaly Score")
            ax.set_title("Anomaly Score Over Time")
            ax.legend()

            plt.tight_layout()
            plt.savefig("training_report.png", dpi=150)
            print(f"  Plot saved to: training_report.png")
            plt.show()
        except ImportError:
            print("  (matplotlib not available — skipping plots)")

    return model_data


def main():
    parser = argparse.ArgumentParser(description="Train pump anomaly detection model")
    parser.add_argument("--data-dir", type=str, default="collected_data",
                        help="Directory containing collected CSV files")
    parser.add_argument("--contamination", type=float, default=0.02,
                        help="Expected outlier fraction in healthy data (default: 0.02)")
    parser.add_argument("--show-plots", action="store_true",
                        help="Show training evaluation plots")
    args = parser.parse_args()

    train(data_dir=args.data_dir,
          contamination=args.contamination,
          show_plots=args.show_plots)


if __name__ == "__main__":
    main()