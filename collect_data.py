"""
collect_data.py — Guided data collection for pump fault detection.

Protocol:
  4 conditions × 5 sections each = 20 recording sessions
  Each section: pump ON for 5 minutes, then pump OFF

Conditions:
  1. healthy     — normal operation, no faults
  2. imbalance   — attach imbalance weight to impeller
  3. looseness   — loosen mounting bolts
  4. no_water    — close water supply valve / dry run

Usage:
  python collect_data.py --api http://localhost:8000

  The script walks you through each session interactively.
  It calls POST /collection/start and /collection/stop on the backend.
  Data is saved to collected_data/ as CSV files (one per session).

After all 20 sessions are recorded:
  python train_model.py --data-dir collected_data --show-plots
"""

import argparse
import time
import requests
import sys


# ── Collection protocol ───────────────────────────────────────────────────
CONDITIONS = ["healthy", "imbalance", "looseness", "no_water"]
SECTIONS_PER_CONDITION = 5
RECORD_MINUTES = 5

INSTRUCTIONS = {
    "healthy": (
        "HEALTHY — Normal operation\n"
        "  • Pump mounted securely, all bolts tight\n"
        "  • Water supply fully open\n"
        "  • No extra weights on impeller"
    ),
    "imbalance": (
        "IMBALANCE — Unbalanced impeller\n"
        "  • Attach imbalance weight to the impeller\n"
        "  • Keep water supply open, bolts tight"
    ),
    "looseness": (
        "STRUCTURAL LOOSENESS — Loose mounting\n"
        "  • Loosen the mounting bolts\n"
        "  • Keep water supply open, no imbalance weight"
    ),
    "no_water": (
        "NO WATER — Dry run / insufficient water\n"
        "  • Close or partially close the water supply valve\n"
        "  • Keep mounting bolts tight, no imbalance weight\n"
        "  ⚠ Monitor pump temperature — don't run dry too long"
    ),
}


def wait_for_backend(api_url, timeout=10):
    """Check backend is reachable."""
    try:
        r = requests.get(f"{api_url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def start_collection(api_url, label):
    """Tell backend to start logging with given label."""
    r = requests.post(f"{api_url}/collection/start", json={"label": label})
    r.raise_for_status()
    return r.json()


def stop_collection(api_url):
    """Tell backend to stop logging."""
    r = requests.post(f"{api_url}/collection/stop")
    r.raise_for_status()
    return r.json()


def get_collection_status(api_url):
    """Check current collection state."""
    r = requests.get(f"{api_url}/collection/status")
    r.raise_for_status()
    return r.json()


def countdown(seconds, message="Recording"):
    """Show a live countdown timer."""
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        status = get_collection_status(api_url) if remaining % 10 == 0 else None
        frames = f" ({status['frames_saved']} frames)" if status else ""
        sys.stdout.write(f"\r  {message}: {mins:02d}:{secs:02d} remaining{frames}   ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r  " + " " * 60 + "\r")
    sys.stdout.flush()


def run_collection(api_url):
    """Run the full interactive collection protocol."""
    print("=" * 60)
    print("  PUMP DATA COLLECTION")
    print("=" * 60)
    print(f"\n  Protocol: {len(CONDITIONS)} conditions × {SECTIONS_PER_CONDITION} sections")
    print(f"  Each section: {RECORD_MINUTES} min recording")
    print(f"  Total sessions: {len(CONDITIONS) * SECTIONS_PER_CONDITION}")
    print(f"  Backend: {api_url}\n")

    # Check backend
    print("  Checking backend...", end=" ")
    if not wait_for_backend(api_url):
        print("FAILED")
        print(f"\n  ✗ Cannot reach {api_url}")
        print(f"    Start the backend first: uvicorn main:app --host 0.0.0.0 --port 8000")
        return
    print("OK ✓\n")

    total_sessions = len(CONDITIONS) * SECTIONS_PER_CONDITION
    session_num = 0

    for condition in CONDITIONS:
        print("─" * 60)
        print(f"\n  CONDITION: {condition.upper()}")
        print(f"  {INSTRUCTIONS[condition]}\n")

        input("  Press ENTER when the pump is set up for this condition...")
        print()

        for section in range(1, SECTIONS_PER_CONDITION + 1):
            session_num += 1
            label = condition
            print(f"  ── Section {section}/{SECTIONS_PER_CONDITION} "
                  f"(session {session_num}/{total_sessions}) ──")

            # Start
            input(f"  Turn ON the pump, then press ENTER to start recording...")
            result = start_collection(api_url, label)
            print(f"  ✓ Recording started → {result.get('csv_path', '?')}")

            # Record for N minutes
            countdown(RECORD_MINUTES * 60, f"Recording [{label}]")

            # Stop
            result = stop_collection(api_url)
            frames = result.get("frames_saved", 0)
            print(f"  ✓ Recording stopped — {frames} frames saved")

            if section < SECTIONS_PER_CONDITION:
                input(f"  Turn OFF the pump. Press ENTER when ready for next section...")
            else:
                print(f"  Turn OFF the pump.")
            print()

    print("=" * 60)
    print("  ✓ ALL DATA COLLECTED!")
    print("=" * 60)
    print(f"\n  Next step: train the anomaly detection model:")
    print(f"    python train_model.py --data-dir collected_data --show-plots\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided pump data collection")
    parser.add_argument("--api", type=str, default="http://localhost:8000",
                        help="Backend API URL (default: http://localhost:8000)")
    args = parser.parse_args()
    api_url = args.api
    run_collection(api_url)
