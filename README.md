# Rotating Machines Health Monitoring System

A low-cost, real-time fault detection system for centrifugal pumps and rotating machinery using vibration and acoustic sensing, machine learning classification, and a live web dashboard.

**Group 109** — Engineering Project and Workshops II, The Hebrew University of Jerusalem

**Live Dashboard:** [https://eiastaha711.github.io/pump-monitor-backend/](https://eiastaha711.github.io/pump-monitor-backend/)

---

## System Overview

The system captures vibration and acoustic data from a pump using a BMA400 MEMS accelerometer and an SPU0410 microphone mounted on a Sovev (iPipe) sensing platform. An ESP32 microcontroller streams binary sensor data over Wi-Fi to a cloud backend, which performs FFT-based feature extraction and classifies the pump's state using a trained Random Forest model.

**Four detectable states:** Healthy, No-Water (Dry Run), Mass Imbalance, Structural Looseness

```
Sensors (BMA400 + Mic) → ESP32 → Wi-Fi → Railway Backend → FFT & Features → Random Forest → Dashboard
```

## Architecture

| Component | Role |
|---|---|
| BMA400 accelerometer | Three-axis vibration sensing (800 Hz, 512 samples) |
| SPU0410 microphone | Acoustic sensing (8000 Hz, 1024 samples) |
| Sovev (iPipe) board | Non-invasive sensor mounting platform |
| ESP32-S3 | Sensor acquisition and Wi-Fi transmission |
| Railway backend | Cloud FFT processing, classification, REST API |
| Random Forest (scikit-learn) | Four-state fault classification |
| Web dashboard | Real-time monitoring, FFT spectrum, fault alerts |

## Repository Structure

| File | Description |
|---|---|
| `PMM_Firmware_3.ino` | ESP32 firmware — sensor acquisition and Wi-Fi streaming |
| `main.py` | FastAPI backend — receives data, computes FFT, runs classifier, serves API |
| `features.py` | Extracts physics-informed features from accelerometer and microphone data |
| `model.py` | Loads trained Random Forest model and returns predictions |
| `train_model.py` | Trains the classifier on collected labeled data |
| `collect_data.py` | Controls labeled data collection sessions via REST API |
| `database.py` | SQLAlchemy/SQLite schema for readings, fault events, and pump registry |
| `pump_fft_analyzer.py` | Local real-time FFT visualizer (Matplotlib + serial) for development |
| `index.html` | Web dashboard — single-page app with live status, FFT charts, and fault analysis |
| `requirements.txt` | Python dependencies |
| `Procfile` | Railway deployment configuration |

## Getting Started

### 1. Deploy the backend

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or deploy directly to [Railway](https://railway.app) by connecting this repository.

### 2. Configure the ESP32

Open `PMM_Firmware_3.ino`, set `PUMP_ID` to your machine name, and upload to the ESP32 board. The firmware will automatically stream sensor data to the backend.

### 3. Collect labeled data

```bash
# Start a collection session with a fault label
curl -X POST http://localhost:8000/collection/start -H "Content-Type: application/json" -d '{"label": "healthy"}'

# Run the pump under the target condition...

# Stop collection
curl -X POST http://localhost:8000/collection/stop
```

Repeat for each fault condition (healthy, no-water, imbalance, looseness).

### 4. Train the model

```bash
python train_model.py --data-dir collected_data
```

### 5. Monitor

Open the [live dashboard](https://eiastaha711.github.io/pump-monitor-backend/) in any browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/pumps/{id}/raw` | Receive binary sensor payload from ESP32 |
| GET | `/pumps` | List all pumps with current status |
| GET | `/pumps/{id}/fft` | Latest FFT spectrum data |
| GET | `/pumps/{id}/history` | 7-day hourly status timeline |
| GET | `/pumps/{id}/events` | Recent fault events |
| GET | `/pumps/{id}/signal` | Live signal data |
| POST | `/collection/start` | Start labeled data collection |
| POST | `/collection/stop` | Stop data collection |

## Tech Stack

- **Hardware:** ESP32-S3, BMA400 accelerometer, SPU0410 microphone, Sovev (iPipe) board
- **Backend:** Python, FastAPI, NumPy, scikit-learn, SQLAlchemy, SQLite
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Deployment:** Railway (backend), GitHub Pages (dashboard)
- **Firmware:** Arduino (C++)

## Team

| Name | Email |
|---|---|
| Mohamad Nassar | mohamd.nassar@mail.huji.ac.il |
| Yousif Taha | yousif.taha@mail.huji.ac.il |
| Eias Taha | eias.taha@mail.huji.ac.il |

**Advisor:** Dr. Menashe Rajuan — iPIPE Ltd.
**Mentor:** Dr. Shimon Mizrahi

## License

This project was developed as part of the Engineering Project and Workshops II course at The Hebrew University of Jerusalem, in collaboration with Sovev (iPIPE Ltd.).
