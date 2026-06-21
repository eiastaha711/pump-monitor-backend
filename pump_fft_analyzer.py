"""
pump_fft_analyzer.py  v1.7
Real-time FFT analyzer for pump health monitoring — PMM_V1 board.

Two streams on one serial port:
  ACC  — ASCII  START / a:ax,ay,az × 512 / END   @ 800 Hz, 512 samples
  MIC  — Binary 0xAA 0x55 + uint16 LE × 1024     @ 8000 Hz, 1024 samples

v1.7 changes:
  - ACC_FS changed from 4000 → 800 to match firmware v2.0 (BMA400 800 Hz ODR)
  - Acc frequency axis now 0–400 Hz instead of 0–2000 Hz

Usage:
  python pump_fft_analyzer.py --port COM4
  python pump_fft_analyzer.py --port /dev/ttyUSB0 --rpm 1500 --blades 5
  python pump_fft_analyzer.py --port COM4 --log-dir data/run1 --no-log   (disable logging)
"""

import argparse
import csv
import os
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import serial
import serial.tools.list_ports

# ── Config ────────────────────────────────────────────────────────────────────
BAUD         = 921600

ACC_N        = 512
ACC_FS       = 800
ACC_SCALE    = 4.0 / 2048.0    # raw int16 → g  (±4g, 12-bit)

MIC_N        = 1024
MIC_FS       = 8000             # different from ACC — separate freq axis
MIC_FRAME    = MIC_N * 2       # bytes per binary frame

RPM          = 1450
N_BLADES     = 6
F1           = RPM / 60
BPF          = F1 * N_BLADES

# Fixed Y-axis limits — adjust after first real run if needed
ACC_YLIM     = (0.0, 0.2)       # g
MIC_YLIM     = (0.0, 3)       # matches mic_test.py ylim(0, 50)

# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock    = threading.Lock()
        self.ax      = np.zeros(ACC_N)
        self.ay      = np.zeros(ACC_N)
        self.az      = np.zeros(ACC_N)
        self.mic     = np.zeros(MIC_N)
        self.acc_rdy = False
        self.mic_rdy = False

state = State()

# ── CSV logger ────────────────────────────────────────────────────────────────
class FFTLogger:
    """
    Saves one CSV per captured frame with columns:
        freq_x, mag_x, freq_y, mag_y, freq_z, mag_z, freq_mic, mag_mic

    ACC FFT bins (257) and MIC FFT bins (513) differ in length since the two
    streams run at different sample rates / N. Shorter columns are padded
    with empty cells so the file stays a single rectangular CSV.
    """
    def __init__(self, log_dir, enabled=True):
        self.enabled = enabled
        self.log_dir = log_dir
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[Logger] Saving FFT frames to: {os.path.abspath(self.log_dir)}")

    def save_frame(self, freq_acc, mag_x, mag_y, mag_z, freq_mic, mag_mic):
        if not self.enabled:
            return

        n_acc = len(freq_acc)
        n_mic = len(freq_mic)
        n_rows = max(n_acc, n_mic)

        fname = f"fft_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}.csv"
        fpath = os.path.join(self.log_dir, fname)

        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["freq_x", "mag_x", "freq_y", "mag_y",
                              "freq_z", "mag_z", "freq_mic", "mag_mic"])
            for i in range(n_rows):
                row = [
                    freq_acc[i] if i < n_acc else "",
                    mag_x[i]    if i < n_acc else "",
                    freq_acc[i] if i < n_acc else "",
                    mag_y[i]    if i < n_acc else "",
                    freq_acc[i] if i < n_acc else "",
                    mag_z[i]    if i < n_acc else "",
                    freq_mic[i] if i < n_mic else "",
                    mag_mic[i]  if i < n_mic else "",
                ]
                writer.writerow(row)

        return fpath


# ── Serial reader — both streams in one thread ────────────────────────────────
class SerialReader:
    def __init__(self, port):
        self.ser    = serial.Serial(port, BAUD, timeout=2)
        self._stop  = False
        self._buf   = b""          # line assembly buffer for ASCII stream
        self._prev  = b"\x00"     # previous byte for binary header detection
        self._ax_tmp, self._ay_tmp, self._az_tmp = [], [], []
        self._in_acc = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        print(f"[Serial] Connected on {port}")

    def start(self): self.thread.start()
    def stop(self):  self._stop = True

    def _read_exact(self, n):
        data = b""
        while len(data) < n and not self._stop:
            chunk = self.ser.read(n - len(data))
            if chunk:
                data += chunk
        return data if len(data) == n else None

    def _handle_line(self, line):
        # Skip comment lines printed by Arduino (e.g. "# MIC Peak: 341.8 Hz")
        if line.startswith('#'):
            print(f"[Arduino] {line}")   # echo to console for debugging
            return

        if line == "START":
            self._ax_tmp, self._ay_tmp, self._az_tmp = [], [], []
            self._in_acc = True
            return

        if line == "END" and self._in_acc:
            if len(self._ax_tmp) == ACC_N:
                with state.lock:
                    state.ax      = np.array(self._ax_tmp, dtype=float) * ACC_SCALE
                    state.ay      = np.array(self._ay_tmp, dtype=float) * ACC_SCALE
                    state.az      = np.array(self._az_tmp, dtype=float) * ACC_SCALE
                    state.acc_rdy = True
            self._in_acc = False
            return

        if self._in_acc and line.startswith('a:'):
            try:
                p = line[2:].split(',')
                self._ax_tmp.append(int(p[0]))
                self._ay_tmp.append(int(p[1]))
                self._az_tmp.append(int(p[2]))
            except (ValueError, IndexError):
                pass

    def _run(self):
        while not self._stop:
            try:
                byte = self.ser.read(1)
            except Exception:
                continue
            if not byte:
                continue

            # ── Detect MIC binary header 0xAA 0x55 ───────────────────────
            if self._prev == b"\xaa" and byte == b"\x55":
                # Discard any partial ASCII line being assembled
                self._buf = b""
                # Read binary payload
                payload = self._read_exact(MIC_FRAME)
                if payload:
                    samples = np.frombuffer(payload, dtype=np.uint16).astype(np.float32)
                    with state.lock:
                        state.mic     = samples   # raw uint16, DC removed in compute_fft
                        state.mic_rdy = True
                self._prev = b"\x00"
                continue

            self._prev = byte

            # ── Assemble ASCII lines byte by byte ─────────────────────────
            if byte == b"\n":
                line = self._buf.decode('ascii', errors='ignore').strip()
                self._buf = b""
                if line:
                    self._handle_line(line)
            else:
                # Reject 0xAA early so it doesn't corrupt lines
                if byte != b"\xaa":
                    self._buf += byte


# ── DSP ───────────────────────────────────────────────────────────────────────
def compute_fft_acc(signal):
    """Hann-windowed FFT for accelerometer (float g). Returns (freqs, magnitude in g)."""
    sig  = signal.astype(np.float64)
    sig -= sig.mean()
    win  = np.hanning(len(sig))
    mag  = np.abs(np.fft.rfft(sig * win)) / (len(sig) / 2)
    freq = np.fft.rfftfreq(len(sig), 1.0 / ACC_FS)
    return freq, mag


def compute_fft_mic(signal):
    """Hann-windowed FFT for microphone. Matches mic_test.py exactly."""
    sig  = signal.astype(np.float32)
    sig -= sig.mean()                        # remove DC (~2048)
    win  = np.hanning(len(sig))
    mag  = np.abs(np.fft.rfft(sig * win)) / (MIC_N / 2)   # same as mic_test.py
    freq = np.fft.rfftfreq(len(sig), 1.0 / MIC_FS)
    return freq, mag


def band_energy(freq, mag, flo, fhi):
    idx = (freq >= flo) & (freq <= fhi)
    return float(np.sqrt(np.mean(mag[idx]**2))) if idx.any() else 0.0


def peak_near(freq, mag, target, tol=3.0):
    idx = (freq >= target - tol) & (freq <= target + tol)
    return float(mag[idx].max()) if idx.any() else 0.0


def orient_from_acc(ax, ay, az):
    mx, my, mz = ax.mean(), ay.mean(), az.mean()
    roll  = np.degrees(np.arctan2(my, mz))
    pitch = np.degrees(np.arctan2(-mx, np.sqrt(my**2 + mz**2)))
    return roll, pitch


def classify_faults(freq_acc, freq_mic, mic_mag, ax_mag, ay_mag, az_mag):
    faults = []

    radial_e = (band_energy(freq_acc, ax_mag, 1, 400) +
                band_energy(freq_acc, ay_mag, 1, 400)) / 2
    axial_e  =  band_energy(freq_acc, az_mag, 1, 400)

    p1x = max(peak_near(freq_acc, ax_mag, F1),
              peak_near(freq_acc, ay_mag, F1))
    if radial_e > 1e-6:
        conf = min(p1x / (radial_e + 1e-9), 1.0)
        if conf > 0.15:
            faults.append(("Imbalance", conf,
                f"1X peak {p1x:.4f} g @ {F1:.1f} Hz"))

    p2x = max(peak_near(freq_acc, ax_mag, 2*F1),
              peak_near(freq_acc, ay_mag, 2*F1))
    axial_ratio   = axial_e / (radial_e + 1e-9)
    misalign_conf = min((p2x / (p1x + 1e-9)) * 0.5 + axial_ratio * 0.5, 1.0)
    if misalign_conf > 0.1:
        faults.append(("Misalignment", misalign_conf,
            f"2X/1X={p2x/(p1x+1e-9):.2f}, axial/radial={axial_ratio:.2f}"))

    # Cavitation — broadband mic energy + BPF sidebands (mic uses its own freq axis)
    bb_mic    = band_energy(freq_mic, mic_mag, 200, MIC_FS // 2)
    bpf_sb    = (peak_near(freq_mic, mic_mag, BPF + F1) +
                 peak_near(freq_mic, mic_mag, BPF - F1))
    cav_conf  = min(bb_mic * 0.0002 + bpf_sb * 0.0003, 1.0)
    if cav_conf > 0.05:
        faults.append(("Cavitation", cav_conf,
            f"Broadband mic {bb_mic:.0f}, BPF sidebands {bpf_sb:.0f}"))

    hf_energy = band_energy(freq_acc, ax_mag, 200, ACC_FS // 2)
    hf_ratio  = hf_energy / (band_energy(freq_acc, ax_mag, 1, 200) + 1e-9)
    bear_conf = min(hf_ratio * 2, 1.0)
    if bear_conf > 0.1:
        faults.append(("Bearing fault", bear_conf,
            f"HF/LF ratio {hf_ratio:.2f} (>200 Hz)"))

    sub       = max(peak_near(freq_acc, ax_mag, 0.5*F1),
                    peak_near(freq_acc, ay_mag, 0.5*F1))
    p3x       = max(peak_near(freq_acc, ax_mag, 3*F1),
                    peak_near(freq_acc, ay_mag, 3*F1))
    loose_conf = min((sub + p3x) / (p1x + 1e-9), 1.0)
    if loose_conf > 0.1:
        faults.append(("Structural looseness", loose_conf,
            f"0.5X={sub:.4f} g, 3X={p3x:.4f} g"))

    faults.sort(key=lambda x: x[1], reverse=True)
    return faults or [("Healthy", 1.0, "No significant fault signatures")]


# ── Figure ────────────────────────────────────────────────────────────────────
def build_figure():
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
        'axes.edgecolor':   '#444',    'axes.labelcolor': '#ccc',
        'xtick.color':      '#999',    'ytick.color':     '#999',
        'text.color':       '#eee',    'grid.color':      '#2a2a4a',
        'grid.linewidth':   0.5,
    })

    fig = plt.figure(figsize=(15, 9), facecolor='#1a1a2e')
    fig.suptitle('PMM — Pump Health Vibration Analyzer', fontsize=14,
                 color='#eee', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(
        5, 2, figure=fig,
        left=0.07, right=0.97, top=0.94, bottom=0.06,
        hspace=0.10, wspace=0.38,
        height_ratios=[1, 1, 1, 1, 1.4]
    )

    ax_x   = fig.add_subplot(gs[0, 0])
    ax_y   = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_z   = fig.add_subplot(gs[2, 0], sharex=ax_x)
    ax_m   = fig.add_subplot(gs[3, 0])          # NOT shared — different Fs & N
    ax_ori = fig.add_subplot(gs[0:2, 1])
    ax_fa  = fig.add_subplot(gs[2:5, 1])
    ax_bar = fig.add_subplot(gs[4, 0])

    # ── ACC axes (shared X: 0 – ACC_FS/2 Hz) ─────────────────────────────
    acc_cfgs = [
        (ax_x, '#4da6ff', 'Acc X — Horizontal (g)'),
        (ax_y, '#50fa7b', 'Acc Y — Vertical (g)'),
        (ax_z, '#ff9d42', 'Acc Z — Axial (g)'),
    ]
    for ax, col, lbl in acc_cfgs:
        ax.set_ylabel(lbl, fontsize=7, color=col)
        ax.yaxis.label.set_color(col)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlim(0, ACC_FS // 2)     # 0 – 2000 Hz
        ax.set_ylim(*ACC_YLIM)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)

    # shared X label on bottom ACC axis
    ax_z.set_xlabel('Frequency (Hz)', fontsize=7, color='#ccc')
    plt.setp(ax_z.get_xticklabels(), visible=True)

    # ── MIC axis (independent X: 0 – MIC_FS/2 Hz, raw FFT counts) ───────
    ax_m.set_ylabel('Mic — FFT magnitude', fontsize=7, color='#cc99ff')
    ax_m.yaxis.label.set_color('#cc99ff')
    ax_m.tick_params(axis='y', labelsize=7, colors='#cc99ff')
    ax_m.tick_params(axis='x', labelsize=7)
    ax_m.set_xlim(0, MIC_FS // 2)      # 0 – 4000 Hz
    ax_m.set_ylim(*MIC_YLIM)
    ax_m.grid(True, alpha=0.3)
    ax_m.set_xlabel('Frequency (Hz)', fontsize=7, color='#ccc')

    # ── Right panels ──────────────────────────────────────────────────────
    ax_ori.set_facecolor('#0d1117'); ax_ori.axis('off')
    ax_ori.set_title('Sensor orientation', fontsize=9, color='#aaa', pad=4)
    ax_fa.set_facecolor('#0d1117');  ax_fa.axis('off')
    ax_fa.set_title('Fault analysis',      fontsize=9, color='#aaa', pad=4)
    ax_bar.axis('off')

    return fig, ax_x, ax_y, ax_z, ax_m, ax_ori, ax_fa, ax_bar


def init_lines(ax_x, ax_y, ax_z, ax_m):
    colors = ['#4da6ff', '#50fa7b', '#ff9d42']
    lines  = []

    for ax, col in zip([ax_x, ax_y, ax_z], colors):
        ln, = ax.plot([], [], color=col, lw=0.9, alpha=0.9)
        ax.axvline(F1,    color='#ff5555', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(2*F1,  color='#ff5555', lw=0.6, ls=':',  alpha=0.4)
        ax.axvline(BPF,   color='#ffb347', lw=0.8, ls='--', alpha=0.6)
        lines.append(ln)

    # Labels on top ACC axis only
    ax_x.text(F1   + 10, ACC_YLIM[1]*0.88, '1X',  color='#ff5555', fontsize=6)
    ax_x.text(2*F1 + 10, ACC_YLIM[1]*0.88, '2X',  color='#ff5555', fontsize=6)
    ax_x.text(BPF  + 10, ACC_YLIM[1]*0.88, 'BPF', color='#ffb347', fontsize=6)

    # MIC line — independent freq axis (0–4000 Hz)
    ln_m, = ax_m.plot([], [], color='#cc99ff', lw=0.9, alpha=0.9)
    ax_m.axvline(F1,  color='#ff5555', lw=0.8, ls='--', alpha=0.6)
    ax_m.axvline(BPF, color='#ffb347', lw=0.8, ls='--', alpha=0.6)
    ax_m.text(F1  + 20, MIC_YLIM[1]*0.88, '1X',  color='#ff5555', fontsize=6)
    ax_m.text(BPF + 20, MIC_YLIM[1]*0.88, 'BPF', color='#ffb347', fontsize=6)
    lines.append(ln_m)

    return lines   # [ln_x, ln_y, ln_z, ln_mic]


# ── Orientation ───────────────────────────────────────────────────────────────
def draw_orientation(ax_ori, roll_deg, pitch_deg):
    ax_ori.cla()
    ax_ori.set_facecolor('#0d1117')
    ax_ori.set_xlim(-1.5, 1.5); ax_ori.set_ylim(-1.2, 1.2)
    ax_ori.set_aspect('equal'); ax_ori.axis('off')
    ax_ori.set_title(f'Sensor orientation   Roll {roll_deg:+.1f}°  Pitch {pitch_deg:+.1f}°',
                     fontsize=8, color='#aaa', pad=4)
    roll  = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    corners = np.array([[-0.8,-0.5,0],[0.8,-0.5,0],[0.8,0.5,0],[-0.8,0.5,0]])
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    proj = ((Ry @ Rx) @ corners.T).T
    pcb = plt.Polygon(list(zip(proj[:,0], proj[:,1])), closed=True,
                      facecolor='#1a3a5c', edgecolor='#4da6ff', linewidth=1.5)
    ax_ori.add_patch(pcb)
    R = Ry @ Rx
    for vec, col, lbl in [([1,0,0],'#4da6ff','X'),([0,1,0],'#50fa7b','Y'),([0,0,1],'#ff9d42','Z')]:
        end = R @ np.array(vec) * 0.7
        ax_ori.annotate('', xy=(end[0],end[1]), xytext=(0,0),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.8))
        ax_ori.text(end[0]*1.25, end[1]*1.25, lbl, color=col, fontsize=9,
                    ha='center', va='center', fontweight='bold')
    ax_ori.plot(0, 0, 'o', color='#fff', markersize=4, zorder=5)


# ── Fault panel ───────────────────────────────────────────────────────────────
def draw_faults(ax_fa, faults):
    ax_fa.cla()
    ax_fa.set_facecolor('#0d1117'); ax_fa.axis('off')
    ax_fa.set_title('Fault analysis', fontsize=9, color='#aaa', pad=4)
    legend_items = [('#ff5555', 'Risk'), ('#ffb347', 'Warning'), ('#50fa7b', 'Healthy')]
    lx = 0.08
    for col, label in legend_items:
        sq = plt.Rectangle((lx, 0.89), 0.045, 0.035,
                           transform=ax_fa.transAxes,
                           facecolor=col, edgecolor='none', clip_on=False)
        ax_fa.add_patch(sq)
        ax_fa.text(lx + 0.055, 0.907, label,
                   transform=ax_fa.transAxes,
                   fontsize=7, color='#aaa', va='center')
        lx += 0.30
    y = 0.8
    for i, (name, conf, desc) in enumerate(faults[:6]):
        bar_color = ('#ff5555' if i==0 and name != 'Healthy'
                     else '#ffb347' if conf > 0.4 else '#50fa7b')
        ax_fa.text(0.02, y, name , transform=ax_fa.transAxes,
                   fontsize=9, color=bar_color, fontweight='bold' if i==0 else 'normal')
        # ax_fa.text(0.72, y, f"{conf*100:.0f}%", transform=ax_fa.transAxes,
        #            fontsize=9, color=bar_color, ha='right')
        # ax_fa.barh(y-0.02, conf, left=0.0, height=0.04,
        #            color=bar_color, alpha=0.35, transform=ax_fa.transAxes)
        ax_fa.text(0.02, y-0.07, desc, transform=ax_fa.transAxes,
                   fontsize=7, color='#888')
        y -= 0.17
    ax_fa.text(0.02, 0.03, 'Ref: ISO 10816-7 / API 610',
               transform=ax_fa.transAxes, fontsize=6.5, color='#555', style='italic')


def draw_status(ax_bar, faults):
    ax_bar.cla(); ax_bar.axis('off')
    name, conf, _ = faults[0]
    if name == 'Healthy':
        color, label = '#50fa7b', '✔  Healthy'
    elif conf > 0.6:
        color, label = '#ff5555', f'⚠  DANGER — {name}'
    else:
        color, label = '#ffb347', f'⚠  Warning — {name}'
    ax_bar.set_facecolor(color + '33')
    ax_bar.text(0.5, 0.5, label, transform=ax_bar.transAxes,
                fontsize=12, color=color, fontweight='bold', ha='center', va='center')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',    type=str,   default=None)
    parser.add_argument('--rpm',     type=float, default=RPM)
    parser.add_argument('--blades',  type=int,   default=N_BLADES)
    parser.add_argument('--log-dir', type=str,   default='fft_logs',
                         help='Directory to save per-frame FFT CSV files into')
    parser.add_argument('--no-log',  action='store_true',
                         help='Disable CSV logging of FFT frames')
    args = parser.parse_args()

    global F1, BPF
    F1  = args.rpm / 60
    BPF = F1 * args.blades

    port = args.port
    if port is None:
        ports = serial.tools.list_ports.comports()
        esp32 = [p for p in ports if any(k in p.description
                 for k in ('CP210','CH340','JTAG','USB Serial'))]
        port = esp32[0].device if esp32 else (ports[0].device if ports else None)
        if port is None:
            print("[Error] No serial port found. Use --port."); return
        print(f"[Serial] Auto-detected: {port}")

    reader = SerialReader(port)
    reader.start()

    logger = FFTLogger(args.log_dir, enabled=not args.no_log)

    fig, ax_x, ax_y, ax_z, ax_m, ax_ori, ax_fa, ax_bar = build_figure()
    lines = init_lines(ax_x, ax_y, ax_z, ax_m)
    # lines = [ln_x, ln_y, ln_z, ln_mic]

    def update(_frame):
        with state.lock:
            acc_rdy = state.acc_rdy
            mic_rdy = state.mic_rdy
            ax_d  = state.ax.copy()
            ay_d  = state.ay.copy()
            az_d  = state.az.copy()
            mic_d = state.mic.copy()

        if not acc_rdy:
            return lines

        # ── ACC FFTs — fixed Y in g, 0–2000 Hz ───────────────────────────
        freq_acc, mag_x = compute_fft_acc(ax_d)
        _,        mag_y = compute_fft_acc(ay_d)
        _,        mag_z = compute_fft_acc(az_d)
        lines[0].set_data(freq_acc, mag_x)
        lines[1].set_data(freq_acc, mag_y)
        lines[2].set_data(freq_acc, mag_z)

        # ── MIC FFT — fixed Y in counts, 0–4000 Hz ───────────────────────
        if mic_rdy:
            freq_mic, mag_mic = compute_fft_mic(mic_d)
            lines[3].set_data(freq_mic, mag_mic)
        else:
            freq_mic = np.fft.rfftfreq(MIC_N, 1.0 / MIC_FS)
            mag_mic  = np.zeros(len(freq_mic))

        # ── Log the 4 FFTs (freq, magnitude) to CSV ──────────────────────
        logger.save_frame(freq_acc, mag_x, mag_y, mag_z, freq_mic, mag_mic)

        # ── Fault classification ──────────────────────────────────────────
        faults = classify_faults(freq_acc, freq_mic, mag_mic, mag_x, mag_y, mag_z)

        # ── Orientation ───────────────────────────────────────────────────
        roll, pitch = orient_from_acc(ax_d, ay_d, az_d)
        draw_orientation(ax_ori, roll, pitch)
        draw_faults(ax_fa, faults)
        draw_status(ax_bar, faults)

        return lines

    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.show()
    reader.stop()


if __name__ == '__main__':
    main()
