"""
High-performance screen capture tool for YOLO training data collection.

Architecture:
  - Main thread: tiny tkinter overlay showing toggle state
  - Capture thread: polls input via Win32 GetAsyncKeyState, grabs screen via mss (DXGI)
  - Writer threads: pool that encodes + saves images from a queue, never blocking capture

Controls:
  - Numpad 5: toggle capture armed/disarmed
  - Left mouse button held: fires captures while armed
  - Escape: quit

Performance notes:
  - mss uses DXGI on Windows (hardware-accelerated screen capture)
  - GetAsyncKeyState high-bit polling works globally, even under fullscreen games
  - Capture region is pre-computed once (center 600x416 of primary monitor)
  - Writer threads handle all encoding/disk I/O off the capture thread
  - Queue acts as backpressure buffer
"""

import ctypes
import ctypes.wintypes
import logging
import logging.handlers
import mss
import mss.tools
import cv2
import numpy as np
import os
import sys
import threading
import time
import tkinter as tk
from collections import deque
from queue import Queue, Full
from datetime import datetime

# ── Self-elevate to Administrator (required for global input hooks in games) ──
def _is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

if not _is_admin():
    # Re-launch this script elevated via UAC prompt
    script = os.path.abspath(sys.argv[0])
    params = " ".join(f'"{a}"' for a in sys.argv[1:])
    # ShellExecuteW returns >32 on success
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, f'"{script}" {params}'.strip(), None, 1
    )
    if ret <= 32:
        print(f"ERROR: UAC elevation failed (code {ret}). Right-click → Run as administrator.")
    sys.exit(0)

# ── Configuration ──────────────────────────────────────────────────────────────
CAPTURE_WIDTH = 600
CAPTURE_HEIGHT = 416
CAPTURES_PER_SECOND_LMB = 10    # left click capture rate
CAPTURES_PER_SECOND_RMB = 5     # right click capture rate
CAPTURES_PER_SECOND_AUTO = 2    # auto-capture rate (Numpad 6 toggle)
IMAGE_FORMAT = ".jpg"           # .jpg is ~10x faster to encode than .png
JPEG_QUALITY = 95               # high quality for training data
WRITER_THREADS = 4              # parallel disk writers
QUEUE_MAX_SIZE = 200            # backpressure buffer
CAPTURES_PER_SECOND = CAPTURES_PER_SECOND_LMB  # default for fps_window sizing
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 3              # keep 3 rotated files (40 MB total max)

# ── Performance logger setup ───────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)

perf_logger = logging.getLogger("vision.perf")
perf_logger.setLevel(logging.DEBUG)
perf_logger.propagate = False

_perf_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "perf.log"),
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_perf_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
))
perf_logger.addHandler(_perf_handler)

app_logger = logging.getLogger("vision.app")
app_logger.setLevel(logging.INFO)
app_logger.propagate = False

_app_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_app_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(threadName)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
app_logger.addHandler(_app_handler)

# Also log app events to console
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
_console_handler.setLevel(logging.INFO)
app_logger.addHandler(_console_handler)


# ── Nanosecond stopwatch helper ────────────────────────────────────────────────
def _ns() -> int:
    """Return current time in nanoseconds (monotonic, high-resolution)."""
    return time.perf_counter_ns()


# ── Rolling performance stats ──────────────────────────────────────────────────
class PerfStats:
    """Thread-safe accumulator for nanosecond timings per operation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, list[int]] = {}  # op_name -> list of ns durations

    def record(self, op: str, duration_ns: int):
        with self._lock:
            if op not in self._data:
                self._data[op] = []
            self._data[op].append(duration_ns)

    def flush_summary(self) -> dict[str, dict]:
        """Drain all accumulated stats and return summary per op."""
        with self._lock:
            snapshot = self._data
            self._data = {}
        summary = {}
        for op, times in snapshot.items():
            if not times:
                continue
            times_sorted = sorted(times)
            n = len(times_sorted)
            summary[op] = {
                "count": n,
                "min_ns": times_sorted[0],
                "max_ns": times_sorted[-1],
                "avg_ns": sum(times_sorted) // n,
                "median_ns": times_sorted[n // 2],
                "p95_ns": times_sorted[int(n * 0.95)] if n >= 20 else times_sorted[-1],
                "p99_ns": times_sorted[int(n * 0.99)] if n >= 100 else times_sorted[-1],
            }
        return summary


perf_stats = PerfStats()


def _fmt_ns(ns: int) -> str:
    """Format nanoseconds into human-readable string."""
    if ns < 1_000:
        return f"{ns}ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.1f}µs"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    else:
        return f"{ns / 1_000_000_000:.3f}s"


# Win32 virtual key codes
VK_NUMPAD5 = 0x65
VK_NUMPAD6 = 0x66
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02

# ── Win32 input (GetAsyncKeyState polling — proven via diagnostic as admin) ──
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short


class GlobalInputMonitor:
    """
    Pure GetAsyncKeyState polling — queries physical key/button state directly.
    Works globally (including fullscreen games) when running as Administrator.
    No hooks, no message pump, no callbacks — zero GIL contention risk.
    Called from the capture thread on each loop iteration.
    """

    def __init__(self):
        self._prev_numpad5 = False
        self._numpad5_triggered = False
        self._prev_numpad6 = False
        self._numpad6_triggered = False

    def start(self):
        app_logger.info("Input monitor started (GetAsyncKeyState polling, admin=%s)",
                        bool(ctypes.windll.shell32.IsUserAnAdmin()))

    def stop(self):
        app_logger.info("Input monitor stopped")

    def poll(self):
        """Call once per loop iteration to update edge-detected key states."""
        t0 = _ns()

        # Numpad 5 — rising edge detect
        num5_now = bool(user32.GetAsyncKeyState(VK_NUMPAD5) & 0x8000)
        if num5_now and not self._prev_numpad5:
            self._numpad5_triggered = True
        self._prev_numpad5 = num5_now

        # Numpad 6 — rising edge detect
        num6_now = bool(user32.GetAsyncKeyState(VK_NUMPAD6) & 0x8000)
        if num6_now and not self._prev_numpad6:
            self._numpad6_triggered = True
        self._prev_numpad6 = num6_now

        perf_stats.record("input_poll", _ns() - t0)

    def consume_numpad5(self) -> bool:
        t0 = _ns()
        val = self._numpad5_triggered
        self._numpad5_triggered = False
        perf_stats.record("consume_numpad5", _ns() - t0)
        return val

    def is_lmb_down(self) -> bool:
        t0 = _ns()
        val = bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        perf_stats.record("is_lmb_down", _ns() - t0)
        return val

    def is_rmb_down(self) -> bool:
        t0 = _ns()
        val = bool(user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        perf_stats.record("is_rmb_down", _ns() - t0)
        return val

    def consume_numpad6(self) -> bool:
        t0 = _ns()
        val = self._numpad6_triggered
        self._numpad6_triggered = False
        perf_stats.record("consume_numpad6", _ns() - t0)
        return val


# ── Compute capture region (center of primary monitor) ─────────────────────────
def get_capture_region() -> dict:
    """Return mss-compatible region dict for the center of the primary monitor."""
    t0 = _ns()
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    left = (screen_w - CAPTURE_WIDTH) // 2
    top = (screen_h - CAPTURE_HEIGHT) // 2

    region = {
        "left": left,
        "top": top,
        "width": CAPTURE_WIDTH,
        "height": CAPTURE_HEIGHT,
    }
    dur = _ns() - t0
    perf_logger.debug("get_capture_region: %s  region=%s", _fmt_ns(dur), region)
    return region


# ── Writer pool ────────────────────────────────────────────────────────────────
class WriterPool:
    """Pool of threads that encode and save images from a queue."""

    def __init__(self, num_threads: int, output_dir: str):
        self.queue: Queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.output_dir = output_dir
        self.stop_event = threading.Event()
        self.saved_count = 0
        self._count_lock = threading.Lock()
        self.threads = []

        os.makedirs(output_dir, exist_ok=True)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if IMAGE_FORMAT == ".jpg" else []
        self._encode_params = encode_params

        for i in range(num_threads):
            t = threading.Thread(target=self._writer_loop, name=f"writer-{i}", daemon=True)
            t.start()
            self.threads.append(t)

    def enqueue(self, frame: np.ndarray, timestamp: float) -> bool:
        """Put a frame on the queue. Returns False if queue is full (backpressure)."""
        t0 = _ns()
        try:
            self.queue.put_nowait((frame, timestamp))
            dur = _ns() - t0
            perf_stats.record("enqueue", dur)
            perf_logger.debug("enqueue: %s  qsize=%d", _fmt_ns(dur), self.queue.qsize())
            return True
        except Full:
            dur = _ns() - t0
            perf_logger.warning("enqueue DROPPED (full): %s  qsize=%d", _fmt_ns(dur), self.queue.qsize())
            perf_stats.record("enqueue_drop", dur)
            return False

    def _writer_loop(self):
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except Exception:
                continue
            if item is None:
                break
            frame, timestamp = item
            self._save(frame, timestamp)
            self.queue.task_done()

    def _save(self, frame: np.ndarray, timestamp: float):
        t0_total = _ns()

        # Format: capture_YYYYMMDD_HHMMSS_microseconds.jpg
        dt = datetime.fromtimestamp(timestamp)
        filename = f"cap_{dt.strftime('%Y%m%d_%H%M%S')}_{dt.microsecond:06d}{IMAGE_FORMAT}"
        filepath = os.path.join(self.output_dir, filename)

        # Encode timing
        t0_encode = _ns()
        cv2.imwrite(filepath, frame, self._encode_params)
        dur_encode = _ns() - t0_encode

        with self._count_lock:
            self.saved_count += 1

        dur_total = _ns() - t0_total
        perf_stats.record("save_encode", dur_encode)
        perf_stats.record("save_total", dur_total)
        perf_logger.debug(
            "save: total=%s  encode=%s  file=%s",
            _fmt_ns(dur_total), _fmt_ns(dur_encode), filename,
        )

    def shutdown(self):
        self.stop_event.set()
        # Drain queue
        for _ in self.threads:
            try:
                self.queue.put_nowait(None)
            except Full:
                pass
        for t in self.threads:
            t.join(timeout=2)


# ── Capture thread ─────────────────────────────────────────────────────────────
class CaptureEngine:
    def __init__(self, writer: WriterPool, overlay_callback, input_mon: GlobalInputMonitor):
        self.writer = writer
        self.overlay_callback = overlay_callback  # called with (armed, capturing, fps, saved)
        self.input = input_mon
        self.armed = False
        self.auto_capture = False
        self.stop_event = threading.Event()
        self.capture_count = 0
        self.fps_actual = 0.0
        self.region = get_capture_region()
        self._thread = threading.Thread(target=self._run, name="capture-engine", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self._thread.join(timeout=3)

    def _run(self):
        interval_lmb = 1.0 / CAPTURES_PER_SECOND_LMB
        interval_rmb = 1.0 / CAPTURES_PER_SECOND_RMB
        interval_auto = 1.0 / CAPTURES_PER_SECOND_AUTO
        fps_window: deque = deque(maxlen=CAPTURES_PER_SECOND_LMB * 2)

        app_logger.info("Capture engine started  LMB=%.1fms  RMB=%.1fms  AUTO=%.1fms  region=%s",
                        interval_lmb * 1000, interval_rmb * 1000, interval_auto * 1000, self.region)

        with mss.mss() as sct:
            last_capture_time = 0.0
            last_overlay_update = 0.0
            last_stats_flush = time.perf_counter()

            while not self.stop_event.is_set():
                t0_loop = _ns()
                now = time.perf_counter()

                # ── Poll physical key states (edge detection for toggles) ──
                self.input.poll()

                # ── Toggle detection (Numpad 5 press) ──
                if self.input.consume_numpad5():
                    self.armed = not self.armed
                    app_logger.info("Toggle: %s", "ARMED" if self.armed else "DISARMED")

                # ── Toggle auto-capture (Numpad 6 press) ──
                if self.input.consume_numpad6():
                    self.auto_capture = not self.auto_capture
                    if self.auto_capture:
                        self.armed = True  # auto-capture implies armed
                    app_logger.info("Auto-capture: %s", "ON" if self.auto_capture else "OFF")

                # ── Capture logic (LMB=10fps priority, RMB=5fps, auto=2fps) ──
                capturing = False
                lmb = self.input.is_lmb_down()
                rmb = self.input.is_rmb_down()
                should_capture = (self.armed and (lmb or rmb)) or self.auto_capture
                if should_capture:
                    if lmb:
                        interval = interval_lmb
                    elif rmb:
                        interval = interval_rmb
                    else:
                        interval = interval_auto
                    elapsed = now - last_capture_time
                    if elapsed >= interval:
                        # Grab screen region (DXGI accelerated)
                        t0_grab = _ns()
                        shot = sct.grab(self.region)
                        dur_grab = _ns() - t0_grab

                        # Convert to numpy BGR (cv2 format) — mss gives BGRA
                        t0_convert = _ns()
                        frame = np.asarray(shot)[:, :, :3].copy()
                        dur_convert = _ns() - t0_convert

                        # Non-blocking enqueue
                        t0_enq = _ns()
                        enqueued = self.writer.enqueue(frame, time.time())
                        dur_enq = _ns() - t0_enq

                        self.capture_count += 1
                        fps_window.append(now)
                        last_capture_time = now
                        capturing = True

                        perf_stats.record("grab", dur_grab)
                        perf_stats.record("convert_bgra_to_bgr", dur_convert)
                        perf_logger.debug(
                            "CAPTURE #%d  grab=%s  convert=%s  enqueue=%s  queued=%s",
                            self.capture_count,
                            _fmt_ns(dur_grab),
                            _fmt_ns(dur_convert),
                            _fmt_ns(dur_enq),
                            "yes" if enqueued else "DROPPED",
                        )

                # ── Compute actual FPS ──
                if fps_window and len(fps_window) >= 2:
                    span = fps_window[-1] - fps_window[0]
                    if span > 0:
                        self.fps_actual = (len(fps_window) - 1) / span
                elif not fps_window:
                    self.fps_actual = 0.0

                # ── Update overlay at ~10 Hz ──
                if now - last_overlay_update > 0.1:
                    self.overlay_callback(
                        self.armed,
                        capturing,
                        self.fps_actual,
                        self.writer.saved_count,
                        self.auto_capture,
                    )
                    last_overlay_update = now

                # ── Periodic performance summary (every 5 seconds) ──
                if now - last_stats_flush >= 5.0:
                    summary = perf_stats.flush_summary()
                    if summary:
                        perf_logger.info("═══ PERF SUMMARY (last 5s) ═══")
                        for op, s in sorted(summary.items()):
                            perf_logger.info(
                                "  %-22s  n=%-6d  min=%-10s  avg=%-10s  med=%-10s  p95=%-10s  p99=%-10s  max=%s",
                                op, s["count"],
                                _fmt_ns(s["min_ns"]), _fmt_ns(s["avg_ns"]),
                                _fmt_ns(s["median_ns"]), _fmt_ns(s["p95_ns"]),
                                _fmt_ns(s["p99_ns"]), _fmt_ns(s["max_ns"]),
                            )
                        perf_logger.info("═══ END SUMMARY  saved=%d  queue=%d ═══",
                                         self.writer.saved_count, self.writer.queue.qsize())
                    last_stats_flush = now

                # ── Log loop iteration time ──
                dur_loop = _ns() - t0_loop
                perf_stats.record("loop_iteration", dur_loop)

                # Sleep a tiny bit to avoid burning CPU when idle
                # When armed+clicking, we poll tighter
                if should_capture:
                    if lmb:
                        active_interval = interval_lmb
                    elif rmb:
                        active_interval = interval_rmb
                    else:
                        active_interval = interval_auto
                    remaining = active_interval - (time.perf_counter() - last_capture_time)
                    if remaining > 0.002:
                        time.sleep(remaining * 0.5)  # sleep half, spin the rest
                else:
                    time.sleep(0.005)  # 5ms idle poll — ~200 Hz input check

        app_logger.info("Capture engine stopped  total_captures=%d", self.capture_count)


# ── Overlay (tiny always-on-top window) ────────────────────────────────────────
class Overlay:
    """Minimal always-on-top status indicator."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vision Capture")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(False)
        self.root.geometry("260x90+10+10")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")

        # Make window click-through friendly — small, out of the way
        self.status_label = tk.Label(
            self.root,
            text="DISARMED",
            font=("Consolas", 16, "bold"),
            bg="#1e1e1e",
            fg="#ff4444",
        )
        self.status_label.pack(pady=(8, 2))

        self.info_label = tk.Label(
            self.root,
            text="FPS: 0.0 | Saved: 0",
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#aaaaaa",
        )
        self.info_label.pack()

        self.hint_label = tk.Label(
            self.root,
            text="[Num5] arm  [Num6] auto  [LMB/RMB] capture",
            font=("Consolas", 7),
            bg="#1e1e1e",
            fg="#666666",
        )
        self.hint_label.pack(pady=(2, 0))

        self._pending_update = None

    def update_status(self, armed: bool, capturing: bool, fps: float, saved: int, auto: bool = False):
        """Thread-safe overlay update via tkinter's thread queue."""
        # Avoid flooding the tk event queue
        def _do_update():
            if auto:
                self.status_label.config(text="● AUTO", fg="#00cccc")
            elif armed and capturing:
                self.status_label.config(text="● CAPTURING", fg="#00ff00")
            elif armed:
                self.status_label.config(text="ARMED", fg="#ffaa00")
            else:
                self.status_label.config(text="DISARMED", fg="#ff4444")
            self.info_label.config(text=f"FPS: {fps:.1f} | Saved: {saved}")

        try:
            self.root.after_idle(_do_update)
        except Exception:
            pass  # window may be closing

    def run(self):
        self.root.mainloop()

    def close(self):
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    app_logger.info("Vision Capture Tool starting")
    app_logger.info("  Capture region : %dx%d (center of primary monitor)", CAPTURE_WIDTH, CAPTURE_HEIGHT)
    app_logger.info("  Rate           : %d captures/sec", CAPTURES_PER_SECOND)
    app_logger.info("  Output         : %s", OUTPUT_DIR)
    app_logger.info("  Logs           : %s", LOG_DIR)
    app_logger.info("  Format         : %s (quality=%d)", IMAGE_FORMAT, JPEG_QUALITY)
    app_logger.info("  Writer threads : %d", WRITER_THREADS)
    app_logger.info("  Controls       : Numpad5=toggle, LMB=capture, Esc=quit")

    print()  # blank line for console readability

    region = get_capture_region()
    app_logger.info("  Monitor region : left=%d, top=%d", region["left"], region["top"])
    print()

    # Start global input hooks (works even under fullscreen games)
    input_mon = GlobalInputMonitor()
    input_mon.start()

    # Create overlay
    overlay = Overlay()

    # Create writer pool
    writer = WriterPool(num_threads=WRITER_THREADS, output_dir=OUTPUT_DIR)

    # Create capture engine
    engine = CaptureEngine(writer=writer, overlay_callback=overlay.update_status, input_mon=input_mon)
    engine.start()

    # Run overlay on main thread (tkinter requires main thread)
    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        app_logger.info("Shutting down...")
        engine.stop()
        input_mon.stop()
        # Flush final perf stats
        summary = perf_stats.flush_summary()
        if summary:
            perf_logger.info("═══ FINAL PERF SUMMARY ═══")
            for op, s in sorted(summary.items()):
                perf_logger.info(
                    "  %-22s  n=%-6d  min=%-10s  avg=%-10s  med=%-10s  p95=%-10s  p99=%-10s  max=%s",
                    op, s["count"],
                    _fmt_ns(s["min_ns"]), _fmt_ns(s["avg_ns"]),
                    _fmt_ns(s["median_ns"]), _fmt_ns(s["p95_ns"]),
                    _fmt_ns(s["p99_ns"]), _fmt_ns(s["max_ns"]),
                )
            perf_logger.info("═══ END FINAL SUMMARY ═══")
        writer.shutdown()
        app_logger.info("Total frames saved: %d", writer.saved_count)
        print(f"\nTotal frames saved: {writer.saved_count}")
        print(f"Perf logs: {os.path.join(LOG_DIR, 'perf.log')}")


if __name__ == "__main__":
    main()
