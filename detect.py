"""
Real-time YOLO detection agent.

Captures the center 600x416 region of the screen at ~30 FPS,
runs YOLO11n inference, and logs detections to console.

When armed (Numpad 5 toggle) and right mouse button is held,
instantly moves the cursor to the closest detected target center.

No on-screen UI — no overlay, no crosshairs, no text — so screen
captures stay clean.

Controls:
    Numpad 2 : toggle detection tick sound on/off
    Numpad 3 : toggle auto-harvest (randomly save 50% of frames with detections)
    Numpad 5 : toggle armed on/off
    Numpad 6 : toggle frame capture on/off
    RMB held : move cursor to nearest detection (+ save frames at 5 FPS if capture on)
"""

import ctypes
import ctypes.wintypes
import cv2
import logging
import math
import os
import queue
import random
import struct
import sys
import threading
import time

import mss
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ── Self-elevate to Administrator ─────────────────────────────────────────────
def _is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

if not _is_admin():
    script = os.path.abspath(sys.argv[0])
    params = " ".join(f'"{a}"' for a in sys.argv[1:])
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, f'"{script}" {params}'.strip(), None, 1
    )
    if ret <= 32:
        print(f"ERROR: UAC elevation failed (code {ret}).")
    sys.exit(0)


# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

CAPTURE_WIDTH = 600
CAPTURE_HEIGHT = 416
TARGET_FPS = 60
CONF_THRESHOLD = 0.60

VK_NUMPAD2 = 0x62
VK_NUMPAD3 = 0x63
VK_NUMPAD5 = 0x65
VK_NUMPAD6 = 0x66
VK_RBUTTON = 0x02

CAPTURE_FPS = 10
CAPTURE_DIR = BASE_DIR / "captures"
HARVEST_DIR = BASE_DIR / "harvest"
# Adaptive harvest: rate scales with detection frequency
HARVEST_MIN  = 0.02   # floor when detections are very frequent (15+/sec)
HARVEST_MAX  = 0.80   # ceiling when detections are rare (<1/sec)
HARVEST_WINDOW = 3.0  # seconds of history for rolling detection rate
JPEG_QUALITY = 95

# ── Streaming tick audio engine ───────────────────────────────────────────────
#
# Keeps a persistent PyAudio output stream running at all times.  The callback
# pulls from a ring buffer; tick() just drops samples in.  Zero per-call
# overhead — works at any FPS.

import pyaudio
import array as _array

_AUDIO_RATE   = 22050
_TICK_FREQ    = 880     # Hz — detection tick
_SAVE_FREQ    = 1320    # Hz — image-saved ping
_TICK_DUR_MS  = 18
_TICK_VOLUME  = 0.25

def _make_tick_samples(freq=_TICK_FREQ, duration_ms=_TICK_DUR_MS,
                       volume=_TICK_VOLUME, rate=_AUDIO_RATE):
    """Generate tick as a list of float samples (−1..1)."""
    n = int(rate * duration_ms / 1000)
    out = []
    for i in range(n):
        t = i / rate
        fade = 1.0 - (i / n)
        out.append(volume * fade * math.sin(2 * math.pi * freq * t))
    return out

_TICK_SAMPLES = _make_tick_samples(_TICK_FREQ)
_SAVE_SAMPLES = _make_tick_samples(_SAVE_FREQ)


class _AudioStream:
    """Persistent low-latency audio output via PyAudio callback.

    call tick() or tick(save=True) from any thread — it just appends
    samples to a lock-protected buffer that the callback drains.
    """
    def __init__(self, rate=_AUDIO_RATE, chunk=256):
        self._rate = rate
        self._chunk = chunk
        self._buf: list[float] = []
        self._lock = threading.Lock()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            output=True,
            frames_per_buffer=chunk,
            stream_callback=self._callback,
        )

    def tick(self, save=False):
        """Inject a tick into the audio buffer (replaces any pending tick)."""
        samples = _SAVE_SAMPLES if save else _TICK_SAMPLES
        with self._lock:
            # Replace buffer — latest tick wins, no queue buildup
            self._buf = list(samples)

    def _callback(self, _in_data, frame_count, _time_info, _status):
        with self._lock:
            if self._buf:
                chunk = self._buf[:frame_count]
                self._buf = self._buf[frame_count:]
            else:
                chunk = []
        # Pad with silence if needed
        out = chunk + [0.0] * (frame_count - len(chunk))
        data = struct.pack(f'<{frame_count}h',
                           *(int(max(-32767, min(32767, s * 32767))) for s in out))
        return (data, pyaudio.paContinue)

_audio = _AudioStream()

# Wrapper matching old _tick_player interface used in main loop
class _TickPlayer:
    def tick(self, wav=None):
        _audio.tick(save=False)
    def tick_save(self):
        _audio.tick(save=True)

_tick_player = _TickPlayer()

user32 = ctypes.windll.user32
GetAsyncKeyState = user32.GetAsyncKeyState
GetAsyncKeyState.argtypes = [ctypes.c_int]
GetAsyncKeyState.restype = ctypes.c_short

GetCursorPos = user32.GetCursorPos
GetCursorPos.argtypes = [ctypes.POINTER(ctypes.wintypes.POINT)]
GetCursorPos.restype = ctypes.c_bool


# ── SendInput with correct struct layout for mouse movement ──────────────────
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),   # ULONG_PTR — 8 bytes on x64
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]

class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]
    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("_union", _INPUT_UNION),
    ]

INPUT_MOUSE = 0

_SendInput = user32.SendInput
_SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
_SendInput.restype = ctypes.c_uint


def move_mouse_relative(dx: int, dy: int):
    """Send a relative mouse move via SendInput with correct struct sizing.

    SendInput is injected into the raw input stream on Windows,
    so games using Raw Input / DirectInput should receive it.
    """
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    inp.mi.time = 0
    inp.mi.dwExtraInfo = 0
    result = _SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    return result  # 1 = success, 0 = blocked (e.g. UIPI)


def _distance_frac(dist: float) -> float:
    """Return an easing fraction that is gentle near center and stronger far away."""
    return MOVE_FRAC_FAR - (MOVE_FRAC_FAR - MOVE_FRAC_CLOSE) * math.exp(-dist / MOVE_DECAY)


def _step_cap(dist: float) -> float:
    """Return the max allowed movement magnitude for the current target distance."""
    return MAX_STEP_CLOSE + (MAX_STEP_FAR - MAX_STEP_CLOSE) * (1.0 - math.exp(-dist / MAX_STEP_DECAY))


def _box_size_damp(box_w: float, box_h: float) -> float:
    """Reduce movement on large on-screen targets, where scoped overshoot shows up most."""
    box_diag = math.hypot(box_w, box_h)
    damp = 140.0 / max(140.0, box_diag)
    return max(BOX_SIZE_DAMP_MIN, min(1.0, damp))


def _compute_move(off_x: float, off_y: float, box_w: float, box_h: float, snap: bool) -> tuple[int, int]:
    """Convert target offset into an assertive but bounded relative mouse step."""
    dist = math.hypot(off_x, off_y)
    if dist <= 0.0:
        return 0, 0

    frac = SNAP_FRAC if snap else _distance_frac(dist)
    damp = _box_size_damp(box_w, box_h)
    move_x = off_x * frac * damp
    move_y = off_y * frac * damp

    cap = _step_cap(dist)
    move_dist = math.hypot(move_x, move_y)
    if move_dist > cap:
        scale = cap / move_dist
        move_x *= scale
        move_y *= scale

    return int(round(move_x)), int(round(move_y))


# ── Smoothing config ──────────────────────────────────────────────────────────
SMOOTH_ALPHA    = 0.80   # EMA blend: heavier filtering to kill bbox jitter
DEAD_ZONE       = 3      # Absorb ±3px noise once on-target
MOVE_FRAC_CLOSE = 0.35   # Gentle when near center (prevents overshoot)
MOVE_FRAC_FAR   = 0.90   # Aggressive when far (fast initial convergence)
MOVE_DECAY      = 60.0   # Distance at which frac transitions
SNAP_FRAC       = 0.55   # First-frame engagement kept conservative to avoid scoped overshoot
MAX_STEP_CLOSE  = 20.0   # px cap near center to avoid oscillation
MAX_STEP_FAR    = 115.0  # px cap when far away to stay assertive without jumping past target
MAX_STEP_DECAY  = 90.0   # Distance scale for step-cap ramp
BOX_SIZE_DAMP_MIN = 0.55 # Large on-screen targets get reduced movement

# ── Target locking ────────────────────────────────────────────────────────────
LOCK_RADIUS       = 120.0  # wider lock for fast combat movement
LOCK_GRACE_FRAMES = 8      # hold longer through detection flicker


# ── Transparent Overlay (capture-proof) ───────────────────────────────────────
#
# Draws a teal glow border 1-3 px OUTSIDE the capture region and a small +
# crosshair on each detection center.  Uses SetWindowDisplayAffinity with
# WDA_EXCLUDEFROMCAPTURE so neither the border nor the crosshairs appear
# in mss screen grabs (DXGI desktop duplication).
#

_gdi32   = ctypes.windll.gdi32
_kernel32 = ctypes.windll.kernel32

# Win32 constants
_WS_POPUP          = 0x80000000
_WS_VISIBLE        = 0x10000000
_WS_EX_LAYERED     = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW  = 0x00000080
_WS_EX_TOPMOST     = 0x00000008
_LWA_COLORKEY      = 1
_WDA_EXCLUDE       = 0x00000011
_WM_PAINT          = 0x000F
_WM_USER           = 0x0400
_WM_REPAINT        = _WM_USER + 1
_PS_SOLID          = 0
_NULL_BRUSH        = 5
_COLOR_KEY         = 0x00000000   # black = transparent

def _rgb(r, g, b):
    return r | (g << 8) | (b << 16)

_TEAL_DIM    = _rgb(0, 120, 130)    # outer / inner glow ring
_TEAL_BRIGHT = _rgb(0, 240, 220)    # core border
_CROSS_COLOR = _rgb(0, 255, 220)    # crosshair on detections

# Border color pairs: (dim, bright) for each state
_GREEN_DIM    = _rgb(0, 130, 40)
_GREEN_BRIGHT = _rgb(0, 240, 80)
_RED_DIM      = _rgb(140, 20, 0)
_RED_BRIGHT   = _rgb(240, 40, 10)
_REC_DOT      = _rgb(255, 30, 30)    # red recording indicator dot
_HARVEST_DOT  = _rgb(255, 180, 0)    # amber harvest indicator dot

_WNDPROC_T = ctypes.WINFUNCTYPE(
    ctypes.c_longlong,   # LRESULT
    ctypes.c_void_p,     # HWND
    ctypes.c_uint,       # msg
    ctypes.c_ulonglong,  # WPARAM
    ctypes.c_longlong,   # LPARAM
)

class _WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize",        ctypes.c_uint),  ("style",        ctypes.c_uint),
        ("lpfnWndProc",   _WNDPROC_T),     ("cbClsExtra",   ctypes.c_int),
        ("cbWndExtra",    ctypes.c_int),    ("hInstance",    ctypes.c_void_p),
        ("hIcon",         ctypes.c_void_p), ("hCursor",      ctypes.c_void_p),
        ("hbrBackground", ctypes.c_void_p), ("lpszMenuName", ctypes.c_wchar_p),
        ("lpszClassName", ctypes.c_wchar_p),("hIconSm",      ctypes.c_void_p),
    ]

class _PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", ctypes.c_void_p), ("fErase", ctypes.c_int),
        ("rcPaint", ctypes.wintypes.RECT),
        ("fRestore", ctypes.c_int), ("fIncUpdate", ctypes.c_int),
        ("rgbReserved", ctypes.c_byte * 32),
    ]

class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
        ("wParam", ctypes.c_ulonglong), ("lParam", ctypes.c_longlong),
        ("time", ctypes.c_ulong), ("pt", ctypes.wintypes.POINT),
    ]

# 64-bit safe function signatures
_kernel32.GetModuleHandleW.restype = ctypes.c_void_p
user32.RegisterClassExW.argtypes = [ctypes.c_void_p]
user32.CreateWindowExW.restype  = ctypes.c_void_p
user32.CreateWindowExW.argtypes = [
    ctypes.c_ulong, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
user32.SetLayeredWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ubyte, ctypes.c_ulong]
user32.SetWindowDisplayAffinity.argtypes   = [ctypes.c_void_p, ctypes.c_ulong]
user32.DefWindowProcW.restype  = ctypes.c_longlong
user32.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_longlong]
user32.PostMessageW.argtypes   = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_longlong]
user32.InvalidateRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
user32.BeginPaint.restype  = ctypes.c_void_p
user32.BeginPaint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.EndPaint.argtypes   = [ctypes.c_void_p, ctypes.c_void_p]
user32.GetMessageW.argtypes      = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
user32.TranslateMessage.argtypes = [ctypes.c_void_p]
user32.DispatchMessageW.argtypes = [ctypes.c_void_p]
user32.SetWindowPos.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_gdi32.CreatePen.restype       = ctypes.c_void_p
_gdi32.CreatePen.argtypes      = [ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
_gdi32.SelectObject.restype    = ctypes.c_void_p
_gdi32.SelectObject.argtypes   = [ctypes.c_void_p, ctypes.c_void_p]
_gdi32.DeleteObject.argtypes   = [ctypes.c_void_p]
_gdi32.GetStockObject.restype  = ctypes.c_void_p
_gdi32.GetStockObject.argtypes = [ctypes.c_int]
_gdi32.Rectangle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_gdi32.MoveToEx.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_gdi32.LineTo.argtypes     = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_gdi32.CreateSolidBrush.restype  = ctypes.c_void_p
_gdi32.CreateSolidBrush.argtypes = [ctypes.c_ulong]
_gdi32.Ellipse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_gdi32.SetBkMode.argtypes      = [ctypes.c_void_p, ctypes.c_int]
_gdi32.SetTextColor.restype     = ctypes.c_ulong
_gdi32.SetTextColor.argtypes    = [ctypes.c_void_p, ctypes.c_ulong]
_gdi32.TextOutW.argtypes        = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int]
_gdi32.CreateFontW.restype      = ctypes.c_void_p
_gdi32.CreateFontW.argtypes     = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
    ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
    ctypes.c_wchar_p]


class Overlay:
    """Capture-proof overlay: green border + status labels + detection crosshairs."""
    PAD = 3       # px of border outside the capture region
    TEXT_H = 16   # px height for status text above border

    def __init__(self, region_left, region_top, capture_w, capture_h):
        self._cap_w = capture_w
        self._cap_h = capture_h
        self._win_w = capture_w  + 2 * self.PAD
        self._win_h = capture_h + 2 * self.PAD + self.TEXT_H
        self._win_x = region_left - self.PAD
        self._win_y = region_top  - self.PAD - self.TEXT_H
        self._dets  = []          # [(cx, cy, conf), ...] in capture-pixel coords
        self._armed = False
        self._capture = False
        self._harvest = False
        self._tick = False
        self._lock  = threading.Lock()
        self._hwnd  = None
        self._ready = threading.Event()
        self._wndproc_ref = None  # prevent GC of callback
        self._frame_count = 0
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(5)

    def update(self, detections, armed=False, capture_on=False, harvest_on=False, tick_on=False):
        """Push new detection centers and trigger repaint."""
        with self._lock:
            self._dets = [(d[0], d[1], d[4]) for d in detections]
            self._armed = armed
            self._capture = capture_on
            self._harvest = harvest_on
            self._tick = tick_on
        if self._hwnd:
            user32.PostMessageW(self._hwnd, _WM_REPAINT, 0, 0)
            self._frame_count += 1
            if self._frame_count % 120 == 0:
                user32.SetWindowPos(
                    self._hwnd, ctypes.c_void_p(-1), 0, 0, 0, 0,
                    0x0002 | 0x0001 | 0x0010)   # NOMOVE|NOSIZE|NOACTIVATE

    # ── Window thread ─────────────────────────────────────────────────────

    def _run(self):
        hmod = _kernel32.GetModuleHandleW(None)
        bg   = _gdi32.CreateSolidBrush(_COLOR_KEY)
        this = self

        def wndproc(hwnd, msg, wp, lp):
            if msg == _WM_PAINT:
                this._paint(hwnd)
                return 0
            if msg == _WM_REPAINT:
                user32.InvalidateRect(hwnd, None, 1)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wp, lp)

        self._wndproc_ref = _WNDPROC_T(wndproc)

        wc = _WNDCLASSEXW()
        wc.cbSize        = ctypes.sizeof(_WNDCLASSEXW)
        wc.lpfnWndProc   = self._wndproc_ref
        wc.hInstance     = hmod
        wc.hbrBackground = bg
        wc.lpszClassName = "YOLODetectOverlay"
        user32.RegisterClassExW(ctypes.byref(wc))

        self._hwnd = user32.CreateWindowExW(
            _WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_TOOLWINDOW | _WS_EX_TOPMOST,
            "YOLODetectOverlay", "",
            _WS_POPUP | _WS_VISIBLE,
            self._win_x, self._win_y, self._win_w, self._win_h,
            None, None, hmod, None)

        user32.SetLayeredWindowAttributes(self._hwnd, _COLOR_KEY, 0, _LWA_COLORKEY)
        user32.SetWindowDisplayAffinity(self._hwnd, _WDA_EXCLUDE)
        self._ready.set()

        msg = _MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    # ── Paint handler ─────────────────────────────────────────────────────

    def _paint(self, hwnd):
        ps  = _PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        if not hdc:
            return

        w, h, pad = self._win_w, self._win_h, self.PAD
        yoff = self.TEXT_H
        null_brush = _gdi32.GetStockObject(_NULL_BRUSH)

        with self._lock:
            is_armed = self._armed
            is_capture = self._capture
            is_harvest = self._harvest
            is_tick = self._tick
            dets = list(self._dets)

        # Always-green border (shifted below text strip)
        colors = [_GREEN_DIM, _GREEN_BRIGHT, _GREEN_DIM]
        for i, color in enumerate(colors):
            pen = _gdi32.CreatePen(_PS_SOLID, 1, color)
            op  = _gdi32.SelectObject(hdc, pen)
            ob  = _gdi32.SelectObject(hdc, null_brush)
            _gdi32.Rectangle(hdc, i, yoff + i, w - i, h - i)
            _gdi32.SelectObject(hdc, op)
            _gdi32.SelectObject(hdc, ob)
            _gdi32.DeleteObject(pen)

        # Status labels: "2:TICK  3:HARVEST  5:ARMED  6:CAPTURE"
        # (drawn on transparent background — window bg is color-key black)
        _gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
        font = _gdi32.CreateFontW(
            13, 0, 0, 0, 700, 0, 0, 0, 1, 0, 0, 3, 0, "Consolas")
        old_font = _gdi32.SelectObject(hdc, font)
        labels = [
            ("2:TICK", is_tick),
            ("3:HARVEST", is_harvest),
            ("5:ARMED", is_armed),
            ("6:CAPTURE", is_capture),
        ]
        tx = pad + 2
        for text, is_on in labels:
            _gdi32.SetTextColor(hdc, _GREEN_BRIGHT if is_on else _RED_BRIGHT)
            _gdi32.TextOutW(hdc, tx, 1, text, len(text))
            tx += (len(text) + 2) * 7
        _gdi32.SelectObject(hdc, old_font)
        _gdi32.DeleteObject(font)

        # Crosshairs (+) with bullseye ring on detection centers
        if dets:
            arm = 12   # crosshair arm length (px from center)
            gap = 3    # gap around center for readability
            ring_r = 8 # bullseye ring radius
            for cx, cy, _conf in dets:
                wx = int(round(cx)) + pad
                wy = int(round(cy)) + pad + yoff
                # Black outline pass (2px pen, drawn first)
                pen_bg = _gdi32.CreatePen(_PS_SOLID, 3, _rgb(0, 0, 0))
                op = _gdi32.SelectObject(hdc, pen_bg)
                ob = _gdi32.SelectObject(hdc, null_brush)
                _gdi32.MoveToEx(hdc, wx - arm, wy, None)
                _gdi32.LineTo(hdc, wx - gap, wy)
                _gdi32.MoveToEx(hdc, wx + gap + 1, wy, None)
                _gdi32.LineTo(hdc, wx + arm + 1, wy)
                _gdi32.MoveToEx(hdc, wx, wy - arm, None)
                _gdi32.LineTo(hdc, wx, wy - gap)
                _gdi32.MoveToEx(hdc, wx, wy + gap + 1, None)
                _gdi32.LineTo(hdc, wx, wy + arm + 1)
                _gdi32.Ellipse(hdc, wx - ring_r, wy - ring_r, wx + ring_r + 1, wy + ring_r + 1)
                _gdi32.SelectObject(hdc, op)
                _gdi32.SelectObject(hdc, ob)
                _gdi32.DeleteObject(pen_bg)
                # Bright foreground pass (1px pen, drawn on top)
                pen_fg = _gdi32.CreatePen(_PS_SOLID, 1, _CROSS_COLOR)
                op = _gdi32.SelectObject(hdc, pen_fg)
                ob = _gdi32.SelectObject(hdc, null_brush)
                _gdi32.MoveToEx(hdc, wx - arm, wy, None)
                _gdi32.LineTo(hdc, wx - gap, wy)
                _gdi32.MoveToEx(hdc, wx + gap + 1, wy, None)
                _gdi32.LineTo(hdc, wx + arm + 1, wy)
                _gdi32.MoveToEx(hdc, wx, wy - arm, None)
                _gdi32.LineTo(hdc, wx, wy - gap)
                _gdi32.MoveToEx(hdc, wx, wy + gap + 1, None)
                _gdi32.LineTo(hdc, wx, wy + arm + 1)
                _gdi32.Ellipse(hdc, wx - ring_r, wy - ring_r, wx + ring_r + 1, wy + ring_r + 1)
                _gdi32.SelectObject(hdc, op)
                _gdi32.SelectObject(hdc, ob)
                _gdi32.DeleteObject(pen_fg)

        user32.EndPaint(hwnd, ctypes.byref(ps))


# ── Background frame writer ───────────────────────────────────────────────────
_write_queue: queue.Queue = queue.Queue(maxsize=200)
_writer_stop = threading.Event()


def _writer_thread():
    """Drain the write queue and save frames to disk as JPEG."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    while not _writer_stop.is_set():
        try:
            filepath, frame = _write_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        except Exception as exc:
            print(f"  !! Write error: {exc}")
        _write_queue.task_done()


# ── Detection loop ────────────────────────────────────────────────────────────
def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    # Compute center capture region
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    region_left = (screen_w - CAPTURE_WIDTH) // 2
    region_top = (screen_h - CAPTURE_HEIGHT) // 2

    region = {
        "left": region_left, "top": region_top,
        "width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT,
    }

    # Transparent overlay (teal border + detection crosshairs, excluded from capture)
    overlay = Overlay(region_left, region_top, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    print(f"YOLO Detection Agent")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Screen : {screen_w}x{screen_h}")
    print(f"  Region : {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} center @ ({region_left},{region_top})")
    print(f"  FPS    : {TARGET_FPS}")
    print(f"  Conf   : {CONF_THRESHOLD}")
    print(f"  Numpad2: toggle tick sound")
    print(f"  Numpad3: toggle harvest (adaptive {HARVEST_MIN:.0%}–{HARVEST_MAX:.0%})")
    print(f"  Numpad5: toggle armed")
    print(f"  Numpad6: toggle capture ({CAPTURE_FPS} FPS while RMB held)")
    print(f"  RMB    : move cursor to target")
    print(f"  Saves  : {CAPTURE_DIR}")
    print(f"  Harvest: {HARVEST_DIR}")
    print()

    model = YOLO(str(MODEL_PATH))

    # Warm up
    dummy = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)
    model(dummy, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    print("Model warmed up — running\n")

    armed = False
    capture_on = False
    harvest_on = False
    tick_on = False
    det_timestamps: list[float] = []   # rolling window for adaptive harvest rate
    prev_numpad2 = False
    prev_numpad3 = False
    prev_numpad5 = False
    prev_numpad6 = False
    interval = 1.0 / TARGET_FPS
    capture_interval = 1.0 / CAPTURE_FPS
    last_capture_t = 0.0

    # EMA-smoothed target position (capture-region coords)
    smooth_x: float | None = None
    smooth_y: float | None = None
    was_tracking = False
    # Target locking state
    lock_cx: float | None = None
    lock_cy: float | None = None
    lock_grace: int = 0

    # Start background writer
    writer = threading.Thread(target=_writer_thread, daemon=True)
    writer.start()

    with mss.mss() as sct:
        while True:
            t0 = time.perf_counter()

            # ── Poll Numpad 2 toggle (tick sound) ─────────────────────
            num2 = bool(GetAsyncKeyState(VK_NUMPAD2) & 0x8000)
            if num2 and not prev_numpad2:
                tick_on = not tick_on
                print(f"[TICK {'ON' if tick_on else 'OFF'}]")
            prev_numpad2 = num2

            # ── Poll Numpad 3 toggle (harvest) ─────────────────────────
            num3 = bool(GetAsyncKeyState(VK_NUMPAD3) & 0x8000)
            if num3 and not prev_numpad3:
                harvest_on = not harvest_on
                print(f"[HARVEST {'ON' if harvest_on else 'OFF'}]")
            prev_numpad3 = num3

            # ── Poll Numpad 5 toggle ──────────────────────────────────────
            num5 = bool(GetAsyncKeyState(VK_NUMPAD5) & 0x8000)
            if num5 and not prev_numpad5:
                armed = not armed
                print(f"[{'ARMED' if armed else 'DISARMED'}]")
            prev_numpad5 = num5

            # ── Poll Numpad 6 toggle ──────────────────────────────────────
            num6 = bool(GetAsyncKeyState(VK_NUMPAD6) & 0x8000)
            if num6 and not prev_numpad6:
                capture_on = not capture_on
                print(f"[CAPTURE {'ON' if capture_on else 'OFF'}]")
            prev_numpad6 = num6

            # ── Grab screen ───────────────────────────────────────────────
            frame = np.array(sct.grab(region), dtype=np.uint8)[:, :, :3]

            # ── Inference ─────────────────────────────────────────────────
            results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

            # ── Extract detections ────────────────────────────────────────
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    detections.append((cx, cy, w, h, conf))

            # ── Update overlay ────────────────────────────────────────────
            overlay.update(detections, armed=armed, capture_on=capture_on,
                           harvest_on=harvest_on, tick_on=tick_on)

            # ── Console log ───────────────────────────────────────────────
            if detections:
                # Tick sound
                if tick_on:
                    _tick_player.tick()
                parts = []
                for cx, cy, w, h, conf in detections:
                    parts.append(f"{conf:.0%} @({cx:.0f},{cy:.0f}) {w:.0f}x{h:.0f}")
                tag = ">>>" if armed else "   "
                print(f"{tag} {len(detections)} det: {', '.join(parts)}")

            # ── Move mouse when armed + RMB held ─────────────────────────
            if armed and (GetAsyncKeyState(VK_RBUTTON) & 0x8000):
                center_x = CAPTURE_WIDTH / 2
                center_y = CAPTURE_HEIGHT / 2
                target = None

                if detections:
                    if lock_cx is not None:
                        # Find detection closest to locked position
                        closest = min(detections, key=lambda d: math.hypot(d[0] - lock_cx, d[1] - lock_cy))
                        if math.hypot(closest[0] - lock_cx, closest[1] - lock_cy) < LOCK_RADIUS:
                            target = closest
                            lock_grace = 0
                        else:
                            # Locked target moved too far — grace period
                            lock_grace += 1
                            if lock_grace > LOCK_GRACE_FRAMES:
                                target = max(detections, key=lambda d: d[4])
                                lock_grace = 0
                    if target is None and lock_grace == 0:
                        # No lock yet or grace expired — acquire best
                        target = max(detections, key=lambda d: d[4])
                else:
                    # No detections this frame — hold lock during grace
                    if lock_cx is not None:
                        lock_grace += 1
                        if lock_grace > LOCK_GRACE_FRAMES:
                            lock_cx = None
                            lock_cy = None
                            lock_grace = 0

                if target is not None:
                    raw_cx, raw_cy, raw_w, raw_h = target[0], target[1], target[2], target[3]
                    lock_cx = raw_cx
                    lock_cy = raw_cy

                    if not was_tracking or smooth_x is None:
                        # SNAP: bounded first frame to avoid long-range scoped overshoot
                        smooth_x = raw_cx
                        smooth_y = raw_cy
                        off_x = raw_cx - center_x
                        off_y = raw_cy - center_y
                        dx, dy = _compute_move(off_x, off_y, raw_w, raw_h, snap=True)
                    else:
                        # TRACK: EMA smoothed + bounded easing
                        smooth_x = SMOOTH_ALPHA * raw_cx + (1 - SMOOTH_ALPHA) * smooth_x
                        smooth_y = SMOOTH_ALPHA * raw_cy + (1 - SMOOTH_ALPHA) * smooth_y
                        off_x = smooth_x - center_x
                        off_y = smooth_y - center_y
                        dx, dy = _compute_move(off_x, off_y, raw_w, raw_h, snap=False)

                    if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
                        move_mouse_relative(dx, dy)

                was_tracking = True
            else:
                # Reset when not actively tracking
                was_tracking = False
                smooth_x = None
                smooth_y = None
                lock_cx = None
                lock_cy = None
                lock_grace = 0

            # ── Capture frame while RMB held ──────────────────────────────
            if capture_on and (GetAsyncKeyState(VK_RBUTTON) & 0x8000):
                now = time.perf_counter()
                if now - last_capture_t >= capture_interval:
                    last_capture_t = now
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{ts}_{int(now * 1000) % 100000:05d}.jpg"
                    fpath = CAPTURE_DIR / fname
                    if not _write_queue.full():
                        _write_queue.put_nowait((fpath, frame.copy()))

            # ── Harvest: randomly save frames with detections ────────────
            if harvest_on and detections:
                now_h = time.perf_counter()
                det_timestamps.append(now_h)
                # Prune timestamps older than the window
                cutoff = now_h - HARVEST_WINDOW
                while det_timestamps and det_timestamps[0] < cutoff:
                    det_timestamps.pop(0)
                # Detection rate: detections per second over the window
                det_rate = len(det_timestamps) / HARVEST_WINDOW
                # Map: 0 det/s → HARVEST_MAX, 15+ det/s → HARVEST_MIN (linear)
                t_rate = min(1.0, det_rate / 15.0)
                harvest_chance = HARVEST_MAX - t_rate * (HARVEST_MAX - HARVEST_MIN)
                if random.random() < harvest_chance:
                    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    ms = int(time.perf_counter() * 1000) % 100000
                    fname = f"{ts}_{ms:05d}.jpg"
                    fpath = HARVEST_DIR / fname
                    if not _write_queue.full():
                        _write_queue.put_nowait((fpath, frame.copy()))
                        if tick_on:
                            _tick_player.tick_save()

            # ── Frame pacing ──────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0.001:
                time.sleep(remaining - 0.001)
            while time.perf_counter() - t0 < interval:
                pass


if __name__ == "__main__":
    # Log crashes to file so admin-elevated console doesn't vanish silently
    _log_path = BASE_DIR / "detect.log"
    logging.basicConfig(
        filename=str(_log_path),
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        main()
    except Exception:
        logging.exception("detect.py crashed")
        raise
