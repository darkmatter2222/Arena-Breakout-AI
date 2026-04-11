"""
Diagnostic: logs ALL detected input events to input_diag.log.
Run this as admin, switch to your game, press keys and click.
After ~15 seconds, check input_diag.log to see what was detected.

Tests three methods simultaneously:
  1. WH_KEYBOARD_LL hook
  2. Raw Input (RIDEV_INPUTSINK) for mouse
  3. GetAsyncKeyState polling for both keyboard and mouse
"""
import ctypes
import ctypes.wintypes
import threading
import time
import sys
import os

# ── Self-elevate ──
def _is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

if not _is_admin():
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable,
        f'"{os.path.abspath(__file__)}"', None, 1,
    )
    sys.exit(0)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_diag.log")
log_lock = threading.Lock()

def log(msg):
    line = f"[{time.perf_counter():12.4f}] {msg}"
    print(line)
    with log_lock:
        with open(LOG, "a") as f:
            f.write(line + "\n")

# Wipe old log
with open(LOG, "w") as f:
    f.write(f"=== Input Diagnostic Started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    f.write(f"Running as admin: {_is_admin()}\n")
    f.write(f"PID: {os.getpid()}\n\n")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

LRESULT = ctypes.wintypes.LPARAM
WPARAM = ctypes.wintypes.WPARAM
LPARAM = ctypes.wintypes.LPARAM
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, WPARAM, LPARAM)
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.wintypes.HWND, ctypes.c_uint, WPARAM, LPARAM)

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.wintypes.DWORD),
        ("scanCode", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint), ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int), ("hInstance", ctypes.wintypes.HINSTANCE),
        ("hIcon", ctypes.wintypes.HICON), ("hCursor", ctypes.c_void_p),
        ("hbrBackground", ctypes.c_void_p), ("lpszMenuName", ctypes.c_wchar_p),
        ("lpszClassName", ctypes.c_wchar_p), ("hIconSm", ctypes.wintypes.HICON),
    ]

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.c_ushort), ("usUsage", ctypes.c_ushort),
        ("dwFlags", ctypes.wintypes.DWORD), ("hwndTarget", ctypes.wintypes.HWND),
    ]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", ctypes.wintypes.DWORD), ("dwSize", ctypes.wintypes.DWORD),
        ("hDevice", ctypes.wintypes.HANDLE), ("wParam", WPARAM),
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", ctypes.c_ushort), ("_pad", ctypes.c_ushort),
        ("usButtonFlags", ctypes.c_ushort), ("usButtonData", ctypes.c_ushort),
        ("ulRawButtons", ctypes.c_ulong), ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long), ("ulExtraInformation", ctypes.c_ulong),
    ]

class RAWINPUT_MOUSE(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]

# API signatures
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, ctypes.wintypes.HINSTANCE, ctypes.wintypes.DWORD]
user32.SetWindowsHookExW.restype = ctypes.c_void_p
user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, WPARAM, LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
user32.GetMessageW.argtypes = [ctypes.POINTER(ctypes.wintypes.MSG), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
user32.GetMessageW.restype = ctypes.wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(ctypes.wintypes.MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(ctypes.wintypes.MSG)]
user32.DefWindowProcW.argtypes = [ctypes.wintypes.HWND, ctypes.c_uint, WPARAM, LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = ctypes.wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    ctypes.wintypes.DWORD, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.wintypes.HWND, ctypes.c_void_p, ctypes.wintypes.HINSTANCE, ctypes.c_void_p,
]
user32.CreateWindowExW.restype = ctypes.wintypes.HWND
user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), ctypes.c_uint, ctypes.c_uint]
user32.RegisterRawInputDevices.restype = ctypes.wintypes.BOOL
user32.GetRawInputData.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint]
user32.GetRawInputData.restype = ctypes.c_uint
user32.PostMessageW.argtypes = [ctypes.wintypes.HWND, ctypes.c_uint, WPARAM, LPARAM]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
kernel32.GetModuleHandleW.restype = ctypes.wintypes.HINSTANCE
kernel32.GetCurrentThreadId.argtypes = []
kernel32.GetCurrentThreadId.restype = ctypes.wintypes.DWORD

# Constants
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100; WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104; WM_SYSKEYUP = 0x0105
WM_INPUT = 0x00FF; WM_DESTROY = 0x0002
RIDEV_INPUTSINK = 0x100; RID_INPUT = 0x10000003
VK_NUMPAD5 = 0x65; VK_LBUTTON = 0x01; VK_ESCAPE = 0x1B

kb_hook = None

# ── Method 1: Keyboard LL Hook ──
def kb_callback(nCode, wParam, lParam):
    if nCode >= 0 and lParam:
        kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        action = "DOWN" if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN) else "UP"
        log(f"[KB_HOOK]  vk=0x{kb.vkCode:02X}  {action}  scan=0x{kb.scanCode:02X}  flags=0x{kb.flags:04X}")
    return user32.CallNextHookEx(kb_hook, nCode, wParam, lParam)

kb_proc_ref = HOOKPROC(kb_callback)

# ── Method 2: Raw Input Mouse ──
hwnd_ri = None

def handle_raw_input(hRawInput):
    dwSize = ctypes.c_uint(0)
    user32.GetRawInputData(hRawInput, RID_INPUT, None, ctypes.byref(dwSize), ctypes.sizeof(RAWINPUTHEADER))
    if dwSize.value == 0:
        return
    buf = (ctypes.c_byte * dwSize.value)()
    ret = user32.GetRawInputData(hRawInput, RID_INPUT, buf, ctypes.byref(dwSize), ctypes.sizeof(RAWINPUTHEADER))
    if ret == ctypes.c_uint(-1).value:
        return
    hdr = ctypes.cast(buf, ctypes.POINTER(RAWINPUTHEADER)).contents
    if hdr.dwType != 0:
        return
    ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT_MOUSE)).contents
    flags = ri.mouse.usButtonFlags
    if flags:  # only log when buttons change, not pure movement
        log(f"[RAW_INPUT] buttonFlags=0x{flags:04X}  dx={ri.mouse.lLastX}  dy={ri.mouse.lLastY}")

def wnd_proc(hwnd, msg, wp, lp):
    if msg == WM_INPUT:
        handle_raw_input(lp)
        return 0
    if msg == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0
    return user32.DefWindowProcW(hwnd, msg, wp, lp)

wnd_proc_ref = WNDPROC(wnd_proc)

# ── Method 3: GetAsyncKeyState polling (separate thread) ──
poll_stop = threading.Event()

def poll_thread():
    """Polls GetAsyncKeyState every 50ms and logs state changes."""
    prev_lmb = False
    prev_num5 = False
    prev_esc = False
    while not poll_stop.is_set():
        lmb = bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        num5 = bool(user32.GetAsyncKeyState(VK_NUMPAD5) & 0x8000)
        esc = bool(user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)

        if lmb != prev_lmb:
            log(f"[POLL]     LMB={'DOWN' if lmb else 'UP'}")
            prev_lmb = lmb
        if num5 != prev_num5:
            log(f"[POLL]     NUMPAD5={'DOWN' if num5 else 'UP'}")
            prev_num5 = num5
        if esc != prev_esc:
            log(f"[POLL]     ESCAPE={'DOWN' if esc else 'UP'}")
            prev_esc = esc

        time.sleep(0.05)

# ── Setup and run ──
def run():
    global kb_hook, hwnd_ri

    hmod = kernel32.GetModuleHandleW(None)

    # Start keyboard hook + raw input on this thread
    kb_hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, kb_proc_ref, hmod, 0)
    log(f"Keyboard hook handle: {kb_hook}")

    # Create hidden window for raw input
    wc = WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
    wc.lpfnWndProc = wnd_proc_ref
    wc.hInstance = hmod
    wc.lpszClassName = "DiagRI"
    user32.RegisterClassExW(ctypes.byref(wc))

    HWND_MESSAGE = ctypes.wintypes.HWND(-3 & 0xFFFFFFFFFFFFFFFF)
    hwnd_ri = user32.CreateWindowExW(0, "DiagRI", "", 0, 0, 0, 0, 0, HWND_MESSAGE, None, hmod, None)
    log(f"Raw Input window handle: {hwnd_ri}")

    rid = RAWINPUTDEVICE()
    rid.usUsagePage = 0x01
    rid.usUsage = 0x02
    rid.dwFlags = RIDEV_INPUTSINK
    rid.hwndTarget = hwnd_ri
    ok = user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE))
    log(f"RegisterRawInputDevices: {ok}")

    # Start polling thread
    pt = threading.Thread(target=poll_thread, daemon=True)
    pt.start()

    log("")
    log("=" * 60)
    log("  DIAGNOSTIC RUNNING — switch to your game now!")
    log("  Press keys and click mouse. Watch this log.")
    log("  Press Ctrl+C here or close window to stop.")
    log("  Results will be in: input_diag.log")
    log("=" * 60)
    log("")

    # Message pump (needed for both LL hooks and Raw Input)
    msg = ctypes.wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except KeyboardInterrupt:
        pass

    poll_stop.set()
    if kb_hook:
        user32.UnhookWindowsHookEx(kb_hook)
    log("Diagnostic stopped.")

if __name__ == "__main__":
    run()
