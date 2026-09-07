"""A Windows global hotkey on its own message thread (no keyboard hook)."""
import ctypes
from ctypes import wintypes
import threading


class GlobalHotkey:
    def __init__(self, events, modifiers=0x0002, key=0x20):
        self.events = events
        self.modifiers = modifiers
        self.key = key
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, name="global-hotkey", daemon=True)
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=0.5)

    def run(self):
        registered = False
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
            user32.RegisterHotKey.restype = wintypes.BOOL
            user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.UnregisterHotKey.restype = wintypes.BOOL
            user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                            wintypes.UINT, wintypes.UINT, wintypes.UINT]
            user32.PeekMessageW.restype = wintypes.BOOL
            # NULL HWND targets this thread; MOD_NOREPEAT prevents held keys toggling repeatedly.
            registered = bool(user32.RegisterHotKey(None, 1, self.modifiers | 0x4000, self.key))
            if not registered:
                error = ctypes.get_last_error()
                self.events.put(("hotkey_status", (False, f"注册失败，可能已被占用（{error}）")))
                return
            self.events.put(("hotkey_status", (True, "")))
            message = wintypes.MSG()
            while not self.stopped.wait(0.02):
                while user32.PeekMessageW(ctypes.byref(message), None, 0x0312, 0x0312, 1):
                    if message.wParam == 1 and not self.stopped.is_set():
                        self.events.put(("hotkey", None))
        except Exception as exc:
            self.events.put(("hotkey_status", (False, str(exc))))
        finally:
            if registered:
                user32.UnregisterHotKey(None, 1)
