"""Side-button input for manual transcript boundaries; no audio work in hooks."""
import ctypes
from ctypes import wintypes
import threading

MOUSE_OPTIONS = {'关闭': 0, '侧键 1（后退）': 1, '侧键 2（前进）': 2}


class SideButtonHotkey:
    def __init__(self, events, active, button=1):
        self.events, self.active, self.button = events, active, button
        self.stopped = threading.Event()
        self.thread = None
        self.held = False

    def handle(self, message, button, injected=False):
        if injected or button != self.button:
            return False
        if message == 0x020B:  # WM_XBUTTONDOWN
            if self.held:
                return True
            if self.active.is_set() and not self.stopped.is_set():
                self.held = True
                self.events.put(('manual_cut', None))
                return True
        elif message == 0x020C and self.held:
            # Swallow the paired release even if recording stopped meanwhile.
            self.held = False
            return True
        return False

    def start(self):
        self.thread = threading.Thread(target=self.run, name='manual-cut-mouse', daemon=True)
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=.5)

    def run(self):
        hook = None
        try:
            user = ctypes.WinDLL('user32', use_last_error=True)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            callback_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int,
                                               wintypes.WPARAM, wintypes.LPARAM)
            class MouseData(ctypes.Structure):
                _fields_ = [('pt', wintypes.POINT), ('mouseData', wintypes.DWORD),
                            ('flags', wintypes.DWORD), ('time', wintypes.DWORD),
                            ('extra', ctypes.c_size_t)]
            user.SetWindowsHookExW.argtypes = (ctypes.c_int, callback_type, wintypes.HINSTANCE, wintypes.DWORD)
            user.SetWindowsHookExW.restype = wintypes.HANDLE
            user.CallNextHookEx.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
            user.CallNextHookEx.restype = ctypes.c_ssize_t
            user.UnhookWindowsHookEx.argtypes = (wintypes.HANDLE,)
            user.UnhookWindowsHookEx.restype = wintypes.BOOL
            kernel.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
            kernel.GetModuleHandleW.restype = wintypes.HMODULE
            user.PeekMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                         wintypes.UINT, wintypes.UINT, wintypes.UINT)
            user.PeekMessageW.restype = wintypes.BOOL
            user.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
            user.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)
            user.DispatchMessageW.restype = ctypes.c_ssize_t

            @callback_type
            def callback(code, message, address):
                if code >= 0 and message in (0x020B, 0x020C):
                    try:
                        value = ctypes.cast(address, ctypes.POINTER(MouseData)).contents
                        if self.handle(message, (value.mouseData >> 16) & 0xffff, bool(value.flags & 1)):
                            return 1
                    except Exception:
                        pass  # Never break other applications' mouse input.
                return user.CallNextHookEx(hook, code, message, address)

            hook = user.SetWindowsHookExW(14, callback, kernel.GetModuleHandleW(None), 0)
            if not hook:
                raise ctypes.WinError(ctypes.get_last_error())
            self.events.put(('manual_mouse_status', (True, '')))
            message = wintypes.MSG()
            while not self.stopped.wait(.01):
                while user.PeekMessageW(ctypes.byref(message), None, 0, 0, 1):
                    user.TranslateMessage(ctypes.byref(message))
                    user.DispatchMessageW(ctypes.byref(message))
        except Exception as exc:
            self.events.put(('manual_mouse_status', (False, str(exc))))
        finally:
            if hook:
                user.UnhookWindowsHookEx(hook)
