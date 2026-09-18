"""Native composition effects for undecorated desktop windows."""
import ctypes
import sys
from ctypes import wintypes


class Margins(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ('left', 'right', 'top', 'bottom')]


def apply_shadow(window):
    """Let DWM draw the shadow; keep Tk's content and hit testing unchanged."""
    if sys.platform != 'win32':
        return False
    api = ctypes.windll.user32
    api.GetParent.argtypes = [wintypes.HWND]
    api.GetParent.restype = wintypes.HWND
    hwnd = api.GetParent(window.winfo_id())
    dwm = ctypes.windll.dwmapi
    dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD,
                                        ctypes.c_void_p, wintypes.DWORD]
    dwm.DwmExtendFrameIntoClientArea.argtypes = [wintypes.HWND, ctypes.POINTER(Margins)]
    policy = ctypes.c_int(2)  # DWMNCRP_ENABLED
    result = dwm.DwmSetWindowAttribute(hwnd, 2, ctypes.byref(policy), ctypes.sizeof(policy))
    margins = Margins(1, 1, 1, 1)
    return result == 0 and dwm.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins)) == 0
