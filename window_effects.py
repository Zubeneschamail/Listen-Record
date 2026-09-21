"""Native composition effects for undecorated desktop windows."""
import ctypes
import sys
from ctypes import wintypes


class Margins(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ('left', 'right', 'top', 'bottom')]


def work_area(window):
    """Return the nearest monitor's usable rectangle, excluding its taskbar."""
    class MonitorInfo(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('monitor', wintypes.RECT),
                    ('work', wintypes.RECT), ('flags', wintypes.DWORD)]

    api = ctypes.windll.user32
    api.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    api.MonitorFromWindow.restype = wintypes.HANDLE
    api.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
    api.GetMonitorInfoW.restype = wintypes.BOOL
    info = MonitorInfo()
    info.size = ctypes.sizeof(info)
    if not api.GetMonitorInfoW(api.MonitorFromWindow(window.winfo_id(), 2), ctypes.byref(info)):
        raise ctypes.WinError()
    return info.work.left, info.work.top, info.work.right, info.work.bottom


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
