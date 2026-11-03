"""Native composition effects for undecorated desktop windows."""
import ctypes
import sys
from ctypes import wintypes


class GUID(ctypes.Structure):
    _fields_ = [('data1', wintypes.DWORD), ('data2', wintypes.WORD),
                ('data3', wintypes.WORD), ('data4', ctypes.c_ubyte * 8)]


def set_taskbar_tab(hwnd, visible):
    """Add or remove a live window's button without remapping its Tk wrapper."""
    if sys.platform != 'win32':
        return
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = ctypes.c_long
    initialized = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
    if initialized < 0:
        raise OSError(f'COM initialization failed: {initialized & 0xffffffff:#x}')
    try:
        ole32.CLSIDFromString.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(GUID)]
        ole32.CLSIDFromString.restype = ctypes.c_long
        ole32.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), ctypes.c_void_p,
                                           wintypes.DWORD, ctypes.POINTER(GUID),
                                           ctypes.POINTER(ctypes.c_void_p)]
        ole32.CoCreateInstance.restype = ctypes.c_long
        clsid, iid = GUID(), GUID()
        for value, target in (('{56FDF344-FD6D-11D0-958A-006097C9A090}', clsid),
                              ('{56FDF342-FD6D-11D0-958A-006097C9A090}', iid)):
            result = ole32.CLSIDFromString(value, ctypes.byref(target))
            if result < 0:
                raise OSError(f'Taskbar GUID failed: {result & 0xffffffff:#x}')
        interface = ctypes.c_void_p()
        result = ole32.CoCreateInstance(ctypes.byref(clsid), None, 1,
                                        ctypes.byref(iid), ctypes.byref(interface))
        if result < 0:
            raise OSError(f'Taskbar interface failed: {result & 0xffffffff:#x}')
        try:
            methods = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            initialize = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(methods[3])
            update = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.HWND)(
                methods[4 if visible else 5])  # ITaskbarList::AddTab / DeleteTab
            result = initialize(interface)
            if result >= 0:
                result = update(interface, hwnd)
            if result < 0:
                raise OSError(f'Taskbar update failed: {result & 0xffffffff:#x}')
        finally:
            release = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(methods[2])
            release(interface)
    finally:
        ole32.CoUninitialize()


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
