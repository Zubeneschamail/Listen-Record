"""One explicit screen snapshot per request; no polling of screen contents."""
import ctypes
from ctypes import wintypes
import io
import threading
import time

from PIL import ImageGrab
from clipboard_watch import ClipboardItem


def foreground_monitor_bounds():
    class MonitorInfo(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('monitor', wintypes.RECT),
                    ('work', wintypes.RECT), ('flags', wintypes.DWORD)]
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.GetForegroundWindow.restype = wintypes.HWND
    api.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    api.MonitorFromWindow.restype = wintypes.HANDLE
    api.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
    api.GetMonitorInfoW.restype = wintypes.BOOL
    monitor = api.MonitorFromWindow(api.GetForegroundWindow(), 2)
    info = MonitorInfo()
    info.size = ctypes.sizeof(info)
    if not monitor or not api.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    rect = info.monitor
    if rect.right <= rect.left or rect.bottom <= rect.top:
        raise OSError('无法确定当前屏幕范围')
    return rect.left, rect.top, rect.right, rect.bottom


def capture_screen():
    # Use physical monitor coordinates without changing Tk's process-wide DPI mode.
    api = ctypes.WinDLL('user32', use_last_error=True)
    set_dpi = getattr(api, 'SetThreadDpiAwarenessContext', None)
    previous = None
    if set_dpi is not None:
        set_dpi.argtypes = [ctypes.c_void_p]
        set_dpi.restype = ctypes.c_void_p
        previous = set_dpi(ctypes.c_void_p(-4))
    try:
        bounds = foreground_monitor_bounds()
        image = ImageGrab.grab(bbox=bounds, all_screens=True)
        try:
            image.thumbnail((4096, 4096))
            image = image.convert('RGB')
            output = io.BytesIO()
            image.save(output, format='PNG')
            data = output.getvalue()
            # High-entropy desktops may not fit the PNG upload budget.
            for side in (4096, 3072, 2560):
                if len(data) <= 8*1024*1024:
                    break
                image.thumbnail((side, side))
                output = io.BytesIO()
                image.save(output, format='JPEG', quality=85)
                data = output.getvalue()
            if len(data) > 8*1024*1024:
                raise ValueError('屏幕图片超过 8 MB，请降低屏幕分辨率后重试')
            return ClipboardItem(
                f'【屏幕截图 {image.width}×{image.height}】请回答当前屏幕中的题目或问题，'
                '先给答案，再给必要依据；不要只复述画面。若有多道题，按画面顺序回答；'
                '关键条件看不清时指出具体缺口。没有题目时简要概括当前屏幕内容。', data)
        finally:
            image.close()
    finally:
        if previous and set_dpi is not None:
            set_dpi(previous)


class ScreenCapture:
    def __init__(self, events, capture=None, clock=time.monotonic):
        self.events, self.capture, self.clock = events, capture or capture_screen, clock
        self.lock = threading.Lock()
        self.generation = 0
        self.busy = False
        self.last_request = float('-inf')

    def cancel(self):
        with self.lock:
            self.generation += 1

    def request(self):
        with self.lock:
            now = self.clock()
            if self.busy or now-self.last_request < .75:
                return False
            generation = self.generation
            self.busy = True
            self.last_request = now
        def work():
            item, error = None, ''
            try:
                item = self.capture()
            except Exception as exc:
                error = '截屏失败：'+str(exc)
            finally:
                with self.lock:
                    self.busy = False
                    if generation == self.generation:
                        self.events.put(('screen_capture', (generation, item, error)))
        threading.Thread(target=work, name='screen-capture', daemon=True).start()
        return True
