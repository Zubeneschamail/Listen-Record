"""Exclude this application's own Tk windows from supported screen capture."""
import ctypes
from ctypes import wintypes
from functools import lru_cache
import logging
import sys
import tkinter as tk


class CapturePrivacyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def display_api():
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    api.GetAncestor.restype = wintypes.HWND
    api.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    api.SetWindowDisplayAffinity.restype = wintypes.BOOL
    api.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    api.GetWindowDisplayAffinity.restype = wintypes.BOOL
    return api


def set_window_capture(root, path, enabled):
    if sys.platform != 'win32' or sys.getwindowsversion().build < 19041:
        if enabled:
            raise CapturePrivacyError('需要 Windows 10 2004 或更新版本。')
        return
    api = display_api()
    client = int(str(root.tk.call('winfo', 'id', path)), 0)
    # Tk's widget HWND is inside its native top-level wrapper.
    hwnd = api.GetAncestor(client, 2)  # GA_ROOT
    flag = 0x11 if enabled else 0  # WDA_EXCLUDEFROMCAPTURE / WDA_NONE
    if not hwnd or not api.SetWindowDisplayAffinity(hwnd, flag):
        raise CapturePrivacyError(f'Windows 未能设置窗口隐藏（错误 {ctypes.get_last_error()}）。')
    actual = wintypes.DWORD()
    if not api.GetWindowDisplayAffinity(hwnd, ctypes.byref(actual)) or actual.value != flag:
        raise CapturePrivacyError('Windows 未确认窗口隐藏状态，请重试。')


class CapturePrivacy:
    def __init__(self, root, enabled=False):
        self.root = root
        self.enabled = bool(enabled)
        self.status = tk.StringVar(master=root, value='已关闭')
        # Includes Tcl-created combobox popdowns as well as Python Toplevels.
        self.binding = root.bind_all('<Map>', self.on_map, add='+')
        self.refresh_status()

    def refresh_status(self):
        self.status.set('已开启 · 请在共享预览中确认' if self.enabled else '已关闭')

    def windows(self, path='.'):
        if str(self.root.tk.call('winfo', 'toplevel', path)) == path:
            yield path
        for child in self.root.tk.splitlist(self.root.tk.call('winfo', 'children', path)):
            yield from self.windows(str(child))

    def set_enabled(self, enabled):
        previous = self.enabled
        try:
            for path in self.windows():
                set_window_capture(self.root, path, enabled)
        except (CapturePrivacyError, tk.TclError) as exc:
            # Include the failing window: the native setter may have succeeded
            # before readback failed. Restore all current windows to the old mode.
            for path in self.windows():
                try:
                    set_window_capture(self.root, path, previous)
                except (CapturePrivacyError, tk.TclError):
                    logging.exception('Restoring window capture mode')
            self.status.set('设置未生效，请重试')
            raise CapturePrivacyError(str(exc)) from exc
        self.enabled = bool(enabled)
        self.refresh_status()

    def on_map(self, event):
        path = str(event.widget)
        try:
            if str(self.root.tk.call('winfo', 'toplevel', path)) != path:
                return
            set_window_capture(self.root, path, self.enabled)
        except (CapturePrivacyError, tk.TclError):
            logging.exception('Applying window capture mode')
            if self.enabled:
                self.status.set('部分窗口未能隐藏，请检查共享预览')
