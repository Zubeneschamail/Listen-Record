"""Check native taskbar eligibility across privacy changes and Tk remapping."""
import ctypes
import sys
import tkinter as tk
import unittest
from ctypes import wintypes
from types import SimpleNamespace

from app import App


@unittest.skipUnless(sys.platform == 'win32', 'Windows window styles')
class TaskbarTests(unittest.TestCase):
    def test_borderless_taskbar_entry_follows_capture_privacy(self):
        root = tk.Tk()
        app = App.__new__(App)
        app.root = root
        app.capture_privacy = SimpleNamespace(enabled=False)
        root.overrideredirect(True)
        root.bind('<Map>', app.on_main_window_mapped, add='+')
        api = ctypes.windll.user32
        api.GetParent.argtypes = [wintypes.HWND]
        api.GetParent.restype = wintypes.HWND
        api.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        api.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        api.IsWindowVisible.argtypes = [wintypes.HWND]
        try:
            for hidden in (False, True, False, True):
                app.capture_privacy.enabled = hidden
                app.sync_taskbar()
                root.update()
                hwnd = api.GetParent(root.winfo_id())
                style = api.GetWindowLongW(hwnd, -20)
                self.assertEqual(bool(style & 0x40000), not hidden)
                self.assertEqual(bool(style & 0x80), hidden)
                self.assertTrue(api.IsWindowVisible(hwnd))
                root.withdraw()
                # Emulate Tk rebuilding the wrapper with the opposite style.
                wrong = (style | 0x40000) & ~0x80 if hidden else (style | 0x80) & ~0x40000
                api.SetWindowLongW(hwnd, -20, wrong)
                root.deiconify()
                root.update()
                style = api.GetWindowLongW(api.GetParent(root.winfo_id()), -20)
                self.assertEqual(bool(style & 0x40000), not hidden)
                self.assertEqual(bool(style & 0x80), hidden)
        finally:
            for identifier in root.tk.call('after', 'info'):
                root.after_cancel(identifier)
            root.destroy()
