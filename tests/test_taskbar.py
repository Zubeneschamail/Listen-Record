"""Check native taskbar eligibility across Tk window remapping."""
import ctypes
import sys
import tkinter as tk
import unittest
from ctypes import wintypes

from app import App


@unittest.skipUnless(sys.platform == 'win32', 'Windows window styles')
class TaskbarTests(unittest.TestCase):
    def test_remapped_borderless_window_keeps_taskbar_entry(self):
        root = tk.Tk()
        app = App.__new__(App)
        app.root = root
        root.overrideredirect(True)
        root.bind('<Map>', app.on_main_window_mapped, add='+')
        api = ctypes.windll.user32
        api.GetParent.argtypes = [wintypes.HWND]
        api.GetParent.restype = wintypes.HWND
        api.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        api.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        try:
            for _ in range(3):
                root.update()
                hwnd = api.GetParent(root.winfo_id())
                style = api.GetWindowLongW(hwnd, -20)
                self.assertTrue(style & 0x40000)
                self.assertFalse(style & 0x80)
                root.withdraw()
                # Emulate Tk rebuilding the wrapper as a tool window.
                api.SetWindowLongW(hwnd, -20, (style | 0x80) & ~0x40000)
                root.deiconify()
        finally:
            for identifier in root.tk.call('after', 'info'):
                root.after_cancel(identifier)
            root.destroy()
