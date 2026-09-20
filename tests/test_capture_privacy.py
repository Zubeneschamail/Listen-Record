import ctypes
from ctypes import wintypes
import sys
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

from capture_privacy import CapturePrivacy, CapturePrivacyError, display_api


@unittest.skipUnless(sys.platform == 'win32' and sys.getwindowsversion().build >= 19041,
                     'Windows capture exclusion support')
class CapturePrivacyTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.privacy = CapturePrivacy(self.root)
        self.root.update()

    def tearDown(self):
        self.privacy.set_enabled(False)
        self.root.destroy()

    def affinity(self, path):
        api = display_api()
        hwnd = api.GetAncestor(int(str(self.root.tk.call('winfo', 'id', path)), 0), 2)
        result = wintypes.DWORD()
        self.assertTrue(api.GetWindowDisplayAffinity(hwnd, ctypes.byref(result)))
        return result.value

    def test_native_exclusion_covers_existing_new_and_remapped_windows(self):
        settings = tk.Toplevel(self.root)
        combo = ttk.Combobox(settings, values=['one', 'two'])
        combo.pack()
        # Tcl-created popdown windows also belong to the application.
        popup = str(self.root.tk.call('ttk::combobox::PopdownWindow', combo))
        self.root.update()
        self.privacy.set_enabled(True)
        for path in (str(self.root), str(settings), popup):
            self.assertEqual(self.affinity(path), 0x11)
        self.assertTrue(self.root.winfo_viewable())
        self.assertTrue(settings.winfo_viewable())
        caption = tk.Toplevel(self.root)
        caption.overrideredirect(True)
        caption.attributes('-transparentcolor', '#010203')
        self.root.update()
        self.assertEqual(self.affinity(str(caption)), 0x11)
        settings.withdraw()
        settings.deiconify()
        self.root.update()
        self.assertEqual(self.affinity(str(settings)), 0x11)
        self.privacy.set_enabled(False)
        for path in self.privacy.windows():
            self.assertEqual(self.affinity(path), 0)

    def test_startup_preference_is_applied_when_windows_map(self):
        self.root.withdraw()
        self.privacy.enabled = True
        self.root.deiconify()
        self.root.update()
        self.assertEqual(self.affinity(str(self.root)), 0x11)

    def test_partial_failure_restores_previous_mode(self):
        settings = tk.Toplevel(self.root)
        self.root.update()
        from capture_privacy import set_window_capture
        def fail_child(root, path, enabled):
            if path == str(settings) and enabled:
                raise CapturePrivacyError('native failure')
            return set_window_capture(root, path, enabled)
        with patch('capture_privacy.set_window_capture', side_effect=fail_child):
            with self.assertRaises(CapturePrivacyError):
                self.privacy.set_enabled(True)
        self.assertFalse(self.privacy.enabled)
        self.assertEqual(self.affinity(str(self.root)), 0)
        self.assertEqual(self.affinity(str(settings)), 0)


if __name__ == '__main__':
    unittest.main()
