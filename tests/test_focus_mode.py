import tkinter as tk
import unittest
from unittest.mock import patch
from app import App


class FocusModeTests(unittest.TestCase):
    def test_focus_mode_fills_existing_window_and_restores_chrome(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), \
             patch('qa_connection.QAConnection.check'):
            root = tk.Tk()
            app = App(root)
            try:
                root.geometry('620x420+30+40')
                root.update()
                for enabled in (False, True):
                    app.qa_enabled.set(enabled)
                    app.toggle_qa()
                    root.update()
                    original = root.geometry()
                    self.assertEqual(app.qa_panel.cget('cursor'), '')
                    # Content surfaces must not move the normal window, even
                    # when a previous title-bar drag left an origin behind.
                    app._drag_origin = (0, 0)
                    for surface in (app.qa_panel, app.chat.canvas, app.chat.inner):
                        surface.event_generate('<ButtonPress-1>', x=5, y=5, rootx=100, rooty=100)
                        surface.event_generate('<B1-Motion>', x=35, y=25, rootx=130, rooty=120)
                        root.update()
                        self.assertEqual(root.geometry(), original)
                    panes = len(app.columns.panes())
                    root.focus_force()
                    root.event_generate('<F11>')
                    root.update()
                    self.assertTrue(app.focus_mode)
                    self.assertEqual(len(app.columns.panes()), 2)
                    self.assertEqual(root.geometry(), original)
                    self.assertEqual(app.columns.winfo_width(), root.winfo_width())
                    self.assertEqual(app.columns.winfo_height(), root.winfo_height())
                    self.assertTrue(app.resize_handles['se'].winfo_ismapped())
                    for widget in (app.header, app.controls, app.status_label):
                        self.assertFalse(widget.winfo_ismapped())
                    root.event_generate('<Escape>')
                    root.update()
                    self.assertFalse(app.focus_mode)
                    self.assertEqual(root.geometry(), original)
                    self.assertEqual(len(app.columns.panes()), panes)
                    self.assertFalse(app.status_label.winfo_ismapped())
                    for widget in (app.header, app.controls, app.resize_handles['se']):
                        self.assertTrue(widget.winfo_ismapped())
                app.toggle_focus_mode()
                root.update()
                x, y = root.winfo_rootx(), root.winfo_rooty()
                app.qa_panel.event_generate('<ButtonPress-1>', x=5, y=5, rootx=x+5, rooty=y+5)
                app.qa_panel.event_generate('<B1-Motion>', x=35, y=25, rootx=x+35, rooty=y+25)
                root.update()
                self.assertEqual((root.winfo_rootx(), root.winfo_rooty()), (x+30, y+20))
                grip = app.resize_handles['se']
                grip.event_generate('<ButtonPress-1>', x=2, y=2, rootx=650, rooty=450)
                grip.event_generate('<B1-Motion>', x=42, y=32, rootx=690, rooty=480)
                root.update()
                self.assertEqual(root.winfo_width(), 620)
                grip.event_generate('<ButtonRelease-1>', x=42, y=32, rootx=690, rooty=480)
                root.update()
                self.assertEqual((root.winfo_width(), root.winfo_height()), (660, 450))
                changed = root.geometry()
                app.exit_focus_mode()
                root.update()
                self.assertEqual(root.geometry(), changed)
            finally:
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()
