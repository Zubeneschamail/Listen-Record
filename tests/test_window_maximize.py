import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import App
from window_effects import work_area


class MaximizeTests(unittest.TestCase):
    def setUp(self):
        for name in ('app.GlobalHotkey', 'app.App.start_tray', 'app.ClipboardWatcher',
                     'qa_connection.QAConnection.check', 'app.save_desktop'):
            mocked = patch(name)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.root = tk.Tk()
        self.app = App(self.root)
        self.root.geometry('560x380+80+90')
        self.root.update()

    def tearDown(self):
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.app.close()

    def bounds(self):
        self.root.update()
        return (self.root.winfo_width(), self.root.winfo_height(),
                self.root.winfo_rootx(), self.root.winfo_rooty())

    def test_button_order_work_area_restore_and_tray_roundtrip(self):
        app = self.app
        buttons = [w for w in app.header.winfo_children() if isinstance(w, tk.Button)]
        ordered = [w.cget('text') for w in sorted(buttons, key=lambda w: w.winfo_x())]
        self.assertEqual(ordered[-3:], ['—', '最大化', '×'])
        original = self.bounds()
        left, top, right, bottom = work_area(self.root)
        for _ in range(2):
            app.maximize_button.invoke()
            self.assertEqual(self.bounds(), (right-left, bottom-top, left, top))
            self.assertEqual(app.maximize_button.cget('text'), '恢复')
            self.assertEqual(str(app.maximize_button.cget('image')), str(app.maximize_button.images['restore']))
            self.root.withdraw()
            self.root.deiconify()
            self.assertEqual(self.bounds(), (right-left, bottom-top, left, top))
            app.maximize_button.invoke()
            self.assertEqual(self.bounds(), original)
            self.assertEqual(app.maximize_button.cget('text'), '最大化')

    def test_drag_restores_under_pointer_and_maximized_edges_do_not_resize(self):
        app = self.app
        app.toggle_maximize()
        self.root.update()
        event = SimpleNamespace(x_root=self.root.winfo_rootx()+self.root.winfo_width()//2,
                                y_root=self.root.winfo_rooty()+20)
        app.resize_begin(event)
        self.assertIsNone(getattr(app, '_resize_preview', None))
        app.drag_begin(event)
        app.drag_move(SimpleNamespace(x_root=500, y_root=200))
        self.assertFalse(app.maximized)
        self.assertEqual(self.bounds(), (560, 380, 220, 180))

    def test_negative_monitor_coordinates_and_theme_keep_restore_icon(self):
        app = self.app
        with patch('window_effects.work_area', return_value=(-1920, -100, 0, 940)):
            app.toggle_maximize()
        self.assertEqual(self.bounds(), (1920, 1040, -1920, -100))
        app.dark_mode.set(True)
        app.apply_theme()
        self.assertEqual(str(app.maximize_button.cget('image')), str(app.maximize_button.images['restore']))
        app.toggle_maximize()
        self.assertEqual(self.bounds(), (560, 380, 80, 90))


if __name__ == '__main__':
    unittest.main()
