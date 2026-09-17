import tkinter as tk
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from app import App
from chat_view import Bubble


class ResizeTests(unittest.TestCase):
    def test_drag_only_updates_outline_and_release_commits_latest_size(self):
        app = App.__new__(App)
        app.root = Mock()
        app.root.winfo_rootx.return_value = 20
        app.root.winfo_rooty.return_value = 30
        app._resize_origin = (100, 100, 640, 400)
        preview = app._resize_preview = Mock()
        app.resize_move(SimpleNamespace(x_root=180, y_root=140))
        app.resize_move(SimpleNamespace(x_root=200, y_root=180))
        app.root.geometry.assert_not_called()
        preview.geometry.assert_called_with('740x480+20+30')
        app.flush_resize(SimpleNamespace(x_root=210, y_root=190))
        app.root.geometry.assert_called_once_with('750x490')
        preview.destroy.assert_called_once()
        self.assertIsNone(app._resize_preview)

    def test_reflow_reuses_measurements_but_text_revision_invalidates_them(self):
        root = tk.Tk()
        root.withdraw()
        try:
            bubble = Bubble(root, {'id': 1, 'text': '测试文字 English 123。' * 8},
                            lambda *args: None, lambda *args: None)
            bubble.layout(400)
            wide_height = int(bubble.cget('height'))
            with patch.object(bubble.font, 'measure', wraps=bubble.font.measure) as measure:
                bubble.layout(180)
                self.assertGreater(int(bubble.cget('height')), wide_height)
                measure.assert_not_called()
                bubble.row = dict(id=1, text='修正后的短句')
                bubble.layout(180)
                self.assertTrue(measure.called)
                self.assertEqual(bubble.measured_text, '修正后的短句')
                self.assertLess(int(bubble.cget('height')), wide_height)
        finally:
            root.destroy()
