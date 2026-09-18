import tkinter as tk
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from app import App
from chat_view import Bubble


class ResizeTests(unittest.TestCase):
    def test_drag_across_zero_keeps_absolute_screen_coordinates(self):
        root = tk.Tk()
        root.overrideredirect(True)
        try:
            root.geometry('640x400+20+30')
            root.update()
            app = App.__new__(App)
            app.root = root
            app._drag_origin = (50, 40)
            for x, y in ((1, 2), (-1, -2), (-200, -30), (20, 30)):
                app.drag_move(SimpleNamespace(x_root=x+50, y_root=y+40))
                root.update()
                self.assertEqual((root.winfo_rootx(), root.winfo_rooty()), (x, y))
        finally:
            root.destroy()

    def test_resize_preview_and_commit_use_absolute_negative_coordinates(self):
        app = App.__new__(App)
        app.root = Mock()
        app._resize_origin = (100, 100, 640, 400)
        app._resize_position = (20, 30)
        app._resize_edge = 'nw'
        preview = app._resize_preview = Mock()
        event = SimpleNamespace(x_root=50, y_root=50)
        app.resize_move(event)
        preview.geometry.assert_called_with('690x450+-30+-20')
        app.flush_resize(event)
        app.root.geometry.assert_called_once_with('690x450+-30+-20')

    def test_all_edges_keep_opposite_edge_fixed_until_release(self):
        expected = {'n': '640x370+20+60', 's': '640x430',
                    'w': '600x400+60+30', 'e': '680x400',
                    'nw': '600x370+60+60', 'ne': '680x370+20+60',
                    'sw': '600x430+60+30', 'se': '680x430'}
        for edge, geometry in expected.items():
            with self.subTest(edge=edge):
                app = App.__new__(App)
                app.root = Mock()
                app._resize_origin = (100, 100, 640, 400)
                app._resize_position = (20, 30)
                app._resize_edge = edge
                app._resize_preview = Mock()
                event = SimpleNamespace(x_root=140, y_root=130)
                app.resize_move(event)
                app.root.geometry.assert_not_called()
                app.flush_resize(event)
                app.root.geometry.assert_called_once_with(geometry)

    def test_top_left_resize_clamps_minimum_size(self):
        app = App.__new__(App)
        app.root = Mock()
        app._resize_origin = (100, 100, 640, 400)
        app._resize_position = (20, 30)
        app._resize_edge = 'nw'
        app._resize_preview = Mock()
        app.flush_resize(SimpleNamespace(x_root=1100, y_root=1100))
        app.root.geometry.assert_called_once_with('420x280+240+150')

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
