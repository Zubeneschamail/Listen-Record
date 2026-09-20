import tkinter as tk
import tkinter.font as tkfont
import unittest
from unittest.mock import patch
from app import App


class FontSizeTests(unittest.TestCase):
    def test_size_updates_existing_and_future_bubbles_and_is_saved(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), \
                patch('qa_connection.QAConnection.check'), \
                patch('app.preferences', return_value={}), \
                patch('app.save_preferences'), patch('app.save_desktop') as save:
            root = tk.Tk()
            app = App(root)
            try:
                self.assertEqual(app.font_size.get(), 9)
                app.chat.add(1, {'id': 1, 'text': '字号预览文字' * 12})
                root.update()
                bubble = app.chat.bubbles[1]
                old_height = bubble.winfo_height()
                app.font_size.set(14)
                app.font_size_picker.event_generate('<<ComboboxSelected>>')
                root.update()
                self.assertEqual(bubble.font.actual('size'), 14)
                self.assertGreater(bubble.winfo_height(), old_height)
                self.assertEqual(tkfont.Font(root=root, font=app.qa_text.cget('font')).actual('size'), 14)
                self.assertEqual(save.call_args.args[0]['font_size'], 14)
                app.chat.add(2, {'id': 2, 'text': '新文字'})
                self.assertEqual(app.chat.bubbles[2].font.actual('size'), 14)
                app.font_size.set(9)
                app.change_font_size()
                root.update()
                self.assertEqual(bubble.winfo_height(), old_height)
            finally:
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()
