from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch
from app import App


class ReferenceSettingsTests(unittest.TestCase):
    def test_knowledge_package_validation_does_not_block_ui(self):
        started, release, ui_tick = threading.Event(), threading.Event(), threading.Event()
        package = r'F:\slow\engine.wlkb'

        def slow_validation(value):
            started.set()
            if not release.wait(5):
                raise TimeoutError('validation stalled')
            return value

        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), \
                patch('qa_connection.QAConnection.check'), \
                patch('reference_settings.load_settings', return_value={'enabled': False, 'paths': []}), \
                patch('reference_settings.save_settings') as save, \
                patch('reference_settings.validate_source', side_effect=slow_validation), \
                patch('knowledge_packages.validate_selection'):
            root = tk.Tk()
            root.withdraw()
            app = App(root)
            try:
                app.reference_controls['add']([package])
                self.assertTrue(started.wait(1))
                self.assertEqual(app.reference_controls['listing'].size(), 0)
                root.after(0, ui_tick.set)
                root.update()
                self.assertTrue(ui_tick.is_set())
                release.set()
                deadline = time.monotonic() + 5
                while not save.called:
                    self.assertLess(time.monotonic(), deadline)
                    root.update()
                    time.sleep(.01)
                self.assertEqual(save.call_args.args[0]['paths'], [package])
            finally:
                release.set()
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()

    def test_add_persist_remove_without_deleting_source(self):
        data = {'enabled': False, 'paths': []}
        with tempfile.TemporaryDirectory() as directory, patch('app.GlobalHotkey'), \
                patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'), \
                patch('reference_settings.load_settings', return_value=dict(data)), \
                patch('reference_settings.save_settings') as save:
            file = Path(directory)/'manual.md'
            file.write_text('sample', encoding='utf-8')
            root = tk.Tk()
            app = App(root)
            try:
                # Main-panel actions work without opening settings and share its list.
                with patch('reference_settings.filedialog.askopenfilenames', return_value=(str(file),)) as picker:
                    app.reference_add_menu.invoke(0)
                    self.assertIs(picker.call_args.kwargs['parent'], root)
                self.assertEqual(app.reference_controls['listing'].size(), 1)
                self.assertEqual(app.reference_path_label.paths, [str(file.resolve())])
                app.reference_path_label.configure(width=16)
                root.update()
                item = app.reference_path_label.items[0]
                self.assertIn('...', item.cget('text'))
                self.assertLessEqual(item.winfo_width(), 224)
                item.show_tip()
                tip_text = item.tip.winfo_children()[0].cget('text')
                self.assertIn(str(file.resolve()), tip_text)
                item.hide_tip()
                with patch('reference_settings.filedialog.askdirectory', return_value=''):
                    app.reference_add_menu.invoke(1)
                self.assertEqual(app.reference_controls['listing'].size(), 1)
                app.toggle_settings()
                app.settings_tabs.select(app.settings_pages['references'])
                root.update()
                controls = app.reference_controls
                controls['add']([str(file), str(file)])
                self.assertEqual(controls['listing'].size(), 1)
                self.assertTrue(controls['enabled'].get())
                self.assertEqual(save.call_args.args[0]['paths'], [str(file.resolve())])
                controls['listing'].selection_set(0)
                controls['remove']()
                self.assertEqual(controls['listing'].size(), 0)
                self.assertEqual(app.reference_path_label.paths, [])
                app.settings_save_button.invoke()
                self.assertEqual(save.call_args.args[0]['paths'], [])
                self.assertTrue(file.exists())
                # Each path's hover action removes only its own reference.
                controls['add']([str(file), directory])
                root.update()
                first, second = app.reference_path_label.items
                first.schedule_tip()
                root.update()
                self.assertTrue(first.close_button.winfo_manager())
                from tkinter import font as tkfont
                self.assertLessEqual(first.close_button.winfo_x(),
                                     tkfont.Font(font=first.cget('font')).measure(first.cget('text')) + 4)
                self.assertFalse(second.close_button.winfo_manager())
                first.close_button.invoke()
                self.assertEqual(app.reference_path_label.paths, [str(Path(directory).resolve())])
                self.assertEqual(controls['listing'].size(), 1)
                self.assertEqual(save.call_args.args[0]['paths'], [str(Path(directory).resolve())])
                self.assertTrue(file.exists())
                app.reference_path_label.items[0].close_button.invoke()
                self.assertEqual(app.reference_path_label.paths, [])
                self.assertTrue(Path(directory).is_dir())
                self.assertEqual(app.settings_window.winfo_height(), 440)
                self.assertEqual(app.settings_window.winfo_width(), 480)
            finally:
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()
