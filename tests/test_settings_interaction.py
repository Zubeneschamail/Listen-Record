import tkinter as tk
from tkinter import ttk
import ctypes
from ctypes import wintypes
from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import App, NO_AUDIO_DEVICE
from qa_settings_dialog import show


class SettingsInteractionTests(unittest.TestCase):
    def setUp(self):
        # Personal desktop settings must not change the expected initial UI state.
        preferences = patch('app.preferences', return_value={})
        preferences.start()
        self.addCleanup(preferences.stop)
        for name in ('app.GlobalHotkey', 'app.App.start_tray', 'app.ClipboardWatcher',
                     'qa_connection.QAConnection.check', 'app.save_desktop',
                     'app.save_preferences'):
            mocked = patch(name)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.root = tk.Tk()
        self.app = App(self.root)
        self.app.toggle_settings()
        self.root.update()

    def tearDown(self):
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.app.close()

    def test_manual_cut_mouse_is_saved_and_cancel_restores_draft(self):
        app = self.app
        original = app.manual_cut_mouse.get()
        app.manual_cut_mouse.set('关闭')
        app.hide_settings()
        self.assertEqual(app.manual_cut_mouse.get(), original)
        app.toggle_settings()
        app.manual_cut_mouse.set('侧键 2（前进）')
        with patch('app.save_desktop') as save, patch('reference_settings.save_settings'):
            app.settings_save_button.invoke()
        self.assertEqual(save.call_args.args[0]['manual_cut_mouse'], '侧键 2（前进）')
        self.assertIsNone(app.cut_mouse)  # Idle: no hook is installed just by saving.

    def test_qwen_model_help_explains_acoustic_hint_behavior(self):
        from qwen_engine import QWEN_LABEL
        self.app.model.set(QWEN_LABEL)
        self.assertIn('普通热词不参与',self.app.recognition_help.get())
        self.assertIn('纠错规则仍然生效',self.app.recognition_help.get())
        self.app.model.set('small · 轻量准确')
        self.assertIn('普通词用于识别提示',self.app.recognition_help.get())

    def test_model_loading_shows_capture_buffering_until_backend_is_ready(self):
        self.app.events.put(('capture_loading', 'qwen3-asr'))
        self.app.poll()
        self.assertEqual(self.app.storage_hint.get(), '音频暂存中…')
        self.assertIn('已开始采集', self.app.gpu_runtime_status.get())
        self.app.events.put(('recognition_backend', ('qwen3-asr', 'cuda')))
        self.app.poll()
        self.assertEqual(self.app.storage_hint.get(), '')
        self.assertIn('GPU 加速已启用', self.app.gpu_runtime_status.get())

    def test_missing_qwen_component_does_not_start_a_whisper_download(self):
        from qwen_engine import QWEN_MODEL
        current=self.app.model.get()
        with patch('model_download.cached_model',return_value=None),patch('multiprocessing.get_context') as context:
            self.app.open_model_manager()
            window=self.app._model_window
            choice=next(w for w in window.winfo_children() if isinstance(w,ttk.Combobox))
            index=next(i for i,label in enumerate(choice['values']) if label.split()[0]==QWEN_MODEL)
            choice.current(index);choice.event_generate('<<ComboboxSelected>>')
            self.root.update()
            button=next(w for w in window.winfo_children() if isinstance(w,tk.Button))
            self.assertEqual(button.cget('text'),'安装说明')
            button.invoke()
            context.assert_not_called()
            labels=[self.root.getvar(w.cget('textvariable')) for w in window.winfo_children()
                    if isinstance(w,ttk.Label) and w.cget('textvariable')]
            self.assertTrue(any('BF16' in text and '组件' in text for text in labels))
            self.assertEqual(self.app.model.get(),current)
            window.destroy()

    def test_sections_are_separated_and_redundant_actions_removed(self):
        tabs = self.app.settings_tabs
        self.assertEqual([button.cget('text') for _, button in tabs.pages.values()],
                         ['转录', '答疑', '资料', '外观', '数据'])
        appearance = str(self.app.settings_pages['appearance'])
        self.assertTrue(str(self.app.font_size_picker).startswith(appearance + '.'))
        self.assertTrue(str(self.app.pin_button).startswith(appearance + '.'))
        self.assertEqual(set(self.app.settings_actions), {'导出记录', '保存问答副本', '打开练习', '历史记录', '关于与更新'})
        self.assertFalse(hasattr(self.app, 'startup_notice'))
        self.assertEqual(self.app.settings_save_button.cget('text'), '保存')

    def test_interview_terms_are_optional_and_cancel_restores_original(self):
        app = self.app
        original = app.hotwords.get()
        app.hotwords.set('密等性=幂等性')
        app.hotwords_presets.invoke(0)
        self.assertIn('LangChain', app.hotwords.get())
        self.assertIn('密等性=幂等性', app.hotwords.get())
        self.root.update()
        self.assertGreater(app.hotwords_entry.winfo_width(), 100)
        app.hide_settings()
        self.assertEqual(app.hotwords.get(), original)

    def test_click_raises_settings_without_pinning_app_or_changing_input_focus(self):
        app = self.app
        app.hotwords_entry.focus_force()
        self.root.update()
        app.settings_window.lower(self.root)
        app.hotwords_entry.event_generate('<ButtonPress-1>', x=5, y=5)
        self.root.update()
        self.assertTrue(self.root.tk.call('wm', 'stackorder', app.settings_window, 'isabove', self.root))
        self.assertIs(self.root.focus_get(), app.hotwords_entry)
        self.assertFalse(self.root.attributes('-topmost'))
        self.assertFalse(app.settings_window.attributes('-topmost'))
        app.toggle_pin()
        self.assertTrue(app.settings_window.attributes('-topmost'))
        app.toggle_pin()
        self.assertFalse(app.settings_window.attributes('-topmost'))

    def test_reopening_settings_keeps_model_dialog_in_front(self):
        window = show(self.app)
        self.root.update()
        self.app.toggle_settings()
        self.root.update()
        self.assertIs(self.root.grab_current(), window)
        self.assertTrue(self.root.tk.call('wm', 'stackorder', window, 'isabove', self.app.settings_window))

    def test_cancel_service_dialog_returns_keyboard_focus_to_settings(self):
        self.app.hotwords_entry.focus_force()
        self.root.update()
        window = show(self.app)
        self.root.update()
        fields = next(child for child in window.winfo_children() if isinstance(child, ttk.Frame))
        entry = next(child for child in fields.winfo_children() if isinstance(child, ttk.Entry))
        entry.focus_force()
        self.root.update()
        actions = [child for child in window.winfo_children() if isinstance(child, ttk.Frame)][-1]
        cancel = next(child for child in actions.winfo_children() if child.cget('text') == '取消')
        cancel.invoke()
        self.root.update()
        self.assertIs(self.root.grab_current(), self.app.settings_window)
        self.assertIs(self.root.focus_get(), self.app.hotwords_entry)
        self.root.focus_get().event_generate('<Escape>')
        self.root.update()
        self.assertFalse(self.app.settings_visible)
        self.assertIsNone(self.root.grab_current())

    def test_escape_cancels_service_draft_before_closing_parent_settings(self):
        self.app.settings_tabs.focus_force()
        self.root.update()
        original = self.app.qa_settings.copy()
        window = show(self.app)
        window.focus_force()
        self.root.update()
        with patch('qa_settings_dialog.save_settings') as save:
            window.event_generate('<Escape>')
            self.root.update()
        save.assert_not_called()
        self.assertEqual(self.app.qa_settings, original)
        self.assertFalse(window.winfo_exists())
        self.assertTrue(self.app.settings_visible)
        self.assertIs(self.root.grab_current(), self.app.settings_window)
        self.assertIs(self.root.focus_get(), self.app.settings_tabs)

    def test_close_model_manager_restores_parent_focus_and_escape(self):
        self.app.model_manage_button.focus_force()
        self.root.update()
        self.app.open_model_manager()
        window = self.app._model_window
        self.root.update()
        window.focus_force()
        self.root.update()
        window.event_generate('<Escape>')
        self.root.update()
        self.assertFalse(window.winfo_exists())
        self.assertIs(self.root.grab_current(), self.app.settings_window)
        self.assertIs(self.root.focus_get(), self.app.model_manage_button)
        self.root.focus_get().event_generate('<Escape>')
        self.root.update()
        self.assertFalse(self.app.settings_visible)

    def test_escape_cancels_model_download_and_restores_keyboard_control(self):
        from unittest.mock import Mock
        callback_errors = []
        self.root.report_callback_exception = lambda *error: callback_errors.append(error)
        receiver, sender, job, context = Mock(), Mock(), Mock(), Mock()
        context.Pipe.return_value = receiver, sender
        context.Process.return_value = job
        receiver.poll.return_value = False
        job.is_alive.return_value = True
        self.app.model_manage_button.focus_force()
        self.root.update()
        with patch('model_download.cached_model', return_value=None), \
                patch('multiprocessing.get_context', return_value=context):
            self.app.open_model_manager()
            window = self.app._model_window
            choice = next(child for child in window.winfo_children() if isinstance(child, ttk.Combobox))
            choice.current(0)
            choice.event_generate('<<ComboboxSelected>>')
            action = next(child for child in window.winfo_children() if isinstance(child, tk.Button))
            action.invoke()
            self.assertEqual(action.cget('text'), '取消下载')
            window.focus_force()
            self.root.update()
            window.event_generate('<Escape>')
            self.root.update()
        job.terminate.assert_called_once()
        job.join.assert_called_once_with(timeout=1)
        receiver.close.assert_called_once()
        self.assertFalse(window.winfo_exists())
        self.assertIs(self.root.focus_get(), self.app.model_manage_button)
        self.assertIs(self.root.grab_current(), self.app.settings_window)
        self.assertEqual(callback_errors, [])

    def test_startup_service_dialog_without_previous_focus_can_escape_back_to_app(self):
        self.root.tk.call('focus', '')
        self.assertIsNone(self.root.focus_get())
        window = show(self.app)
        window.focus_force()
        self.root.update()
        window.event_generate('<Escape>')
        self.root.update()
        self.assertIs(self.root.focus_get(), self.app.settings_window)
        self.assertIs(self.root.grab_current(), self.app.settings_window)
        self.root.focus_get().event_generate('<Escape>')
        self.root.update()
        self.assertFalse(self.app.settings_visible)
        self.assertIsNone(self.root.grab_current())

    def test_escape_on_owner_closes_active_child_without_orphaning_it(self):
        for kind in ('service', 'model'):
            with self.subTest(dialog=kind):
                if kind == 'service':
                    window = show(self.app)
                else:
                    self.app.open_model_manager()
                    window = self.app._model_window
                self.root.update()
                self.app.settings_tabs.focus_force()
                self.root.update()
                self.assertIs(self.root.grab_current(), window)
                self.app.hide_settings(SimpleNamespace(keysym='Escape'))
                self.root.update()
                self.assertFalse(window.winfo_exists())
                self.assertTrue(self.app.settings_visible)
                self.assertTrue(self.app.settings_window.winfo_viewable())
                self.assertIs(self.root.grab_current(), self.app.settings_window)

    def test_cancel_restores_drafts_and_preview_without_saving(self):
        app = self.app
        original = (app.hotwords.get(), app.font_size.get(), app.dark_mode.get(), app.mode.get())
        refs = app.reference_controls['snapshot']()
        with patch('app.save_preferences') as hotwords, patch('app.save_desktop') as desktop, \
                patch('reference_settings.save_settings') as references, \
                patch.object(app.clipboard_watcher, 'start') as start:
            app.hotwords.set('尚未保存')
            app.font_size.set(14)
            app.change_font_size()
            app.dark_mode.set(not original[2])
            app.change_theme()
            app.mode.current((app.mode.current()+1) % 3)
            app.settings_auto_qa.set(True)
            app.reference_controls['enabled'].set(not refs['enabled'])
            app.hide_settings()
            self.assertEqual((app.hotwords.get(), app.font_size.get(), app.dark_mode.get(), app.mode.get()), original)
            self.assertEqual(app.reference_controls['snapshot'](), refs)
            hotwords.assert_not_called()
            desktop.assert_not_called()
            references.assert_not_called()
            start.assert_not_called()
            app.toggle_settings()
            self.assertEqual(app.hotwords.get(), original[0])

    def test_echo_switch_is_saved_explicitly_and_cancel_restores_it(self):
        app = self.app
        original = app.echo_cancellation.get()
        app.echo_cancellation.set(not original)
        app.hide_settings()
        self.assertEqual(app.echo_cancellation.get(), original)
        app.toggle_settings()
        with patch('app.save_desktop') as save, patch('app.save_preferences'), \
                patch('reference_settings.save_settings'), patch.object(app.engine, 'start') as start:
            app.echo_checkbox.invoke()
            save.assert_not_called()
            app.settings_save_button.invoke()
            self.assertEqual(save.call_args.args[0]['echo_cancellation'], not original)
            start.assert_not_called()
        app.toggle_settings()
        self.assertEqual(app.echo_cancellation.get(), not original)

    def test_question_source_is_draft_until_save_and_automatically_handles_mic_only(self):
        app = self.app
        self.assertEqual(app.qa.question_source, 'system')
        app.question_source.set('麦克风')
        self.assertEqual(app.qa.question_source, 'system')
        app.hide_settings()
        self.assertEqual(app.question_source.get(), '自动（优先系统声音）')
        app.toggle_settings()
        app.question_source.set('麦克风')
        with patch('app.save_desktop') as save, patch('reference_settings.save_settings'):
            app.settings_save_button.invoke()
        self.assertEqual(app.qa.question_source, 'microphone')
        self.assertEqual(save.call_args.args[0]['question_source'], '麦克风')
        app.question_source.set('两路声音')
        with patch('app.preferences', return_value=save.call_args.args[0]):
            app.load_desktop_settings()
        self.assertEqual(app.question_source.get(), '麦克风')
        self.assertEqual(app.qa.question_source, 'microphone')
        app.toggle_settings()
        app.question_source.set('自动（优先系统声音）')
        app.recording_sources = {'microphone'}
        app.apply_question_source()
        self.assertEqual(app.qa.question_source, 'microphone')
        app.recording_sources = {'system', 'microphone'}
        app.update_question_source()
        self.assertEqual(app.qa.question_source, 'system')

    def test_failed_save_does_not_apply_question_source(self):
        app = self.app
        app.question_source.set('麦克风')
        with patch.object(app, 'save_desktop_settings', side_effect=OSError('write failed')), \
                patch('app.messagebox.showerror'):
            app.settings_save_button.invoke()
        self.assertEqual(app.qa.question_source, 'system')
        self.assertEqual(app.question_source.get(), '麦克风')

    def test_unselected_audio_devices_save_cancel_and_reload(self):
        app = self.app
        original = (app.device.get(), app.microphone.get())
        for widget in (app.device, app.microphone):
            self.assertIn(NO_AUDIO_DEVICE, widget['values'])
            widget.set(NO_AUDIO_DEVICE)
        app.hide_settings()
        self.assertEqual((app.device.get(), app.microphone.get()), original)
        app.toggle_settings()
        app.device.set(NO_AUDIO_DEVICE)
        app.microphone.set(NO_AUDIO_DEVICE)
        with patch('app.save_desktop') as save, patch('app.save_preferences'), \
                patch('reference_settings.save_settings'):
            app.settings_save_button.invoke()
            saved = save.call_args.args[0]
        self.assertEqual((saved['output'], saved['input']), (NO_AUDIO_DEVICE, NO_AUDIO_DEVICE))
        app.device.set('')
        app.microphone.set('')
        with patch('app.preferences', return_value=saved):
            app.load_desktop_settings()
        self.assertEqual((app.device.get(), app.microphone.get()), (NO_AUDIO_DEVICE, NO_AUDIO_DEVICE))

    def test_refresh_preserves_none_and_tracks_real_device_after_reordering(self):
        app = self.app
        app.devices = []
        app.input_devices = []
        app.device.set('')
        app.microphone.set('')
        outputs = [{'name': '扬声器 A', 'index': 10}, {'name': '扬声器 B', 'index': 11}]
        inputs = [{'name': '麦克风 A', 'index': 20, 'hostApi': 0, 'maxInputChannels': 1}]
        with patch('app.pa.PyAudio') as factory:
            audio = factory.return_value.__enter__.return_value
            audio.get_loopback_device_info_generator.side_effect = lambda: iter(outputs)
            audio.get_host_api_info_by_type.return_value = {'index': 0, 'defaultInputDevice': 20}
            audio.get_default_wasapi_loopback.return_value = {'index': 11}
            audio.get_device_count.side_effect = lambda: len(inputs)
            audio.get_device_info_by_index.side_effect = lambda i: inputs[i]
            app.refresh()
            self.assertEqual((app.device.get(), app.microphone.get()), ('扬声器 B', '麦克风 A'))
            app.microphone.set(NO_AUDIO_DEVICE)
            outputs.reverse()
            app.refresh()
            self.assertEqual((app.device.get(), app.microphone.get()), ('扬声器 B', NO_AUDIO_DEVICE))
            app.device.set(NO_AUDIO_DEVICE)
            app.refresh()
            self.assertEqual((app.device.get(), app.microphone.get()), (NO_AUDIO_DEVICE, NO_AUDIO_DEVICE))
            outputs.clear()
            inputs.clear()
            app.refresh()
            self.assertEqual(tuple(app.device['values']), (NO_AUDIO_DEVICE,))
            self.assertEqual(tuple(app.microphone['values']), (NO_AUDIO_DEVICE,))

    def test_recording_skips_unselected_sources_and_rejects_empty_capture(self):
        app = self.app
        app.hide_settings()
        app.devices = [{'index': 10}]
        app.input_devices = [{'index': 20}]
        app.device.configure(values=['扬声器', NO_AUDIO_DEVICE])
        app.microphone.configure(values=['麦克风', NO_AUDIO_DEVICE])
        with tempfile.TemporaryDirectory() as folder, patch('app.RECORDINGS', Path(folder)), \
                patch('app.save_preferences'), patch.object(app.engine, 'start') as start, \
                patch.object(app, 'toggle_settings') as settings, patch('app.messagebox.showinfo') as info:
            for mode, output, microphone, expected in (
                    ('系统声音 + 麦克风', 0, 1, [{'index': 10, 'source': 'system'}]),
                    ('系统声音 + 麦克风', 1, 0, [{'index': 20, 'source': 'microphone'}]),
                    ('系统声音 + 麦克风', 0, 0, [{'index': 10, 'source': 'system'}, {'index': 20, 'source': 'microphone'}]),
                    ('系统声音 + 麦克风', 1, 1, []),
                    ('仅系统声音', 1, 0, []),
                    ('仅麦克风', 0, 1, [])):
                with self.subTest(mode=mode, output=output, microphone=microphone):
                    app.set_busy(False)
                    app.capture_mode.set(mode)
                    app.device.current(output)
                    app.microphone.current(microphone)
                    start.reset_mock()
                    settings.reset_mock()
                    info.reset_mock()
                    files_before = list(Path(folder).iterdir())
                    app.start()
                    if expected:
                        self.assertEqual(start.call_args.args[0], expected)
                        info.assert_not_called()
                    else:
                        start.assert_not_called()
                        settings.assert_called_once()
                        info.assert_called_once()
                        self.assertFalse(app.busy)
                        self.assertEqual(list(Path(folder).iterdir()), files_before)
            app.set_busy(False)

    def test_save_commits_settings_and_only_then_enables_automatic_requests(self):
        app = self.app
        with patch('app.save_preferences') as hotwords, patch('app.save_desktop') as desktop, \
                patch('reference_settings.save_settings') as references, \
                patch.object(app.clipboard_watcher, 'start') as start:
            app.hotwords.set('闻录,新词')
            app.font_size.set(12)
            app.change_font_size()
            app.settings_auto_qa.set(True)
            start.assert_not_called()
            desktop.assert_not_called()
            app.settings_save_button.invoke()
            self.assertFalse(app.settings_visible)
            self.assertIsNone(self.root.grab_current())
            self.assertEqual(hotwords.call_args.args[1], '闻录,新词')
            self.assertEqual(desktop.call_args.args[0]['font_size'], 12)
            references.assert_called_once()
            start.assert_called_once()
            # The writes above are mocks; model their persisted value when the
            # dialog now refreshes the saved word list on reopening.
            with patch('app.load_wordlist_snapshot', return_value=('闻录,新词', None)) as read:
                app.toggle_settings()
                read.assert_called_once()
            self.assertEqual(app.hotwords.get(), '闻录,新词')
            self.assertTrue(app.settings_auto_qa.get())

    def test_failed_save_rolls_back_files_and_keeps_dialog_draft(self):
        app = self.app
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            original = path / 'desktop-settings.json'
            original.write_bytes(b'original settings')
            def desktop_save(values):
                original.write_bytes(b'new settings')
            with patch('settings_session.DATA', path), patch('app.DATA', path), \
                    patch('app.save_desktop', side_effect=desktop_save), \
                    patch('reference_settings.save_settings', side_effect=OSError('disk failure')), \
                    patch('app.messagebox.showerror') as error:
                app.hotwords.set('失败时保留草稿')
                app.settings_save_button.invoke()
                self.assertTrue(app.settings_visible)
                self.assertEqual(app.hotwords.get(), '失败时保留草稿')
                self.assertEqual(original.read_bytes(), b'original settings')
                self.assertFalse((path / 'recognition-settings.json').exists())
                error.assert_called_once()

    def test_capture_privacy_only_applies_after_save_and_can_be_disabled(self):
        app = self.app
        api = ctypes.windll.user32
        api.GetParent.argtypes = [wintypes.HWND]
        api.GetParent.restype = wintypes.HWND
        api.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        def taskbar_style():
            return api.GetWindowLongW(api.GetParent(self.root.winfo_id()), -20)
        self.assertFalse(app.capture_privacy.enabled)
        app.capture_hidden.set(True)
        self.assertFalse(app.capture_privacy.enabled)
        app.hide_settings()
        self.assertFalse(app.capture_hidden.get())
        app.toggle_settings()
        app.capture_hidden.set(True)
        with patch('app.save_desktop') as save, patch('reference_settings.save_settings'):
            app.settings_save_button.invoke()
            self.assertTrue(app.capture_privacy.enabled)
            self.assertTrue(save.call_args.args[0]['capture_hidden'])
            self.assertTrue(taskbar_style() & 0x80)
            self.assertFalse(taskbar_style() & 0x40000)
            app.toggle_settings()
            app.capture_hidden.set(False)
            self.assertTrue(app.capture_privacy.enabled)
            app.settings_save_button.invoke()
            self.assertFalse(app.capture_privacy.enabled)
            self.assertFalse(save.call_args.args[0]['capture_hidden'])
            self.assertFalse(taskbar_style() & 0x80)
            self.assertTrue(taskbar_style() & 0x40000)

    def test_failed_save_restores_previous_capture_mode(self):
        app = self.app
        app.capture_hidden.set(True)
        with patch.object(app, 'save_desktop_settings', side_effect=OSError('write failed')), \
                patch('app.messagebox.showerror') as error:
            app.settings_save_button.invoke()
            self.assertFalse(app.capture_privacy.enabled)
            self.assertTrue(app.capture_hidden.get())
            self.assertTrue(app.settings_visible)
            error.assert_called_once()

    def test_wheel_on_combo_scrolls_page_once_without_changing_selection(self):
        page = self.app.settings_pages['audio']
        combo = self.app.mode
        combo.current(1)
        changed = []
        combo.bind('<<ComboboxSelected>>', lambda e: changed.append(combo.get()), add='+')
        self.app.settings_tabs.select(page)
        self.root.update()
        page.canvas.yview_moveto(0)
        combo.event_generate('<MouseWheel>', delta=-120)
        self.root.update()
        observed = page.canvas.yview()
        self.assertGreater(observed[0], 0)
        self.assertEqual(combo.current(), 1)
        self.assertEqual(changed, [])
        page.canvas.yview_moveto(0)
        page.wheel(SimpleNamespace(delta=-120))
        self.root.update()
        self.assertEqual(page.canvas.yview(), observed)
        self.app.settings_tabs.select(self.app.settings_pages['appearance'])
        self.root.update()
        size = self.app.font_size.get()
        self.app.font_size_picker.event_generate('<MouseWheel>', delta=120)
        self.assertEqual(self.app.font_size.get(), size)
        # Explicit selection still changes the setting and triggers its callback.
        self.app.font_size_picker.set(14)
        self.app.font_size_picker.event_generate('<<ComboboxSelected>>')
        self.assertEqual(self.app.font_size.get(), 14)

    def test_model_service_dialog_wheel_does_not_change_provider_or_model(self):
        window = show(self.app)
        self.root.update()
        def combos(parent):
            for child in parent.winfo_children():
                if isinstance(child, ttk.Combobox):
                    yield child
                yield from combos(child)
        controls = list(combos(window))
        self.assertEqual(len(controls), 2)
        for combo in controls:
            original = combo.get()
            for delta in (-120, 120):
                combo.event_generate('<MouseWheel>', delta=delta)
                self.root.update()
                self.assertEqual(combo.get(), original)
        self.assertIs(self.root.grab_current(), window)

    def test_balance_display_clears_on_failure_and_ignores_previous_account(self):
        from decimal import Decimal
        from qa_balance import Balance
        app = self.app
        app.qa_settings = {'provider': 'deepseek', 'profiles': {}}
        app.configure_qa_provider()
        revision = app.balance_query.revision
        app.handle_qa_balance((revision, Balance(True, {'CNY': Decimal('25')}), ''))
        self.assertEqual(app.balance_status.get(), '¥25.00')
        self.assertRegex(app.balance_detail.get(), r'^更新于 \d{2}:\d{2}:\d{2}$')
        self.assertFalse(hasattr(app, 'balance_progress'))
        app.handle_qa_balance((revision, None, '网络不可用'))
        self.assertEqual(app.balance_status.get(), '查询失败')
        app.qa_settings['provider'] = 'compatible'
        app.configure_qa_provider()
        app.handle_qa_balance((revision, Balance(True, {'CNY': Decimal('99')}), ''))
        self.assertEqual(app.balance_status.get(), '暂不支持')
        self.assertEqual(str(app.balance_check_button.cget('state')), 'disabled')

    def test_detection_dots_follow_results_and_theme_changes(self):
        indicator = self.app.connection_status_indicator
        for state, expected in (('unknown', '#b45309'), ('authenticated', '#b45309'),
                                ('checking', '#2563eb'), ('verified', '#15803d'),
                                ('unavailable', '#b91c1c'), ('unauthenticated', '#b91c1c')):
            self.app.qa_connection.state = state
            self.app.render_qa_connection()
            self.assertEqual(indicator.dot.cget('fg'), expected)
            self.assertNotIn('●', self.app.connection_status.get())
        for state, expected in (('progress', '#2563eb'), ('success', '#15803d'), ('error', '#b91c1c')):
            self.app.events.put(('gpu_check', (self.app.gpu_check.revision, state, '检测结果')))
            self.app.poll()
            self.assertEqual(self.app.gpu_status_indicator.dot.cget('fg'), expected)
        self.app.qa_connection.state = 'verified'
        self.app.render_qa_connection()
        self.app.dark_mode.set(True)
        self.app.apply_theme()
        self.assertEqual(indicator.dot.cget('fg'), '#4ADE80')
        self.assertEqual(self.app.gpu_status_indicator.dot.cget('fg'), '#F87171')
        self.app.dark_mode.set(False)
        self.app.apply_theme()
        self.assertEqual(indicator.dot.cget('fg'), '#15803d')
        self.assertEqual(self.app.gpu_status_indicator.dot.cget('fg'), '#b91c1c')


if __name__ == '__main__':
    unittest.main()
