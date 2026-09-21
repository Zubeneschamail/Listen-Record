import json
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from qa_provider import APIProvider, DEFAULTS
from startup_checks import StartupChecks, check_local


class LocalChecksTests(unittest.TestCase):
    def run_local(self, cached='cached', errors=(), models=None):
        results = {}
        with patch('model_download.cached_model', return_value=cached), \
                patch('model_runtime.prepare_cuda'), patch('model_runtime.cuda_runtime_errors', return_value=errors), \
                patch('ctranslate2.get_cuda_device_count', return_value=1), \
                patch('model_runtime.load_model', side_effect=models or [SimpleNamespace(model=SimpleNamespace(device='cuda'))]) as load:
            check_local('small', lambda key, state, detail: results.update({key:(state, detail)}))
        return results, load

    def test_missing_model_is_actionable_without_claiming_gpu_success(self):
        results, load = self.run_local(cached=None)
        self.assertEqual(results['model'][0], 'error')
        self.assertEqual(results['gpu'][0], 'blocked')
        load.assert_not_called()

    def test_gpu_success_validates_both_with_one_model_load(self):
        results, load = self.run_local()
        self.assertEqual([value[0] for value in results.values()], ['success', 'success'])
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.kwargs['device'], 'cuda')

    def test_gpu_failure_still_checks_cpu_model_and_model_corruption(self):
        results, load = self.run_local(errors=['cublas64_12.dll：missing'])
        self.assertEqual(results['gpu'][0], 'error')
        self.assertEqual(results['model'][0], 'success')
        self.assertEqual(load.call_args.kwargs['device'], 'cpu')
        results, _ = self.run_local(models=[RuntimeError('GPU fail'), RuntimeError('corrupt model')])
        self.assertEqual(results['model'][0], 'error')
        self.assertIn('corrupt model', results['model'][1])


class StartupSchedulerTests(unittest.TestCase):
    def setup_check(self):
        context = Mock()
        receiver, sender = Mock(), Mock()
        context.Pipe.return_value = receiver, sender
        receiver.poll.return_value = False
        context.Process.return_value.is_alive.return_value = True
        clock = [0]
        return StartupChecks(context, clock=lambda:clock[0]), context, receiver, clock

    def test_network_timeout_and_local_crash_finish_without_hanging(self):
        check, context, receiver, clock = self.setup_check()
        release = threading.Event()
        with patch('qa_provider.APIProvider.check_connection', side_effect=lambda *a, **k: release.wait(2)):
            try:
                check.start('small', DEFAULTS['deepseek'])
                clock[0] = 26
                check.poll()
                self.assertEqual(check.results['deepseek'][0], 'error')
                context.Process.return_value.is_alive.return_value = False
                check.poll()
                self.assertFalse(check.active)
                self.assertEqual(check.results['model'][0], 'error')
            finally:
                release.set()
                check.close()

    def test_local_timeout_and_successful_network_do_not_overwrite_each_other(self):
        check, context, receiver, clock = self.setup_check()
        with patch('qa_provider.APIProvider.check_connection', return_value='OK'):
            check.start('small', dict(DEFAULTS['deepseek']))
            deadline = time.monotonic() + 2
            while check.messages.empty() and time.monotonic() < deadline:
                time.sleep(.01)
            clock[0] = 121
            check.poll()
            self.assertFalse(check.active)
            self.assertEqual(check.results['deepseek'][0], 'success')
            self.assertEqual(check.results['model'][0], 'error')
            context.Process.return_value.terminate.assert_called_once()

    def test_connection_probe_has_small_budget_and_no_user_data_or_tools(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, text='data: '+json.dumps({'choices':[{'delta':{'content':'OK'},'finish_reason':'stop'}]})+'\n\n')
        backend = APIProvider('deepseek', dict(DEFAULTS['deepseek'], api_key='fake'),
                              httpx.MockTransport(handle), workspace_loader=Mock(side_effect=AssertionError('No reference reads')))
        self.assertEqual(backend.check_connection(threading.Event()), 'OK')
        self.assertEqual(requests[0]['max_tokens'], 8)
        self.assertEqual(requests[0]['messages'], [{'role':'user', 'content':'只回复 OK。'}])
        self.assertNotIn('tools', requests[0])


class StartupUITests(unittest.TestCase):
    def setUp(self):
        from app import App
        self.patches = [patch('app.GlobalHotkey'), patch('app.App.start_tray'),
                        patch('qa_connection.QAConnection.check'), patch('app.save_desktop'),
                        patch('app.App.save_desktop_settings')]
        for item in self.patches:
            item.start()
        self.root = tk.Tk()
        self.app = App(self.root)
        self.root.update()

    def tearDown(self):
        if self.app.startup_checks:
            self.app.startup_checks.close()
        if self.app.startup_overlay:
            self.app.startup_overlay.destroy()
        self.app.gpu_check.close()
        self.app.qa.set_enabled(False)
        self.app.qa_connection.close()
        self.app.hotkey.close()
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.root.destroy()
        for item in reversed(self.patches):
            item.stop()

    def start_with(self, results):
        fake = Mock()
        fake.active = True
        fake.results = results
        with patch('startup_checks.StartupChecks', return_value=fake):
            self.app.begin_startup_checks()
        return fake

    def test_overlay_covers_window_blocks_recording_and_success_cleans_up(self):
        fake = self.start_with({key:('success','正常') for key in ('model','gpu','deepseek')})
        self.root.update()
        overlay = self.app.startup_overlay
        self.assertEqual(overlay.winfo_width(), self.root.winfo_width())
        self.assertEqual(overlay.winfo_height(), self.root.winfo_height())
        self.assertIs(self.root.grab_current(), overlay)
        self.assertIsNotNone(overlay.timer)
        with patch.object(self.app.engine, 'start') as start:
            self.app.start()
            start.assert_not_called()
        self.root.geometry('700x460')
        self.root.update()
        self.assertEqual(overlay.winfo_width(), 700)
        fake.active = False
        self.app.poll()
        self.assertIsNone(self.app.startup_overlay)
        self.assertIsNone(self.root.grab_current())
        self.assertFalse(self.app.settings_visible)

    def test_errors_open_model_gpu_then_deepseek_without_changing_provider(self):
        self.app.qa_settings['provider'] = 'compatible'
        fake = self.start_with({key:('error',key+' unavailable') for key in ('model','gpu','deepseek')})
        with patch('desktop_dialogs.model_manager') as manager, patch('qa_settings_dialog.show') as show:
            fake.active = False
            self.app.poll()
            self.assertEqual(manager.call_args.args[0], self.app)
            manager.call_args.kwargs['on_close']()
            self.assertTrue(self.app.startup_advance_on_hide)
            self.assertEqual(self.app.settings_tabs.select(), str(self.app.settings_pages['audio']))
            self.app.hide_settings()
            self.root.update()
            self.assertEqual(show.call_args.kwargs['initial_provider'], 'deepseek')
            self.assertEqual(self.app.qa_settings['provider'], 'compatible')
            self.assertEqual(self.app.startup_pending_issues, [])

    def test_loading_surface_and_children_drag_without_interrupting_checks(self):
        fake = self.start_with({key:('checking','等待') for key in ('model','gpu','deepseek')})
        self.root.geometry('560x380+100+100')
        self.root.update()
        overlay = self.app.startup_overlay
        surfaces = [overlay]
        for widget in surfaces:
            surfaces.extend(child for child in widget.winfo_children() if not isinstance(child, tk.Button))
        for widget in surfaces:
            with self.subTest(widget=widget.winfo_class()):
                self.root.geometry('560x380+100+100')
                self.root.update()
                widget.event_generate('<ButtonPress-1>', x=5, y=5, rootx=150, rooty=150)
                widget.event_generate('<B1-Motion>', x=25, y=20, rootx=170, rooty=165)
                widget.event_generate('<ButtonRelease-1>', x=25, y=20, rootx=170, rooty=165)
                self.root.update()
                self.assertEqual((self.root.winfo_rootx(), self.root.winfo_rooty()), (120, 115))
                self.assertIsNone(self.app._drag_origin)
                self.assertIs(self.root.grab_current(), overlay)
                self.assertIsNotNone(overlay.timer)
                self.assertTrue(fake.active)
        close = next(child for child in overlay.winfo_children() if isinstance(child, tk.Button))
        self.assertFalse(close.bind('<ButtonPress-1>'))
        self.assertFalse(close.bind('<B1-Motion>'))

    def test_deepseek_only_failure_automatically_opens_its_configuration(self):
        fake = self.start_with({'model':('success','OK'),'gpu':('success','OK'),'deepseek':('error','API Key 无效')})
        fake.active = False
        self.app.poll()
        self.root.update()
        window = self.app.qa_settings_window
        self.assertTrue(window.winfo_viewable())
        self.assertIs(self.root.grab_current(), window)
        self.assertEqual(self.app.settings_tabs.select(), str(self.app.settings_pages['ai']))

    def test_close_cancels_workers_and_animation(self):
        fake = self.start_with({key:('checking','等待') for key in ('model','gpu','deepseek')})
        with patch.object(self.root, 'destroy'):
            self.app.close()
        fake.close.assert_called_once()
        self.assertIsNone(self.app.startup_overlay)
