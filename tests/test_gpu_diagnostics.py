import queue
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from gpu_diagnostics import GPUCheck, probe_gpu
from model_runtime import cuda_runtime_errors, load_model


class ProbeTests(unittest.TestCase):
    def test_library_failures_include_each_name_and_do_not_cache_partial_success(self):
        with patch('model_runtime._CUDA_LIBRARIES', []) as libraries, \
                patch('ctypes.WinDLL', side_effect=[object(), OSError('dependency missing'), OSError('wrong architecture')]):
            errors = cuda_runtime_errors()
            self.assertEqual(len(errors), 2)
            self.assertIn('cublas64_12.dll', errors[0])
            self.assertIn('cudnn64_9.dll', errors[1])
            self.assertEqual(libraries, [])

    def test_missing_gpu_or_runtime_or_model_never_claims_success(self):
        for count, errors, cached, expected in (
                (0, [], 'cached', '显卡'),
                (1, ['cudnn64_9.dll: failed'], 'cached', 'cudnn64_9.dll'),
                (1, [], None, '尚未下载')):
            with self.subTest(expected=expected), patch('model_runtime.prepare_cuda'), \
                    patch('ctranslate2.get_cuda_device_count', return_value=count), \
                    patch('model_runtime.cuda_runtime_errors', return_value=errors), \
                    patch('model_download.cached_model', return_value=cached), \
                    patch('model_runtime.load_model') as load:
                with self.assertRaisesRegex(RuntimeError, expected):
                    probe_gpu('small', Mock())
                load.assert_not_called()

    def test_real_loader_must_exhaust_gpu_inference_before_success(self):
        called = []
        def generate():
            called.append('inference')
            yield SimpleNamespace(text='')
        model = Mock()
        model.model.device = 'cuda'
        model.transcribe.return_value = (generate(), None)
        with patch('model_runtime.prepare_cuda'), patch('ctranslate2.get_cuda_device_count', return_value=1), \
                patch('model_runtime.cuda_runtime_errors', return_value=[]), \
                patch('model_download.cached_model', return_value='cached'), \
                patch('model_runtime.WhisperModel', return_value=model) as create:
            self.assertIn('GPU 上完成', probe_gpu('small', Mock()))
            self.assertEqual(called, ['inference'])
            self.assertEqual(create.call_args.kwargs['device'], 'cuda')

    def test_explicit_gpu_failure_does_not_fall_back_to_cpu(self):
        def fail():
            raise RuntimeError('out of memory')
            yield
        model = Mock()
        model.transcribe.return_value = (fail(), None)
        with patch('model_runtime.prepare_cuda'), patch('ctranslate2.get_cuda_device_count', return_value=1), \
                patch('model_runtime.cuda_runtime_errors', return_value=[]), \
                patch('model_download.cached_model', return_value='cached'), \
                patch('model_runtime.WhisperModel', return_value=model) as create:
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                load_model('small', device='cuda')
            self.assertEqual(create.call_count, 1)

    def test_accidental_cpu_result_is_rejected(self):
        with patch('model_runtime.prepare_cuda'), patch('ctranslate2.get_cuda_device_count', return_value=1), \
                patch('model_runtime.cuda_runtime_errors', return_value=[]), \
                patch('model_download.cached_model', return_value='cached'), \
                patch('model_runtime.load_model', return_value=SimpleNamespace(model=SimpleNamespace(device='cpu'))):
            with self.assertRaisesRegex(RuntimeError, '未使用 CUDA'):
                probe_gpu('small', Mock())


class CheckTests(unittest.TestCase):
    def make_check(self):
        context = Mock()
        receiver, sender = Mock(), Mock()
        context.Pipe.return_value = (receiver, sender)
        receiver.poll.return_value = False
        context.Process.return_value.is_alive.return_value = True
        clock = [0]
        check = GPUCheck(queue.Queue(), context=context, clock=lambda: clock[0])
        check.start('small')
        return check, context, receiver, clock

    def test_timeout_stops_only_probe_and_invalidates_old_messages(self):
        check, context, receiver, clock = self.make_check()
        old_revision = check.revision
        check.start('small')
        self.assertEqual(context.Process.call_count, 1)
        clock[0] = 121
        check.poll()
        context.Process.return_value.terminate.assert_called_once()
        self.assertFalse(check.active)
        self.assertNotEqual(old_revision, check.revision)
        self.assertIn('超时', list(check.events.queue)[-1][1][-1])
        receiver.close.assert_called_once()

    def test_success_crash_and_start_failure(self):
        check, context, receiver, _ = self.make_check()
        receiver.poll.return_value = True
        receiver.recv.return_value = ('success', 'GPU OK')
        check.poll()
        self.assertFalse(check.active)
        self.assertEqual(list(check.events.queue)[-1][1][1:], ('success', 'GPU OK'))
        check, context, receiver, _ = self.make_check()
        context.Process.return_value.is_alive.return_value = False
        context.Process.return_value.exitcode = -1
        check.poll()
        self.assertIn('异常退出', list(check.events.queue)[-1][1][-1])
        context.Process.return_value.start.side_effect = OSError('blocked')
        check.start('small')
        self.assertFalse(check.active)
        self.assertIn('启动失败', list(check.events.queue)[-1][1][-1])


class GPUSettingsTests(unittest.TestCase):
    def test_button_and_runtime_result_are_independent_and_busy_is_guarded(self):
        from app import App
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'):
            root = tk.Tk()
            app = App(root)
        try:
            with patch.object(app.gpu_check, 'start') as start:
                app.busy = True
                app.gpu_check_button.invoke()
                start.assert_not_called()
                self.assertIn('停止转写', app.gpu_check_detail.get())
                app.busy = False
                app.gpu_check_button.invoke()
                start.assert_called_once_with(app.model.get().split()[0])
            app.events.put(('recognition_backend', ('small', 'cpu')))
            app.events.put(('gpu_check', (app.gpu_check.revision, 'success', '测试通过')))
            app.poll()
            self.assertIn('CPU', app.gpu_runtime_status.get())
            self.assertEqual(app.gpu_check_status.get(), 'GPU 检测通过')
            self.assertEqual(str(app.gpu_check_button.cget('state')), 'normal')
            app.events.put(('gpu_check', (app.gpu_check.revision - 1, 'error', '过期消息')))
            app.poll()
            self.assertEqual(app.gpu_check_status.get(), 'GPU 检测通过')
        finally:
            app.gpu_check.close()
            app.qa.set_enabled(False)
            app.qa_connection.close()
            app.hotkey.close()
            for timer in root.tk.call('after', 'info'):
                root.after_cancel(timer)
            root.destroy()
