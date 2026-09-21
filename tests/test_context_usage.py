import json
import queue
import tempfile
import threading
import tkinter as tk
import unittest
from unittest.mock import patch

import httpx

from context_usage import context_limit, usage_report, usage_label
from context_meter import ContextMeter
from qa_provider import APIProvider
from qa_worker import QAWorker, make_prompt


def chunk(delta=None, finish=None, usage=None, empty=False):
    return 'data: ' + json.dumps({'choices': [] if empty else [
        {'delta': delta or {}, 'finish_reason': finish}], 'usage': usage}) + '\n\n'


PROFILE = dict(base_url='https://api.deepseek.com', model='deepseek-v4-pro', api_key='fake')


class ContextUsageTests(unittest.TestCase):
    def test_capacity_unknown_gateways_and_small_percentages(self):
        self.assertEqual(context_limit('deepseek', PROFILE), 1_000_000)
        self.assertIsNone(context_limit('compatible', PROFILE))
        self.assertIsNone(context_limit('deepseek', dict(PROFILE, base_url='https://example.com')))
        self.assertIsNone(context_limit('deepseek', dict(PROFILE, model='custom')))
        report = usage_report('deepseek', PROFILE, [], usage={'prompt_tokens': 300, 'completion_tokens': 20})
        self.assertEqual(usage_label(report), '<0.1%')
        self.assertEqual(report['used'], 320)
        self.assertFalse(report['estimated'])

    def test_estimate_excludes_image_base64_and_marks_missing_image_tokens(self):
        def estimate(size):
            return usage_report('deepseek', PROFILE, [{'content': [
                {'type': 'text', 'text': '图中问题'}, {'type': 'image_url', 'image_url': {'url': 'x'*size}}]}], image=True)
        self.assertEqual(estimate(100)['used'], estimate(100000)['used'])
        self.assertIn('≈', usage_label(estimate(100)))
        self.assertTrue(estimate(100)['image_unestimated'])
        malformed = usage_report('deepseek', PROFILE, [], usage={'prompt_tokens': -1, 'completion_tokens': 2})
        self.assertTrue(malformed['estimated'])

    def test_final_usage_only_chunk_is_read_and_not_confused_with_cache_hits(self):
        reports = []
        def handler(request):
            return httpx.Response(200, text=chunk({'content': '答案'}, 'stop') +
                chunk(usage={'prompt_tokens': 12000, 'completion_tokens': 300,
                             'prompt_cache_hit_tokens': 11000}, empty=True) + 'data: [DONE]\n\n')
        backend = APIProvider('deepseek', PROFILE, httpx.MockTransport(handler))
        self.assertEqual(backend.run(make_prompt('', '问题'), threading.Event(), usage=reports.append), '答案')
        self.assertTrue(reports[0]['estimated'])
        self.assertFalse(reports[-1]['estimated'])
        self.assertEqual(reports[-1]['used'], 12300)
        self.assertEqual(usage_label(reports[-1]), '1.2%')

    def test_multiple_tool_rounds_replace_usage_instead_of_accumulating(self):
        with tempfile.TemporaryDirectory() as directory:
            requests, reports = [], []
            def handler(request):
                requests.append(json.loads(request.content))
                if len(requests) == 1:
                    return httpx.Response(200, text=chunk({'tool_calls': [{'index': 0, 'id': 'list',
                        'function': {'name': 'list_reference_files', 'arguments': '{}'}}]}, 'tool_calls',
                        {'prompt_tokens': 1000, 'completion_tokens': 50}))
                return httpx.Response(200, text=chunk({'content': '答案'}, 'stop',
                    {'prompt_tokens': 1400, 'completion_tokens': 100}))
            backend = APIProvider('deepseek', PROFILE, httpx.MockTransport(handler),
                                  lambda: {'enabled': True, 'paths': [directory]})
            backend.run(make_prompt('', '读取目录'), threading.Event(), usage=reports.append)
            self.assertEqual(reports[-1]['used'], 1500)
            self.assertEqual(len(reports), 4)

    def test_flash_extraction_usage_does_not_overwrite_pro_context(self):
        reports = []
        def handler(request):
            flash = json.loads(request.content)['model'] == 'deepseek-flash'
            return httpx.Response(200, text=chunk({'content': '27 + 15' if flash else '42'}, 'stop',
                {'prompt_tokens': 30000 if flash else 600, 'completion_tokens': 30}))
        backend = APIProvider('deepseek', PROFILE, httpx.MockTransport(handler))
        self.assertEqual(backend.run(make_prompt('', '回答图片问题'), threading.Event(),
                                     image=b'image', usage=reports.append), '42')
        self.assertEqual(reports[-1]['used'], 630)
        self.assertTrue(all(report['model'] == 'deepseek-v4-pro' for report in reports))

    def test_worker_usage_callback_is_cancelled_with_request(self):
        events, entered, release = queue.Queue(), threading.Event(), threading.Event()
        qa = QAWorker(events)
        qa.track_usage = True
        def run(prompt, cancel, partial, usage):
            usage({'used': 1})
            entered.set()
            release.wait(2)
            usage({'used': 999})
            return '旧回答'
        qa.stream_runner = run
        qa.set_enabled(True)
        qa.ask('问题', [])
        self.assertTrue(entered.wait(2))
        old = qa.generation
        qa.reset()
        release.set()
        reports = []
        while True:
            _, (generation, state, payload) = events.get(timeout=2)
            if state == 'usage':
                reports.append(payload)
                self.assertEqual(generation, old)
            if state == 'done':
                break
        self.assertEqual(reports, [{'used': 1}])

    def test_widget_theme_and_clear(self):
        root = tk.Tk()
        self.addCleanup(root.destroy)
        meter = ContextMeter(root)
        meter.pack()
        meter.set_usage(dict(model='test', used=910, limit=1000, estimated=False))
        root.update()
        self.assertEqual(meter.itemcget('percentage', 'text'), '91%')
        self.assertEqual(meter.itemcget('percentage', 'state'), 'hidden')
        meter.event_generate('<Enter>')
        root.update()
        self.assertEqual(meter.itemcget('percentage', 'state'), 'normal')
        self.assertEqual(meter.itemcget('percentage', 'fill'), '#b91c1c')
        meter.apply_theme(True)
        self.assertEqual(meter.itemcget('percentage', 'fill'), '#F87171')
        meter.set_usage()
        self.assertEqual(meter.itemcget('percentage', 'text'), '—')
        meter.event_generate('<Leave>')
        root.update()
        self.assertEqual(meter.itemcget('percentage', 'state'), 'hidden')

    def test_app_stale_events_clear_and_collapsed_layout(self):
        from app import App
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('app.ClipboardWatcher'), \
                patch('qa_connection.QAConnection.check'), patch('app.save_desktop'):
            root = tk.Tk()
            app = App(root)
            try:
                root.geometry('560x370')
                root.update()
                report = dict(model='test', used=1200, limit=10000, estimated=False)
                app.handle_qa((app.qa.generation, 'usage', report))
                self.assertEqual(app.context_meter.report, report)
                app.handle_qa((app.qa.generation-1, 'usage', dict(report, used=9999)))
                self.assertEqual(app.context_meter.report, report)
                app.composer.collapse()
                root.update()
                self.assertFalse(app.context_meter.winfo_ismapped())
                app.composer.expand()
                root.update()
                self.assertTrue(app.context_meter.winfo_viewable())
                self.assertLess(app.context_meter.winfo_x(), app.ask_selected_button.winfo_x())
                meter = app.context_meter
                ring_x = meter.winfo_rootx() + 14
                send_x = app.ask_selected_button.winfo_rootx()
                meter.set_hovered(True)
                root.update()
                self.assertEqual(meter.winfo_rootx() + 66, ring_x)
                self.assertEqual(app.ask_selected_button.winfo_rootx(), send_x)
                self.assertEqual(app.composer.actions.winfo_x(), 4)
                app.clear()
                self.assertIsNone(app.context_meter.report)
            finally:
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                app.close()
