import asyncio
import base64
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from qa_worker import QAWorker, make_prompt
from qa_provider import APIProvider


def response(text, finish='stop'):
    return httpx.Response(200, text='data: ' + json.dumps({'choices': [
        {'delta': {'content': text}, 'finish_reason': finish}]}) + '\n\n')


class VisionPipelineTests(unittest.TestCase):
    def backend(self, handler, workspace_loader=None):
        return APIProvider('deepseek', dict(base_url='https://example.com/v1',
            model='deepseek-v4-pro', api_key='fake-key'), httpx.MockTransport(handler), workspace_loader)

    def test_flash_receives_image_pro_receives_extraction_and_only_final_answer_streams(self):
        bodies, stages, partials = [], [], []
        def handler(request):
            self.assertEqual(str(request.url), 'https://example.com/v1/chat/completions')
            self.assertEqual(request.headers['Authorization'], 'Bearer fake-key')
            body = json.loads(request.content)
            bodies.append(body)
            if len(bodies) == 1:
                self.assertEqual(body['model'], 'deepseek-flash')
                self.assertNotIn('tools', body)
                self.assertEqual(body['max_tokens'], 4096)
                content = body['messages'][-1]['content']
                self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',')[1]), b'image-bytes')
                self.assertIn('不补写看不清', body['messages'][0]['content'])
                return response('原文：27 + 15 = ?；不确定之处：无。')
            self.assertEqual(body['model'], 'deepseek-v4-pro')
            self.assertIsInstance(body['messages'][-1]['content'], str)
            data = json.loads(body['messages'][-1]['content'])
            self.assertEqual(json.loads(data['原始请求'])['当前问题'], '计算图片中的题目')
            self.assertIn('27 + 15', data['图片识别结果'])
            self.assertNotIn('image_url', json.dumps(body))
            self.assertIn('不可信参考数据', body['messages'][0]['content'])
            return response('42')
        backend = self.backend(handler)
        result = backend.run(make_prompt('', '计算图片中的题目'), threading.Event(),
            partials.append, image=b'image-bytes', phase=stages.append)
        self.assertEqual(result, '42')
        self.assertEqual(partials, ['42'])
        self.assertEqual(stages, ['Flash 正在读图…', '读图完成，Pro 正在分析…'])
        self.assertEqual(backend.profile['model'], 'deepseek-v4-pro')
        self.assertEqual(len(bodies), 2)

    def test_text_and_probe_use_pro_once(self):
        bodies = []
        def handler(request):
            bodies.append(json.loads(request.content))
            return response('文字答案')
        stages = []
        self.assertEqual(self.backend(handler).run('文字问题', threading.Event(), phase=stages.append), '文字答案')
        self.assertEqual([b['model'] for b in bodies], ['deepseek-v4-pro'])
        self.assertFalse(stages)

    def test_extraction_failures_do_not_send_partial_or_start_pro(self):
        for bad in (httpx.Response(401, text='secret'), response('半段文字', 'length'),
                    response(''), httpx.Response(200, text='data: [DONE]\n\n')):
            with self.subTest(status=bad.status_code, body=bad.text):
                requests, partials = [], []
                def handler(request):
                    requests.append(request)
                    return bad
                with self.assertRaisesRegex(RuntimeError, 'Flash 读图失败'):
                    self.backend(handler).run('图片问题', threading.Event(), partials.append, image=b'image')
                self.assertEqual(len(requests), 1)
                self.assertFalse(partials)

    def test_pro_failure_is_labeled_as_analysis_failure(self):
        requests = []
        def handler(request):
            requests.append(request)
            return response('识别文字') if len(requests) == 1 else httpx.Response(429)
        with self.assertRaisesRegex(RuntimeError, 'Pro 分析失败'):
            self.backend(handler).run('问题', threading.Event(), image=b'image')
        self.assertEqual(len(requests), 2)

    def test_cancel_between_stages_prevents_second_request(self):
        cancel, requests = threading.Event(), []
        def handler(request):
            requests.append(request)
            return response('识别文字')
        def phase(message):
            if 'Pro' in message:
                cancel.set()
        with self.assertRaisesRegex(RuntimeError, '已取消'):
            self.backend(handler).run('问题', cancel, image=b'image', phase=phase)
        self.assertEqual(len(requests), 1)

    def test_timeout_budget_is_shared_and_expired_flash_never_starts_pro(self):
        for flash_seconds in (8, 31):
            now, calls = [0], []
            def request(backend, *args, **kwargs):
                calls.append((backend.profile['model'], kwargs['timeout']))
                now[0] += flash_seconds
                return '识别文字' if len(calls) == 1 else '答案'
            with patch('qa_provider.time.monotonic', side_effect=lambda: now[0]), \
                    patch.object(APIProvider, '_run_request', new=request):
                if flash_seconds > 30:
                    with self.assertRaisesRegex(RuntimeError, '超时'):
                        self.backend(None).run('问题', threading.Event(), timeout=30, image=b'image')
                    self.assertEqual(len(calls), 1)
                else:
                    self.assertEqual(self.backend(None).run('问题', threading.Event(), timeout=30, image=b'image'), '答案')
                    self.assertEqual(calls, [('deepseek-flash', 30), ('deepseek-v4-pro', 22)])

    def test_cancelling_during_either_request_closes_the_transport(self):
        for blocked_model in ('deepseek-flash', 'deepseek-v4-pro'):
            cancel, closed, started = threading.Event(), threading.Event(), threading.Event()
            models = []
            async def handler(request):
                model = json.loads(request.content)['model']
                models.append(model)
                if model != blocked_model:
                    return response('识别文字')
                started.set()
                try:
                    await asyncio.sleep(10)
                finally:
                    closed.set()
            def stop():
                if started.wait(2):
                    cancel.set()
            worker = threading.Thread(target=stop)
            worker.start()
            begin = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, '已取消'):
                self.backend(handler).run('问题', cancel, image=b'image')
            worker.join(2)
            self.assertTrue(closed.is_set())
            self.assertLess(time.monotonic()-begin, 2)
            self.assertEqual(models[-1], blocked_model)

    def test_reference_tools_are_available_only_to_pro(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'reference.txt'
            file.write_text('对照数据', encoding='utf-8')
            loads, bodies = [], []
            def load():
                loads.append(True)
                return {'enabled': True, 'paths': [str(file)]}
            def handler(request):
                body = json.loads(request.content)
                bodies.append(body)
                if body['model'] == 'deepseek-flash':
                    self.assertFalse(loads)
                    self.assertNotIn('tools', body)
                    return response('识别文字')
                self.assertIn('tools', body)
                return response('对照分析')
            self.assertEqual(self.backend(handler, load).run('问题', threading.Event(), image=b'image'), '对照分析')
            self.assertEqual(len(loads), 1)

    def test_worker_emits_image_context_separately_from_displayed_answer(self):
        events = queue.Queue()
        def handler(request):
            return response('图片原文') if json.loads(request.content)['model'] == 'deepseek-flash' else response('最终答案')
        qa = QAWorker(events)
        qa.stream_runner = self.backend(handler).run
        qa.set_enabled(True)
        qa.ask('图片问题', [], image=b'image')
        received = []
        while True:
            value = events.get(timeout=3)[1]
            received.append(value)
            if value[1] == 'done':
                break
        self.assertEqual([v[2] for v in received if v[1] == 'phase'],
            ['Flash 正在读图…', '读图完成，Pro 正在分析…'])
        self.assertEqual([v[2] for v in received if v[1] == 'answer'], [('图片问题', '最终答案')])
        self.assertEqual([v[2] for v in received if v[1] == 'image_context'], ['图片原文'])
        self.assertNotIn('图片原文', str([v for v in received if v[1] in ('answer', 'partial')]))
