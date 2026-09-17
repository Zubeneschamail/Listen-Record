import asyncio
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest

import httpx
from codex_qa import CodexQA, make_prompt
from qa_provider import APIProvider, endpoint, load_settings, save_settings


def event(content='', finish=None):
    return 'data: ' + json.dumps({'choices': [{'delta': {'content': content}, 'finish_reason': finish}]}) + '\n\n'


class ProviderTests(unittest.TestCase):
    def backend(self, handler, provider='deepseek'):
        return APIProvider(provider, {'base_url': 'https://example.com/v1', 'model': 'test-model',
                                     'api_key': 'secret-test-key'}, httpx.MockTransport(handler))

    def test_stream_payload_and_incremental_answer(self):
        requests, partials = [], []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, text=': heartbeat\n\n' + event('你') + event('好', 'stop') + 'data: [DONE]\n\n')
        answer = self.backend(handler).run(make_prompt('context', 'question'), threading.Event(), partials.append)
        self.assertEqual(answer, '你好')
        self.assertEqual(partials[0], '你')
        self.assertEqual(str(requests[0].url), 'https://example.com/v1/chat/completions')
        body = json.loads(requests[0].content)
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertTrue(body['stream'])
        self.assertEqual(body['messages'][0]['role'], 'system')
        self.assertEqual(json.loads(body['messages'][1]['content'])['当前问题'], 'question')

    def test_custom_provider_does_not_send_deepseek_parameters(self):
        def handler(request):
            self.assertNotIn('thinking', json.loads(request.content))
            return httpx.Response(200, text=event('OK', 'stop'))
        self.assertEqual(self.backend(handler, 'compatible').run('probe', threading.Event()), 'OK')

    def test_http_errors_do_not_expose_response_or_key_and_no_redirect(self):
        for code in (401, 402, 404, 429, 500, 307):
            requests = []
            def handler(request):
                requests.append(request)
                return httpx.Response(code, text='secret-test-key', headers={'Location': 'https://other.example'})
            with self.assertRaises(RuntimeError) as error:
                self.backend(handler).run('probe', threading.Event())
            self.assertNotIn('secret-test-key', str(error.exception))
            self.assertEqual(len(requests), 1)

    def test_interrupted_empty_malformed_and_truncated_streams_fail(self):
        for stream in (event('partial'), 'data: [DONE]\n\n', 'data: broken\n\n', event('partial', 'length')):
            with self.assertRaises(RuntimeError):
                self.backend(lambda r: httpx.Response(200, text=stream)).run('probe', threading.Event())

    def test_cancellation_and_total_timeout_close_pending_transport(self):
        for should_cancel in (True, False):
            closed = threading.Event()
            async def handler(request):
                try:
                    await asyncio.sleep(10)
                finally:
                    closed.set()
            cancel = threading.Event()
            timer = threading.Timer(0.15, cancel.set)
            if should_cancel:
                timer.start()
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, '已取消' if should_cancel else '超时'):
                self.backend(handler).run('probe', cancel, timeout=2 if should_cancel else 0.2)
            self.assertLess(time.monotonic() - start, 1.5)
            self.assertTrue(closed.is_set())
            timer.cancel()

    def test_endpoint_validation(self):
        self.assertEqual(endpoint('https://example.com/v1/'), 'https://example.com/v1/chat/completions')
        self.assertEqual(endpoint('http://localhost:8000/v1'), 'http://localhost:8000/v1/chat/completions')
        for url in ('http://example.com', 'https://user:key@example.com', 'https://example.com?a=key', ''):
            with self.assertRaises(ValueError):
                endpoint(url)

    def test_credentials_roundtrip_encrypted_and_profiles_are_separate(self):
        settings = {'provider': 'deepseek', 'profiles': {
            'deepseek': {'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash', 'api_key': 'secret-one'},
            'compatible': {'base_url': 'https://example.com/v1', 'model': 'custom', 'api_key': 'secret-two'}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            save_settings(settings, path)
            self.assertNotIn('secret-', path.read_text(encoding='utf-8'))
            self.assertEqual(load_settings(path), settings)
            data = json.loads(path.read_text(encoding='utf-8'))
            data['profiles']['deepseek']['encrypted_key'] = 'invalid'
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertEqual(load_settings(path)['profiles']['deepseek']['api_key'], '')

    def test_stream_worker_emits_partial_and_discards_cancelled_completion(self):
        events = queue.Queue()
        qa = CodexQA(events)
        started, release = threading.Event(), threading.Event()
        def stream(prompt, cancel, partial):
            partial('draft')
            started.set()
            release.wait(2)
            partial('stale draft')
            return 'stale answer'
        qa.stream_runner = stream
        qa.set_enabled(True)
        qa.ask('question', [])
        self.assertTrue(started.wait(1))
        first = [events.get_nowait()[1][1] for _ in range(events.qsize())]
        self.assertIn('partial', first)
        qa.reset()
        release.set()
        item = events.get(timeout=2)
        self.assertEqual(item[1][1], 'done')
        self.assertTrue(events.empty())
