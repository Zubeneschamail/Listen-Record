import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from qa_provider import APIProvider
from qa_worker import make_prompt
from reference_files import ReferenceCache, ReferenceTools
from reference_policy import reference_partial


def chunk(text, finish=None):
    return ('data: ' + json.dumps({'choices': [
        {'delta': {'content': text}, 'finish_reason': finish}]}) + '\n\n').encode()


class ReferencePerformanceTests(unittest.TestCase):
    def test_provider_reuses_text_across_requests_but_refreshes_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'sample.txt'
            file.write_text('first contents', encoding='utf-8')
            reads, results = [], []
            original = Path.read_bytes
            def read(path):
                reads.append(path)
                return original(path)
            def handler(request):
                messages = json.loads(request.content)['messages']
                if messages[-1]['role'] == 'user':
                    return httpx.Response(200, text='data: ' + json.dumps({'choices': [{
                        'delta': {'tool_calls': [{'index': 0, 'id': 'read', 'function': {
                            'name': 'read_reference_file', 'arguments': '{"path":"r1/sample.txt"}'}}]},
                        'finish_reason': 'tool_calls'}]})+'\n\n')
                results.append(messages[-1]['content'])
                return httpx.Response(200, content=chunk('已完成读取。', 'stop'))
            settings = {'enabled': True, 'paths': [directory]}
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                                  httpx.MockTransport(handler), lambda: settings)
            with patch.object(Path, 'read_bytes', read):
                for _ in range(2):
                    backend.run(make_prompt('', '读取 sample.txt'), threading.Event())
                self.assertEqual(reads, [file])
                file.write_text('changed contents and new facts', encoding='utf-8')
                backend.run(make_prompt('', '读取 sample.txt'), threading.Event())
                self.assertEqual(reads, [file, file])
            self.assertEqual(results[0], results[1])
            self.assertIn('changed contents', results[-1])
            settings['enabled'] = False
            denied = ReferenceTools(settings, threading.Event(), cache=backend.reference_cache)
            self.assertIn('error', json.loads(denied.execute('read_reference_file', '{"path":"r1/sample.txt"}')))

    def test_cache_lru_memory_bound_and_new_version_replaces_old(self):
        cache = ReferenceCache(max_bytes=300, max_files=2)
        cache.put(('a', 1), ['abc'])
        cache.put(('b', 1), ['abc'])
        self.assertEqual(cache.get(('a', 1)), ['abc'])
        cache.put(('c', 1), ['abc'])
        self.assertIsNone(cache.get(('b', 1)))
        cache.put(('a', 2), ['new'])
        self.assertIsNone(cache.get(('a', 1)))
        cache.put(('big', 1), ['x'*1000])
        self.assertIsNone(cache.get(('big', 1)))
        self.assertLessEqual(cache.size, 300)

    def test_listing_stops_after_one_page_and_lookahead(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = ReferenceTools({'enabled': True, 'paths': [directory]}, threading.Event())
            scanned = []
            def files(scope=''):
                for i in range(500):
                    scanned.append(i)
                    yield f'r1/{i:04}.txt', Path(directory)/f'{i:04}.txt'
            with patch.object(tools, 'files', files):
                first = json.loads(tools.execute('list_reference_files', '{}'))
                self.assertEqual(len(scanned), 61)
                self.assertEqual(first['next_offset'], 60)
                scanned.clear()
                second = json.loads(tools.execute('list_reference_files', '{"offset":60}'))
                self.assertEqual(len(scanned), 121)
                self.assertFalse(set(first['files']) & set(second['files']))
                last = json.loads(tools.execute('list_reference_files', '{"offset":480}'))
                self.assertEqual(len(last['files']), 20)
                self.assertIsNone(last['next_offset'])

    def test_reference_answer_is_visible_before_stream_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            received, release, finished = threading.Event(), threading.Event(), threading.Event()
            errors, partials = [], []
            class Stream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield chunk('二分查找每次将搜索范围缩小一半。')
                    while not release.is_set():
                        await asyncio.sleep(.01)
                    yield chunk('时间复杂度为 O(log n)。', 'stop')
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream())),
                lambda: {'enabled': True, 'paths': [directory]})
            def partial(text):
                partials.append(text)
                received.set()
            def run():
                try:
                    backend.run(make_prompt('', '解释二分查找'), threading.Event(), partial)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    finished.set()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                self.assertTrue(received.wait(2), 'Answer stayed buffered until completion')
                self.assertFalse(finished.is_set())
                self.assertIn('二分查找', partials[0])
            finally:
                release.set()
                thread.join(3)
            self.assertFalse(errors)
            self.assertTrue(finished.is_set())

    def test_fragmented_read_promises_and_protocol_stay_hidden(self):
        for text in ('让我现在读取参考资料，稍等。', '<｜DSML｜invoke name="read_reference_file">',
                     '我现在搜索这个文件，然后给出分析。'):
            for end in range(1, len(text)+1):
                self.assertEqual(reference_partial(text[:end]), '')
        self.assertEqual(reference_partial('先缩小搜索范围，再检查边界条件。'), '先缩小搜索范围，再检查边界条件。')
