import json
from pathlib import Path
import tempfile
import threading
import unittest
import httpx
from qa_worker import make_prompt
from qa_provider import APIProvider


def chunk(delta, finish=None):
    return 'data: '+json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}]})+'\n\n'


class ReferenceProviderTests(unittest.TestCase):
    def test_streamed_tools_read_file_then_answer_with_citation(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'spec.txt'
            file.write_text('验收精度 0.25 毫米', encoding='utf-8')
            requests, partials = [], []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) == 1:
                    self.assertEqual(len(body['tools']), 3)
                    return httpx.Response(200, text=chunk({'tool_calls': [{'index': 0, 'id': 'read1', 'type': 'function',
                        'function': {'name': 'read_reference_file', 'arguments': '{"path":'}}]})+
                        chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '"r1/spec.txt"}'}}]}, 'tool_calls'))
                self.assertEqual(body['messages'][-1]['role'], 'tool')
                self.assertIn('0.25', body['messages'][-1]['content'])
                self.assertEqual(body['messages'][-1]['tool_call_id'], 'read1')
                return httpx.Response(200, text=chunk({'content': '精度为 0.25 毫米 [r1/spec.txt:1]。'}, 'stop'))
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [str(file)]})
            result = backend.run(make_prompt('', '精度是多少？'), threading.Event(), partials.append)
            self.assertIn('0.25', result)
            self.assertIn('r1/spec.txt:1', result)
            self.assertEqual(len(requests), 2)
            self.assertIn('正在查阅参考资料…', partials)
            self.assertNotIn(str(file), json.dumps(requests, ensure_ascii=False))

    def test_probe_and_disabled_references_never_expose_tools_or_paths(self):
        for enabled, probe in ((True, True), (False, False)):
            def handler(request):
                self.assertNotIn('tools', json.loads(request.content))
                return httpx.Response(200, text=chunk({'content': 'OK'}, 'stop'))
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': enabled, 'paths': []})
            self.assertEqual(backend.run('probe', threading.Event(), use_references=not probe), 'OK')

    def test_lookup_rounds_are_bounded_and_cancelled_tools_are_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            counter = []
            def handler(request):
                body = json.loads(request.content)
                counter.append(body)
                if body.get('tool_choice') == 'none':
                    return httpx.Response(200, text=chunk({'content': '根据已有资料无法确定。'}, 'stop'))
                return httpx.Response(200, text=chunk({'tool_calls': [{'index': 0, 'id': f'c{len(counter)}',
                    'function': {'name': 'list_reference_files', 'arguments': '{}'}}]}, 'tool_calls'))
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [directory]})
            self.assertIn('无法确定', backend.run('test', threading.Event()))
            self.assertEqual(len(counter), 7)
            cancelled = threading.Event()
            cancelled.set()
            counter.clear()
            with self.assertRaisesRegex(RuntimeError, '已取消'):
                backend.run('test', cancelled)
            self.assertFalse(counter)

    def test_outside_read_failure_is_returned_to_model_without_file_data(self):
        with tempfile.TemporaryDirectory() as directory:
            count = []
            def handler(request):
                body = json.loads(request.content)
                count.append(1)
                if len(count) == 1:
                    return httpx.Response(200, text=chunk({'tool_calls': [{'index': 0, 'id': 'outside',
                        'function': {'name': 'read_reference_file', 'arguments': '{"path":"r1/../secret.txt"}'}}]}, 'tool_calls'))
                self.assertIn('error', body['messages'][-1]['content'])
                return httpx.Response(200, text=chunk({'content': '无法访问该文件。'}, 'stop'))
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [directory]})
            self.assertIn('无法访问', backend.run('read', threading.Event()))
