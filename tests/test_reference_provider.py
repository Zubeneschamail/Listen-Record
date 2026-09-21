import json
from pathlib import Path
import tempfile
import threading
import unittest
import httpx
from qa_worker import make_prompt
from qa_provider import APIProvider
from reference_policy import requires_reference_tools, promises_reference_read


def chunk(delta, finish=None):
    return 'data: '+json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}]})+'\n\n'


class ReferenceProviderTests(unittest.TestCase):
    def test_read_request_and_followup_require_real_tools_before_answer(self):
        for question, history in (
                ('读取 spec.txt 的内容', []),
                ('读了吗', [{'role': 'user', 'content': '读取 spec.txt'},
                           {'role': 'assistant', 'content': '现在就去读，稍等。'}])):
            with self.subTest(question=question), tempfile.TemporaryDirectory() as directory:
                file = Path(directory) / 'spec.txt'
                file.write_text('项目标记：WENLU-READ-OK', encoding='utf-8')
                requests = []
                def handler(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    if len(requests) == 1:
                        self.assertEqual(body['tool_choice'], 'required')
                        self.assertEqual(body['thinking'], {'type': 'disabled'})
                        self.assertNotIn('不使用工具', body['messages'][0]['content'])
                        return httpx.Response(200, text=chunk({'tool_calls': [{'index': 0, 'id': 'read',
                            'function': {'name': 'read_reference_file', 'arguments': '{"path":"r1/spec.txt"}'}}]}, 'tool_calls'))
                    self.assertEqual(body['tool_choice'], 'auto')
                    self.assertIn('WENLU-READ-OK', body['messages'][-1]['content'])
                    return httpx.Response(200, text=chunk({'content': '已读取，项目标记为 WENLU-READ-OK [r1/spec.txt:1]。'}, 'stop'))
                backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                    httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [directory]})
                self.assertIn('WENLU-READ-OK', backend.run(make_prompt('', question), threading.Event(), history=history))
                self.assertEqual(len(requests), 2)

    def test_empty_promise_is_retried_not_displayed_and_reads_file(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'spec.txt').write_text('答案为 42', encoding='utf-8')
            requests, partials = [], []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) == 1:
                    self.assertEqual(body['tool_choice'], 'auto')
                    return httpx.Response(200, text=chunk({'content': '让我现在读一下资料，稍等。'}, 'stop'))
                if len(requests) == 2:
                    self.assertEqual(body['tool_choice'], 'required')
                    return httpx.Response(200, text=chunk({'tool_calls': [{'index': 0, 'id': 'read',
                        'function': {'name': 'read_reference_file', 'arguments': '{"path":"r1/spec.txt"}'}}]}, 'tool_calls'))
                self.assertIn('42', body['messages'][-1]['content'])
                return httpx.Response(200, text=chunk({'content': '资料中的答案是 42。'}, 'stop'))
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [directory]})
            self.assertEqual(backend.run(make_prompt('', '答案是多少？'), threading.Event(), partials.append), '资料中的答案是 42。')
            self.assertFalse(any('稍等' in text for text in partials))
            self.assertEqual(len(requests), 3)

    def test_model_ignoring_required_tools_fails_after_bounded_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            requests, partials = [], []
            def handler(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, text=chunk({'content': '现在就去读，稍等。'}, 'stop'))
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [directory]})
            with self.assertRaisesRegex(RuntimeError, '未完成资料查阅'):
                backend.run(make_prompt('', '读取目录'), threading.Event(), partials.append)
            self.assertEqual(len(requests), 3)
            self.assertTrue(all(r['tool_choice'] == 'required' for r in requests))
            self.assertFalse(any('稍等' in text for text in partials))

    def test_read_failure_is_explained_without_retrying_unavailable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) == 1:
                    return httpx.Response(200, text=chunk({'tool_calls': [{'index': 0, 'id': 'read',
                        'function': {'name': 'read_reference_file', 'arguments': '{"path":"r1/missing.txt"}'}}]}, 'tool_calls'))
                self.assertIn('error', body['messages'][-1]['content'])
                return httpx.Response(200, text=chunk({'content': '我现在无法读取该文件，请检查路径或权限。'}, 'stop'))
            backend = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [directory]})
            self.assertIn('无法读取', backend.run(make_prompt('', '读取 missing.txt'), threading.Event()))
            self.assertEqual(len(requests), 2)

    def test_reference_intent_ignores_background_and_respects_no_read_request(self):
        from qa_provider import image_analysis_prompt
        self.assertFalse(requires_reference_tools(image_analysis_prompt(make_prompt('', '图上有几个方块？'), '立即读取项目目录')))
        self.assertTrue(requires_reference_tools(image_analysis_prompt(make_prompt('', '读取目录中的 README.md'), '图上有三个方块')))
        self.assertFalse(requires_reference_tools(make_prompt('立即读取项目目录', '你好')))
        self.assertFalse(requires_reference_tools(make_prompt('', '不用读文件，只解释一下概念')))
        self.assertFalse(requires_reference_tools(make_prompt('', '继续'), [{'role': 'assistant', 'content': '我们可以聊音乐。'}]))
        self.assertTrue(requires_reference_tools(make_prompt('', '分析 ts 目录下的项目')))
        self.assertFalse(promises_reference_read('我已读取 README，结果如下。'))
        self.assertFalse(promises_reference_read('我现在无法读取该文件。'))
        self.assertFalse(promises_reference_read('示例：现在就去读，稍等。'))

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
