import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from qa_provider import APIProvider
from qa_worker import make_prompt
from reference_files import ReferenceTools
from reference_policy import clean_tool_history, has_tool_markup, visible_partial


RAW = ('<｜DSML｜tool_calls>\n<｜DSML｜invoke name="read_reference_file">\n'
       '<｜DSML｜parameter name="path" string="true">r1/unsafe.txt</｜DSML｜parameter>\n'
       '</｜DSML｜invoke>\n</｜DSML｜tool_calls>')


def chunk(delta, finish=None):
    return 'data: '+json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}]})+'\n\n'


def call(name, arguments, identifier):
    return chunk({'tool_calls': [{'index': 0, 'id': identifier,
                  'function': {'name': name, 'arguments': json.dumps(arguments)}}]}, 'tool_calls')


class ToolMarkupTests(unittest.TestCase):
    def backend(self, handler, directory=None):
        return APIProvider('deepseek', {'base_url': 'https://example.com', 'model': 'test', 'api_key': 'fake'},
            httpx.MockTransport(handler), (lambda: {'enabled': True, 'paths': [directory]}) if directory else None)

    def test_leaked_markup_after_listing_is_retried_and_only_structured_calls_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'spec.txt').write_text('Reference marker: 98765', encoding='utf-8')
            requests, partials, executed = [], [], []
            execute = ReferenceTools.execute
            def tracked(tools, name, arguments):
                executed.append((name, arguments))
                return execute(tools, name, arguments)
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) == 1:
                    return httpx.Response(200, text=call('list_reference_files', {}, 'list'))
                if len(requests) == 2:
                    self.assertEqual(body['tool_choice'], 'auto')
                    return httpx.Response(200, text=chunk({'content': RAW}, 'stop'))
                if len(requests) == 3:
                    self.assertEqual(body['tool_choice'], 'required')
                    self.assertFalse(any(has_tool_markup(m.get('content')) for m in body['messages']))
                    return httpx.Response(200, text=call('read_reference_file', {'path': 'r1/spec.txt'}, 'read'))
                self.assertIn('98765', body['messages'][-1]['content'])
                return httpx.Response(200, text=chunk({'content': '项目标记是 98765。'}, 'stop'))
            with patch.object(ReferenceTools, 'execute', tracked):
                answer = self.backend(handler, directory).run(make_prompt('', '分析项目实现'), threading.Event(), partials.append)
            self.assertEqual(answer, '项目标记是 98765。')
            self.assertEqual(len(executed), 2)
            self.assertNotIn('unsafe.txt', str(executed))
            self.assertFalse(any(has_tool_markup(p) for p in partials))

    def test_repeated_or_truncated_protocol_is_never_executed_or_shown(self):
        for raw, finish in ((RAW, 'stop'), (RAW[:45], 'length'), ('<｜DS', 'stop')):
            with self.subTest(finish=finish, raw=raw), tempfile.TemporaryDirectory() as directory:
                requests, partials = [], []
                def handler(request):
                    requests.append(request)
                    return httpx.Response(200, text=chunk({'content': raw}, finish))
                with patch.object(ReferenceTools, 'execute') as execute, self.assertRaisesRegex(RuntimeError, '工具指令格式无效'):
                    self.backend(handler, directory).run(make_prompt('', '读取文件'), threading.Event(), partials.append)
                execute.assert_not_called()
                self.assertEqual(len(requests), 3)
                self.assertFalse(any('<' in part or 'DSML' in part for part in partials))

    def test_fragmented_stream_without_tools_is_blocked_including_partial_prefix(self):
        for raw in (RAW, RAW.replace('｜', '|'), '< | | DSML | | invoke name="read_reference_file">'):
            partials = []
            response = ''.join(chunk({'content': char}) for char in raw)+chunk({}, 'stop')
            with patch('qa_provider.time.monotonic', side_effect=iter(range(10000))), self.assertRaisesRegex(RuntimeError, '工具指令格式无效'):
                self.backend(lambda request: httpx.Response(200, text=response)).run('hello', threading.Event(), partials.append, timeout=10000, use_references=False)
            self.assertEqual(partials, [])
        for text in ('x < 3', '<div>普通 HTML 示例</div>', '使用 DSML 的含义是什么？'):
            self.assertFalse(has_tool_markup(text))
            self.assertEqual(visible_partial(text), text)

    def test_native_tool_calls_take_precedence_over_leaked_body(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) == 1:
                    return httpx.Response(200, text=chunk({'content': RAW})+call('list_reference_files', {}, 'list'))
                self.assertIsNone(body['messages'][-2]['content'])
                self.assertEqual(body['messages'][-2]['tool_calls'][0]['id'], 'list')
                return httpx.Response(200, text=chunk({'content': '资料目录为空。'}, 'stop'))
            self.assertEqual(self.backend(handler, directory).run('读取目录', threading.Event()), '资料目录为空。')
            self.assertEqual(len(requests), 2)

    def test_invalid_assistant_history_is_not_replayed_and_user_content_is_unchanged(self):
        history = [{'role': 'user', 'content': RAW}, {'role': 'assistant', 'content': RAW},
                   {'role': 'assistant', 'content': '正常回答'}]
        cleaned = clean_tool_history(history)
        self.assertEqual(cleaned[0], history[0])
        self.assertEqual(cleaned[2], history[2])
        self.assertFalse(has_tool_markup(cleaned[1]['content']))
        self.assertEqual(history[1]['content'], RAW)
        def handler(request):
            messages = json.loads(request.content)['messages']
            self.assertFalse(any(has_tool_markup(m.get('content')) for m in messages if m['role'] == 'assistant'))
            return httpx.Response(200, text=chunk({'content': '收到'}, 'stop'))
        self.assertEqual(self.backend(handler).run('下一问', threading.Event(), history=history), '收到')

    def test_batch_can_read_more_than_four_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for i in range(6):
                (Path(directory) / f'file{i}.txt').write_text(f'marker-{i}', encoding='utf-8')
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) == 1:
                    calls = [{'index': i, 'id': f'read{i}', 'function': {'name': 'read_reference_file',
                              'arguments': json.dumps({'path': f'r1/file{i}.txt'})}} for i in range(6)]
                    return httpx.Response(200, text=chunk({'tool_calls': calls}, 'tool_calls'))
                results = body['messages'][-6:]
                for i, result in enumerate(results):
                    self.assertEqual(result['tool_call_id'], f'read{i}')
                    self.assertIn(f'marker-{i}', result['content'])
                return httpx.Response(200, text=chunk({'content': '六个文件已经读取完成。'}, 'stop'))
            answer = self.backend(handler, directory).run(make_prompt('', '分析目录中的所有文件'), threading.Event())
            self.assertIn('读取完成', answer)
            self.assertEqual(len(requests), 2)

    def test_total_budget_returns_a_result_for_every_call_and_stops_new_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) <= 2:
                    calls = [{'index': i, 'id': f'r{len(requests)}_{i}',
                              'function': {'name': 'list_reference_files', 'arguments': '{}'}} for i in range(16)]
                    return httpx.Response(200, text=chunk({'tool_calls': calls}, 'tool_calls'))
                self.assertEqual(body['tool_choice'], 'none')
                self.assertEqual(sum(m['role'] == 'tool' for m in body['messages']), 32)
                self.assertTrue(all('error' in json.loads(m['content']) for m in body['messages'][-8:]))
                return httpx.Response(200, text=chunk({'content': '已达到查阅限额，未读取剩余文件。'}, 'stop'))
            with patch.object(ReferenceTools, 'execute', return_value='{"files": []}') as execute:
                answer = self.backend(handler, directory).run('分析目录', threading.Event())
            self.assertEqual(execute.call_count, 24)
            self.assertIn('查阅限额', answer)

    def test_oversized_batch_is_rejected_before_any_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            response = chunk({'tool_calls': [{'index': 16, 'id': 'extra',
                             'function': {'name': 'list_reference_files', 'arguments': '{}'}}]}, 'tool_calls')
            with patch.object(ReferenceTools, 'execute') as execute, self.assertRaisesRegex(RuntimeError, '本轮查阅任务过多'):
                self.backend(lambda request: httpx.Response(200, text=response), directory).run('读取目录', threading.Event())
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
