import json
from pathlib import Path
import tempfile
import threading
import unittest

import httpx

from interview_context import needs_previous_question, concise_interview
from knowledge_packages import make_query
from qa_provider import APIProvider
from qa_worker import make_prompt
from reference_policy import asks_about_personal_experience, promises_reference_read
from session_context import build_context, build_conversation, encoded


class InterviewContextTests(unittest.TestCase):
    def setUp(self):
        self.exchanges = [
            ('介绍一下 Java 订单接口', '订单接口使用唯一约束防重复写入。'),
            ('介绍一下 Agent 和 Code Review', '评审任务进入队列后调用模型。'),
            ('那怎么控制成本？', '评审模型按 token 设置预算。'),
        ]

    def test_new_backend_topic_excludes_ai_answers_but_preserves_user_facts(self):
        rows = [('microphone', '我在电商团队参与过 Java 营销接口。')]
        original = list(self.exchanges)
        background, history = build_conversation(rows, self.exchanges, '高并发怎么处理？')
        self.assertEqual(len(history), 2)
        self.assertIn('订单接口', history[-1]['content'])
        self.assertNotIn('评审任务进入队列', encoded([background, history]))
        self.assertNotIn('评审模型按 token', encoded([background, history]))
        self.assertIn('介绍一下 Agent', encoded(background))
        self.assertIn('Java 营销接口', encoded(background))
        legacy = build_context(rows, self.exchanges, '高并发怎么处理？')
        self.assertNotIn('评审任务进入队列', encoded(legacy))
        self.assertIn('订单接口使用唯一约束', encoded(legacy))
        self.assertEqual(self.exchanges, original)
        self.assertEqual(len(rows), 1)

    def test_short_standalone_queries_do_not_borrow_code_review(self):
        history = [{'role': 'user', 'content': q} for q, _ in self.exchanges]
        for question in ('高并发怎么处理？', '说说事务', 'Vue 响应式原理是什么？', 'Redis 缓存击穿呢？',
                         'LangChain 适合什么场景？'):
            with self.subTest(question=question):
                query, nearby = make_query(make_prompt('', question), history)
                self.assertEqual(query, question)
                self.assertEqual(nearby, '')

    def test_explicit_project_and_elliptical_followups_keep_context(self):
        for question in ('刚才那个项目的高并发怎么处理？', '这个评审方案怎么优化？', '那怎么处理？', '继续', '第二点怎么实现？'):
            with self.subTest(question=question):
                self.assertTrue(needs_previous_question(question))
                _, history = build_conversation([], self.exchanges, question)
                self.assertEqual(len(history), 6)
                self.assertIn('Code Review', make_query(make_prompt('', question), history)[1])
        _, history = build_conversation([], self.exchanges, 'Code Review 的高并发怎么处理？')
        self.assertTrue(any('评审任务' in item['content'] for item in history))

    def test_topic_switch_works_in_both_directions(self):
        exchanges = self.exchanges + [('高并发怎么处理？', '业务接口用限流保护数据库。')]
        _, history = build_conversation([], exchanges, 'MCP 工具如何接入？')
        self.assertNotIn('业务接口用限流', encoded(history))
        self.assertIn('评审任务', encoded(history))
        _, history = build_conversation([], exchanges, 'Vue 响应式怎么实现？')
        self.assertEqual(history, [])

    def test_return_to_named_project_retrieves_old_topic_not_latest_turns(self):
        exchanges = [
            ('介绍一下 Code Review', '评审使用 MCP。'),
            ('这套评审流程怎么接入研发工作流？', '调用 report 发布结果。'),
            ('列表数据多了以后翻到后面越来越慢，这种情况怎么处理？', '使用游标分页。'),
            ('Nginx 返回 502 怎么查？', '检查上游端口。'),
        ]
        question = '回到刚才的 Code Review，如果 Jira 接口超时，你会怎么处理？'
        background, history = build_conversation([], exchanges, question)
        self.assertEqual(len(history), 4)
        self.assertNotIn('使用游标分页', encoded(history))
        self.assertNotIn('检查上游端口', encoded(history))
        # Exercise retrieval both with filtered history and the full provider input.
        full = [{'role': 'user', 'content': q} for q, _ in exchanges]
        for messages in (history, full):
            _, nearby = make_query(make_prompt('', question), messages)
            self.assertIn('Code Review', nearby)
            self.assertIn('评审流程', nearby)
            self.assertNotIn('Nginx', nearby)
            self.assertNotIn('列表', nearby)
        self.assertIn('Nginx', encoded(background))  # User facts remain available.

    def test_natural_pagination_question_filters_ai_context(self):
        _, history = build_conversation([], self.exchanges,
            '列表数据多了以后翻到后面越来越慢，这种情况怎么处理？')
        self.assertNotIn('评审', encoded(history))
        self.assertIn('订单', encoded(history))

    def test_backend_scenario_loads_resume_and_respects_opt_out(self):
        self.assertTrue(asks_about_personal_experience(make_prompt('', '高并发怎么处理？')))
        self.assertFalse(asks_about_personal_experience(make_prompt('', '不用参考简历，高并发怎么处理？')))
        self.assertFalse(asks_about_personal_experience(make_prompt('高并发怎么处理', '事务是什么？')))

    def test_short_answers_do_not_confuse_ui_lists_or_code_review_with_format_requests(self):
        for question in ('介绍 Code Review 项目', '说说代码评审', '列表翻到后面很慢怎么办？'):
            self.assertTrue(concise_interview(make_prompt('', question), question))
        for question in ('详细解释事务', '给出完整代码', '列出问题清单', '回答500字', '核实源码实现'):
            self.assertFalse(concise_interview(make_prompt('', question), question))
        self.assertFalse(concise_interview('普通请求', '事务是什么'))

    def test_troubleshooting_description_is_not_a_promise_to_read_local_files(self):
        for answer in ('我会先看 Nginx 的错误日志，确认上游连接是否正常。',
                       '我先查执行计划，确认索引是否命中。'):
            self.assertFalse(promises_reference_read(answer, scenario=True))
        self.assertTrue(promises_reference_read('让我现在读一下参考资料，稍等。', scenario=True))

    def test_troubleshooting_answer_streams_with_references_without_forcing_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, partials = [], []
            answer = '我会先查 Nginx 的错误日志，确认上游连接是否正常，再直连服务的健康检查接口。'
            def handler(request):
                calls.append(request)
                return httpx.Response(200, text='data: ' + json.dumps({'choices': [
                    {'delta': {'content': answer}, 'finish_reason': 'stop'}]}) + '\n\n')
            provider = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [directory]})
            result = provider.run(make_prompt('', 'Nginx 502 怎么排查？'), threading.Event(), partials.append)
            self.assertEqual(result, answer)
            self.assertIn(answer, partials)
            self.assertEqual(len(calls), 1)

    def test_long_interview_answer_gets_one_bounded_rewrite_without_tools(self):
        bodies = []
        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            answer = '先确认代理调用是否生效。' * 40 if len(bodies) == 1 else '先确认调用经过代理，再检查异常类型和捕获位置。'
            return httpx.Response(200, text='data: ' + json.dumps({'choices': [
                {'delta': {'content': answer}, 'finish_reason': 'stop'}]}) + '\n\n')
        provider = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                               httpx.MockTransport(handler))
        result = provider.run(make_prompt('', '事务没有回滚怎么办？'), threading.Event())
        self.assertIn('捕获位置', result)
        self.assertEqual(len(bodies), 2)
        self.assertNotIn('tools', bodies[0])  # Ordinary technical questions don't trigger web searches.
        self.assertIn('压缩后的完整口头回答', bodies[1]['messages'][-1]['content'])
        bodies.clear()
        provider.run(make_prompt('', '详细解释事务没有回滚的原因'), threading.Event())
        self.assertEqual(len(bodies), 1)

    def test_rewrite_remains_bounded_if_model_ignores_length_request(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, text='data: ' + json.dumps({'choices': [
                {'delta': {'content': '方案细节。' * 100}, 'finish_reason': 'stop'}]}) + '\n\n')
        provider = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                               httpx.MockTransport(handler))
        answer = provider.run(make_prompt('', 'Vue 首屏怎么优化？'), threading.Event())
        self.assertEqual(len(calls), 2)
        self.assertEqual(answer, '方案细节。' * 100)  # Never cut a caveat or retry indefinitely.

    def test_provider_receives_backend_history_and_actual_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / '简历.txt'
            resume.write_text('工作经历\n永辉：参与 Java 营销接口开发。\n项目经历\n视客：运营后台。\n教育经历\n本科。', encoding='utf-8')
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                messages = body['messages']
                self.assertIn('永辉：参与 Java 营销接口开发', messages[-1]['content'])
                self.assertNotIn('评审任务进入队列', encoded(messages))
                self.assertNotIn('评审模型按 token', encoded(messages))
                self.assertTrue(any(m['role'] == 'assistant' and '唯一约束' in m['content'] for m in messages))
                return httpx.Response(200, text='data: ' + json.dumps({'choices': [
                    {'delta': {'content': '先确认业务接口的瓶颈。'}, 'finish_reason': 'stop'}]}) + '\n\n')
            provider = APIProvider('deepseek', dict(base_url='https://example.com', model='test', api_key='fake'),
                httpx.MockTransport(handler), lambda: {'enabled': True, 'paths': [str(resume)]})
            background, history = build_conversation([], self.exchanges, '高并发怎么处理？')
            result = provider.run(make_prompt('', '高并发怎么处理？', background), threading.Event(), history=history)
            self.assertEqual(result, '先确认业务接口的瓶颈。')
            self.assertEqual(len(requests), 1)


if __name__ == '__main__':
    unittest.main()
