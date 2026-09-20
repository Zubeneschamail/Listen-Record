import json
import queue
import unittest
from qa_worker import QAWorker
from session_context import build_context, build_conversation, encoded, CONTEXT_BYTES


class SessionContextTests(unittest.TestCase):
    def test_api_conversation_preserves_roles_without_duplicated_history(self):
        background, messages = build_conversation([], [('两数之和', '需要代码吗？')], '要的')
        self.assertEqual(messages, [{'role': 'user', 'content': '两数之和'},
                                    {'role': 'assistant', 'content': '需要代码吗？'}])
        self.assertNotIn('需要代码', encoded(background))

    def test_long_api_conversation_keeps_recent_pairs_and_bounded_background(self):
        exchanges = [('题目'+str(i), ('推导内容。'*200)+f'这是第{i}步') for i in range(200)]
        background, messages = build_conversation([('system', '额外背景'*1000)], exchanges, '继续')
        self.assertLessEqual(len(encoded([background, messages]).encode('utf-8')), CONTEXT_BYTES)
        self.assertEqual(messages[-1], {'role': 'assistant', 'content': exchanges[-1][1]})
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant']*(len(messages)//2))
        self.assertNotIn(exchanges[-1][1], encoded(background))

    def test_oversized_api_turn_preserves_answer_tail_and_does_not_mutate_history(self):
        exchanges = [('图片问题'+('\\\"\n'*40000), '开头'+('很长的描述'*30000)+'需要我写示例代码吗？')]
        before = list(exchanges)
        background, messages = build_conversation([], exchanges, '要的')
        self.assertLessEqual(len(encoded([background, messages]).encode('utf-8')), CONTEXT_BYTES)
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
        self.assertTrue(messages[-1]['content'].endswith('需要我写示例代码吗？'))
        self.assertEqual(exchanges, before)

    def test_short_session_keeps_all_transcripts_and_answers(self):
        rows = [('system', f'第{i}段讨论') for i in range(30)]
        exchanges = [('先前问题', '第一点：使用缓存。第二点：增量更新。')]
        result = build_context(rows, exchanges, '第二点怎么实现？')
        self.assertEqual(result['模式'], '完整会话')
        self.assertEqual(len(result['全部转写']), 30)
        self.assertIn('增量更新', encoded(result))

    def test_long_session_is_bounded_and_retrieves_early_evidence(self):
        rows = [('system', '海豚项目约定使用 SQLite，每天备份。')]
        rows += [('microphone', f'{i} 普通背景。' * 80) for i in range(500)]
        result = build_context(rows, [('历史问题', 'AI 的推测')]*100, '海豚项目用什么数据库？')
        self.assertEqual(result['模式'], '压缩会话')
        self.assertLessEqual(len(encoded(result).encode('utf-8')), CONTEXT_BYTES)
        self.assertIn('SQLite', result['相关早期原文摘录'])
        self.assertIn('499', result['近期转写原文'])
        self.assertIn('AI 的推测', result['近期问答原文'])

    def test_request_uses_session_snapshot_after_reset(self):
        events, prompts = queue.Queue(), []
        qa = QAWorker(events, lambda p, c: prompts.append(p) or '新答案')
        rows, answers = [('system', '最早的转录')], [('修改后的问题', '修订后的答案')]
        qa.context_provider = lambda: (list(rows), list(answers))
        qa.set_enabled(True)
        qa.reset()
        qa.ask('刚才的答案再展开', [])
        for _ in range(3):
            events.get(timeout=3)
        data = json.loads(prompts[0].split('\n', 1)[1])
        self.assertIn('修订后的答案', encoded(data))
        self.assertIn('最早的转录', encoded(data))
        rows.clear(); answers.clear()
        qa.ask('新会话', [])
        for _ in range(3):
            events.get(timeout=3)
        self.assertNotIn('修订后的答案', prompts[-1])

    def test_oversized_newest_entry_keeps_a_recent_excerpt(self):
        result = build_context([('system', '开头 '+('很长的记录。'*30000)+' 最后约定周五交付')], [], '继续')
        self.assertIn('周五交付', result['近期转写原文'])
        self.assertLessEqual(len(encoded(result).encode('utf-8')), CONTEXT_BYTES)
