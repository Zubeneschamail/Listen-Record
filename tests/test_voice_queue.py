import json
from pathlib import Path
import queue
import tempfile
import threading
import tkinter as tk
import unittest
from unittest.mock import patch

from app import App
from qa_worker import QAWorker


class VoiceQueueTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.]
        self.events = queue.Queue()
        self.prompts = []
        self.qa = QAWorker(self.events, self.run_answer, lambda: self.now[0])
        self.qa.set_enabled(True)

    def run_answer(self, prompt, cancel):
        self.prompts.append(json.loads(prompt.split('\n', 1)[1]))
        return '本地回答'

    def finish(self):
        messages = [self.events.get(timeout=3)[1] for _ in range(3)]
        self.assertEqual([item[1] for item in messages], ['thinking', 'answer', 'done'])
        self.qa.active = False
        return messages[0][2]

    def collect(self, text, source='system', key=None):
        self.qa.feed(text, source, record_key=key)
        self.now[0] += 2
        self.qa.collect_voice()

    def test_finished_questions_stay_separate_during_an_active_answer(self):
        self.qa.ask('之前的问题', [])
        self.collect('Redis 缓存击穿怎么处理？', key=0)
        self.collect('MySQL 慢查询怎么排查？', key=1)
        self.assertEqual(len(self.qa.voice_queue), 2)
        self.finish()
        self.qa.tick()
        self.assertEqual(self.finish(), 'Redis 缓存击穿怎么处理？')
        self.assertNotIn('MySQL', self.prompts[1]['背景转写'])
        self.qa.tick()
        self.assertEqual(self.finish(), 'MySQL 慢查询怎么排查？')

    def test_speaking_and_unstable_fragments_are_not_sealed(self):
        self.qa.active = True
        self.qa.feed('Redis 怎么')
        self.now[0] = 1
        self.qa.tick()
        self.assertFalse(self.qa.voice_queue)
        self.qa.feed('避免缓存击穿？')
        self.qa.observe_voice('system', True)
        self.now[0] = 10
        self.qa.tick()
        self.assertFalse(self.qa.voice_queue)
        self.qa.observe_voice('system', False)
        self.qa.tick()
        self.assertEqual(self.qa.voice_queue[0].question.text(), 'Redis 怎么\n避免缓存击穿？')

    def test_ready_sources_follow_finish_order_and_candidate_is_context_only(self):
        self.qa.active = True
        self.qa.feed('系统问题是什么？', 'system')
        self.now[0] = .4
        self.qa.feed('麦克风问题是什么？', 'microphone')
        self.now[0] = 3
        self.qa.tick()
        self.assertEqual([row.source for row in self.qa.voice_queue], ['system', 'microphone'])
        self.qa.set_question_source('system')
        self.collect('我负责数据库。', 'microphone')
        self.assertFalse(self.qa.voice_queue)
        self.assertIn('我负责数据库。', self.qa.context)

    def test_repeat_waiting_in_queue_is_suppressed_until_context_changes(self):
        self.qa.set_question_source('system')
        self.qa.active = True
        self.collect('C++ 是什么？')
        self.collect('Ｃ＋＋ 是什么?')
        self.assertEqual(len(self.qa.voice_queue), 1)
        self.qa.feed('我主要使用Java。', 'microphone')
        self.collect('C++ 是什么？')
        self.assertEqual(len(self.qa.voice_queue), 2)

    def test_queue_is_bounded_and_overflow_has_a_recovery_notice(self):
        self.qa.active = True
        for index in range(9):
            self.collect(f'第{index}个问题是什么？', key=index)
        self.assertEqual(len(self.qa.voice_queue), 8)
        kind, (epoch, message) = self.events.get_nowait()
        self.assertEqual(kind, 'qa_notice')
        self.assertEqual(epoch, self.qa.voice_epoch)
        self.assertIn('未自动发送', message)
        self.assertIn('转录气泡', message)

    def test_reset_disable_source_change_clear_queue_but_cancel_keeps_it(self):
        for action in ('reset', 'disable', 'source'):
            with self.subTest(action=action):
                self.qa.set_enabled(True)
                self.qa.set_question_source('both')
                self.qa.active = True
                self.collect('解释一下事务？')
                self.qa.cancel_request()
                self.assertEqual(len(self.qa.voice_queue), 1)
                if action == 'reset':
                    self.qa.reset()
                elif action == 'disable':
                    self.qa.set_enabled(False)
                else:
                    self.qa.set_question_source('microphone')
                self.assertFalse(self.qa.voice_queue)

    def test_correction_changes_only_the_identified_queued_question_and_context(self):
        self.qa.active = True
        self.qa.set_question_source('system')
        self.collect('什么是密等性？', key=0)
        self.qa.feed('我负责支付接口。', 'microphone', record_key=1)
        self.collect('什么是密等性？', key=2)
        self.qa.correct_transcript(0, '什么是幂等性？')
        self.qa.active = False
        self.qa.tick()
        self.assertEqual(self.finish(), '什么是幂等性？')
        self.assertIn('幂等性', self.prompts[0]['背景转写'])
        self.qa.tick()
        self.assertEqual(self.finish(), '什么是密等性？')
        self.assertFalse(self.qa.voice_queue)

    def test_background_correction_does_not_merge_two_independent_repeats(self):
        self.qa.active = True
        self.qa.set_question_source('system')
        self.qa.feed('项目背景。', 'microphone', record_key=0)
        self.collect('为什么？', key=1)
        self.qa.feed('因为先写数据库。', 'microphone', record_key=2)
        self.collect('为什么？', key=3)
        self.qa.correct_transcript(0, '修正后的项目背景。')
        self.qa.active = False
        self.qa.tick()
        self.assertEqual(self.finish(), '为什么？')
        self.qa.tick()
        self.assertTrue(self.qa.active)
        self.assertEqual(self.finish(), '为什么？')


class VoiceQueueAppTests(unittest.TestCase):
    def setUp(self):
        for target, options in (
                ('app.GlobalHotkey', {}), ('app.App.start_tray', {}),
                ('app.ClipboardWatcher', {}), ('qa_connection.QAConnection.check', {}),
                ('app.save_desktop', {}), ('app.preferences', {'return_value': {}}),
                ('app.load_settings', {'return_value': {'provider': 'compatible', 'profiles': {}}})):
            patched = patch(target, **options)
            patched.start()
            self.addCleanup(patched.stop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = App(self.root)
        self.folder = tempfile.TemporaryDirectory()
        self.app.session = Path(self.folder.name)/'interview.jsonl'
        self.app.auto_qa.set(True)
        self.app.toggle_auto_qa()
        self.app.qa_connection.state = 'verified'
        self.now = [0.]
        self.app.qa.clock = lambda: self.now[0]
        self.release = threading.Event()
        self.requests = []
        self.app.qa.stream_runner = self.stream

    def stream(self, prompt, cancel, partial, **options):
        payload = json.loads(prompt.split('\n', 1)[1])
        self.requests.append((payload, options.get('history', [])))
        if payload['当前问题'] == '旧问题':
            self.release.wait(5)
        return '回答：'+payload['当前问题']

    def tearDown(self):
        self.release.set()
        self.app.qa.set_enabled(False)
        self.app.qa_connection.close()
        self.app.gpu_check.close()
        self.app.hotkey.close()
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.root.destroy()
        self.folder.cleanup()

    def add(self, text, source='system'):
        index = len(self.app.rows)
        self.app.events.put(('segment', dict(start=index*2, end=index*2+1, text=text, source=source)))
        self.app.poll()

    def settle(self):
        self.now[0] += 2
        self.app.poll()

    def finish(self):
        while True:
            kind, value = self.app.events.get(timeout=3)
            if kind == 'qa':
                self.app.handle_qa(value)
                if value[1] == 'done':
                    return

    def test_queued_turn_uses_its_transcript_boundary_and_latest_completed_answer(self):
        self.app.qa.ask('旧问题', [])
        self.add('Redis 缓存击穿怎么处理？')
        self.settle()
        self.add('MySQL 慢查询怎么排查？')
        self.settle()
        self.assertEqual(len(self.app.qa.voice_queue), 2)
        self.release.set()
        self.finish()
        self.app.poll()
        self.finish()
        question, history = self.requests[1]
        self.assertEqual(question['当前问题'], 'Redis 缓存击穿怎么处理？')
        self.assertNotIn('MySQL', json.dumps(question['会话资料'], ensure_ascii=False))
        self.assertIn('回答：旧问题', json.dumps(history, ensure_ascii=False))
        self.app.poll()
        self.finish()
        self.assertEqual([q for q, _ in self.app.qa_history],
                         ['旧问题', 'Redis 缓存击穿怎么处理？', 'MySQL 慢查询怎么排查？'])

    def test_unavailable_service_still_seals_turns_then_resumes_in_order(self):
        self.app.qa_connection.state = 'unavailable'
        for text in ('解释一下Redis？', '解释一下MySQL？'):
            self.add(text)
            self.settle()
        self.assertEqual(len(self.app.qa.voice_queue), 2)
        self.assertFalse(self.requests)
        self.app.qa_connection.state = 'verified'
        self.app.poll()
        self.finish()
        self.app.poll()
        self.finish()
        self.assertEqual([q for q, _ in self.app.qa_history], ['解释一下Redis？', '解释一下MySQL？'])

    def test_correction_of_queued_row_refreshes_question_and_bounded_background(self):
        self.app.qa_connection.state = 'unavailable'
        self.add('什么是密等性？')
        self.settle()
        self.add('下一个问题，MySQL如何优化？')
        self.settle()
        original = self.app.session.read_bytes()
        self.app.correct_transcript(0, '什么是幂等性？')
        self.assertEqual(self.app.session.read_bytes(), original)
        self.app.qa_connection.state = 'verified'
        self.app.poll()
        self.finish()
        payload = self.requests[0][0]
        self.assertEqual(payload['当前问题'], '什么是幂等性？')
        background = json.dumps(payload['会话资料'], ensure_ascii=False)
        self.assertIn('用户已修正', background)
        self.assertNotIn('密等性', background)
        self.assertNotIn('MySQL', background)

    def test_overflow_notice_keeps_all_original_transcript_rows(self):
        self.app.qa_connection.state = 'unavailable'
        for index in range(9):
            self.add(f'第{index}道题是什么？')
            self.settle()
        self.app.poll()
        self.assertEqual(len(self.app.rows), 9)
        self.assertEqual(len(self.app.session.read_text(encoding='utf-8').splitlines()), 9)
        self.assertEqual(len(self.app.qa.voice_queue), 8)
        self.assertIn('未自动发送', self.app.qa_status.get())
        self.app.auto_qa.set(False)
        self.app.toggle_auto_qa()
        self.assertFalse(self.app.qa.voice_queue)

    def test_queued_overflow_notice_is_ignored_after_conversation_clear(self):
        self.app.qa_connection.state = 'unavailable'
        for index in range(9):
            self.add(f'第{index}道题是什么？')
            self.settle()
        # The final collector just emitted its notice for the next event poll.
        self.app.clear(preserve_recording=True)
        self.app.poll()
        self.assertFalse(self.app.qa.voice_queue)
        self.assertEqual(self.app.qa_status.get(), '')
