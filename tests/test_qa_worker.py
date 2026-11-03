import json
import queue
import threading
import unittest
from qa_worker import QAWorker, is_question


class QAWorkerTests(unittest.TestCase):
    def test_different_technical_symbols_are_not_duplicate_questions(self):
        qa, now, events = self.make_worker()
        questions = ('C++ 是什么？', 'C# 是什么？', 'C 是什么？',
                     '.NET 是什么？', 'NET 是什么？')
        for question in questions:
            qa.feed(question)
            now[0] += 2
            qa.tick()
            self.assertTrue(qa.active, question)
            self.assertEqual(self.finish_request(qa, events), question)

    def test_technical_duplicate_still_ignores_case_width_and_question_mark(self):
        qa, now, events = self.make_worker()
        qa.feed('什么是 C++？')
        now[0] = 2
        qa.tick()
        self.finish_request(qa, events)
        qa.feed('什么是Ｃ＋＋?')
        now[0] += 2
        qa.tick()
        self.assertFalse(qa.active)
        self.assertTrue(events.empty())

    def test_correction_updates_only_identified_context_and_pending_part(self):
        qa, now, events = self.make_worker()
        qa.feed('什么是密等性？', record_key=0)
        qa.feed('什么是密等性？', record_key=1)
        now[0] = 2
        qa.correct_transcript(0, '什么是幂等性？')
        self.assertEqual(list(qa.context), ['什么是幂等性？', '什么是密等性？'])
        self.assertEqual(qa.pending['system'].parts, ['什么是幂等性？', '什么是密等性？'])
        qa.tick()
        self.assertTrue(events.empty())
        now[0] = 4
        qa.tick()
        self.assertEqual(self.finish_request(qa, events), '什么是幂等性？\n什么是密等性？')

    def test_bounded_question_parts_keep_their_correction_identity(self):
        qa, now, events = self.make_worker()
        for index in range(20):
            qa.feed('什么是术语'+str(index)+'？', record_key=index)
        self.assertEqual(list(qa.context_keys), list(range(8, 20)))
        self.assertEqual(qa.pending['system'].keys, [0]+list(range(13, 20)))
        qa.correct_transcript(0, '什么是幂等性？')
        self.assertEqual(qa.pending['system'].parts[0], '什么是幂等性？')
        qa.clear_auto()
        self.assertFalse(qa.context_keys)

    def make_worker(self):
        now = [0.]
        events = queue.Queue()
        worker = QAWorker(events, lambda prompt, cancel: '本地测试回答', lambda: now[0])
        worker.set_enabled(True)
        return worker, now, events

    def finish_request(self, qa, events):
        messages = [events.get(timeout=2)[1] for _ in range(3)]
        self.assertEqual([item[1] for item in messages], ['thinking', 'answer', 'done'])
        qa.active = False
        return messages[0][2]

    def test_interview_commands_without_question_words(self):
        commands = ('举个具体例子。', '请举一个实际的例子。', '展开讲讲。',
                    '设计一个订单系统。', '请你实现一个限流器。',
                    '手写一个LRU缓存。', '编写一段SQL。',
                    '写个SQL查询。', '对比一下两种方案。', '比较一下两种索引。')
        for command in commands:
            with self.subTest(command=command):
                qa, now, events = self.make_worker()
                qa.feed(command)
                now[0] = 2
                qa.tick()
                self.assertEqual(self.finish_request(qa, events), command)
        statements = ('我负责设计一个订单系统。', '我们来对比一下两种方案。',
                      '我刚才举了一个具体例子。', '他让我手写一个缓存。',
                      '继续处理订单。', '这段代码实现了一个限流器。')
        for statement in statements:
            with self.subTest(statement=statement):
                self.assertFalse(is_question(statement))

    def test_short_followups_need_an_earlier_question(self):
        for followup in ('再具体一点。', '请详细一些。', '继续。', '再说一遍。'):
            with self.subTest(followup=followup):
                qa, now, events = self.make_worker()
                qa.feed(followup)
                now[0] = 2
                qa.tick()
                self.assertTrue(events.empty())
                qa.ask('介绍一下缓存方案。', [])
                self.finish_request(qa, events)
                qa.feed(followup)
                now[0] += 2
                qa.tick()
                self.assertEqual(self.finish_request(qa, events), followup)

    def test_short_followups_still_respect_interviewer_source(self):
        qa, now, events = self.make_worker()
        qa.set_question_source('system')
        qa.ask('介绍一下缓存方案。', [])
        self.finish_request(qa, events)
        qa.feed('再具体一点。', 'microphone')
        now[0] = 2
        qa.tick()
        self.assertTrue(events.empty())
        self.assertIn('再具体一点。', qa.context)

    def test_same_followup_after_candidate_reply_is_a_new_question(self):
        qa, now, events = self.make_worker()
        qa.set_question_source('system')
        qa.feed('为什么？', 'system')
        now[0] = 2
        qa.tick()
        self.assertEqual(self.finish_request(qa, events), '为什么？')
        qa.feed('因为重复消息可能造成二次扣款。', 'microphone')
        qa.feed('为什么？', 'system')
        now[0] = 4
        qa.tick()
        self.assertEqual(self.finish_request(qa, events), '为什么？')

    def test_candidate_context_before_submission_does_not_make_repeat_new(self):
        qa, now, events = self.make_worker()
        qa.set_question_source('system')
        qa.feed('为什么？', 'system')
        qa.feed('我的项目里负责接口。', 'microphone')
        now[0] = 2
        qa.tick()
        self.finish_request(qa, events)
        qa.feed('为什么？', 'system')
        now[0] = 4
        qa.tick()
        self.assertTrue(events.empty())

    def test_recent_duplicate_is_ignored_but_later_repeat_is_allowed(self):
        qa, now, events = self.make_worker()
        qa.feed('为什么？')
        now[0] = 2
        qa.tick()
        self.finish_request(qa, events)
        qa.feed('为什么。')  # Punctuation changes do not make a fresh turn.
        now[0] = 4
        qa.tick()
        self.assertTrue(events.empty())
        qa.feed('为什么？')
        now[0] = 11
        qa.tick()
        self.assertEqual(self.finish_request(qa, events), '为什么？')

    def test_same_words_from_two_sources_are_independent(self):
        qa, now, events = self.make_worker()
        qa.feed('为什么？', 'system')
        now[0] = 2
        qa.tick()
        self.finish_request(qa, events)
        qa.feed('为什么？', 'microphone')
        now[0] = 4
        qa.tick()
        self.assertEqual(self.finish_request(qa, events), '为什么？')

    def test_reset_forgets_elliptical_followup_context(self):
        qa, now, events = self.make_worker()
        qa.ask('介绍一下缓存方案。', [])
        self.finish_request(qa, events)
        qa.reset()
        qa.feed('继续。')
        now[0] = 2
        qa.tick()
        self.assertTrue(events.empty())

    def test_candidate_speech_stays_context_without_changing_interviewer_question(self):
        qa, now, events = self.make_worker()
        qa.set_question_source('system')
        qa.feed('请介绍一下你的项目？', 'system')
        qa.feed('我负责后端接口。', 'microphone')
        qa.feed('我当时考虑怎么保证幂等性。', 'microphone')
        now[0] = 2
        qa.tick()
        self.assertEqual(events.get(timeout=2)[1][2], '请介绍一下你的项目？')
        self.assertIn('我负责后端接口。', qa.context)
        self.assertFalse(qa.pending)

    def test_both_sources_queue_independent_questions(self):
        qa, now, events = self.make_worker()
        qa.feed('为什么用消息队列？', 'system')
        qa.feed('缓存怎么失效？', 'microphone')
        qa.feed('请结合订单场景。', 'system')
        now[0] = 2
        qa.tick()
        self.assertEqual(events.get(timeout=2)[1][2], '为什么用消息队列？\n请结合订单场景。')
        events.get(timeout=2)
        events.get(timeout=2)
        qa.active = False
        qa.tick()
        self.assertEqual(events.get(timeout=2)[1][2], '缓存怎么失效？')

    def test_long_prompt_waits_for_final_speech_and_stable_transcript(self):
        qa, now, events = self.make_worker()
        qa.feed('解释一下分布式事务')
        for second in range(1, 10):
            now[0] = second
            qa.feed('还有一个补充条件。')
            qa.tick()
            self.assertFalse(qa.active)
        qa.observe_voice('system', True)
        now[0] = 25
        qa.tick()
        self.assertTrue(events.empty())
        qa.feed('最后说明失败恢复方案。')
        qa.observe_voice('system', False)
        now[0] += 1
        qa.tick()
        self.assertTrue(events.empty())
        now[0] += .6
        qa.tick()
        question = events.get(timeout=2)[1][2]
        self.assertTrue(question.startswith('解释一下分布式事务'))
        self.assertTrue(question.endswith('最后说明失败恢复方案。'))

    def test_candidate_activity_does_not_block_interviewer_question(self):
        qa, now, events = self.make_worker()
        qa.set_question_source('system')
        qa.observe_voice('microphone', True)
        qa.feed('事务怎么回滚？')
        now[0] = 2
        qa.tick()
        self.assertEqual(events.get(timeout=2)[1][2], '事务怎么回滚？')

    def test_changing_question_source_discards_old_voice_requests(self):
        qa, now, events = self.make_worker()
        qa.feed('为什么？')
        now[0] = 2
        qa.tick()
        cancellation = qa.cancel
        qa.feed('接下来讲一下缓存。')
        qa.set_question_source('microphone')
        self.assertTrue(cancellation.is_set())
        self.assertFalse(qa.active)
        self.assertFalse(qa.pending)
        qa.feed('系统声音不再作为问题？', 'system')
        self.assertFalse(qa.pending)

    def test_long_question_keeps_opening_and_last_condition_within_budget(self):
        qa, now, events = self.make_worker()
        qa.feed('请解释一下'+'开头'*600)
        for i in range(12):
            qa.feed('条件'+str(i)+'细节'*600)
        qa.feed('最后条件')
        now[0] = 2
        qa.tick()
        question = events.get(timeout=2)[1][2]
        self.assertLessEqual(len(question), 2400)
        self.assertTrue(question.startswith('请解释一下'))
        self.assertTrue(question.endswith('最后条件'))

    def test_question_source_change_keeps_manual_and_clipboard_requests(self):
        for source in ('manual', 'clipboard'):
            qa, now, events = self.make_worker()
            qa.ask('手动指定的问题', [], source=source)
            cancellation = qa.cancel
            qa.set_question_source('microphone')
            self.assertFalse(cancellation.is_set())
            self.assertTrue(qa.active)

    def test_spoken_requests_without_question_marks_trigger_auto_answer(self):
        examples = (
            '介绍一下', '说一说', '讲一下',
            '介绍一下你在视客的全栈开发经历。',
            '请介绍你自己的工作经历。',
            '请你简单地介绍一下负责的模块。',
            '请简单说一说你负责的项目。',
            '那你先讲一下 Cesium 的加载流程。',
            '接下来我想请你介绍一下做过的项目。',
            '聊聊你遇到的难点。',
            '谈一谈你的技术选型。',
            '解释一下为什么使用缓存。',
        )
        for utterance in examples:
            with self.subTest(utterance=utterance):
                self.assertTrue(is_question(utterance))
        for statement in ('我来介绍一下这个项目。', '我们先说一说背景。',
                          '他刚才讲了一下设计。', '这篇文章介绍了算法。'):
            with self.subTest(statement=statement):
                self.assertFalse(is_question(statement))

        now = [10.0]
        events = queue.Queue()
        qa = QAWorker(events, lambda prompt, cancel: '回答', lambda: now[0])
        qa.set_enabled(True)
        qa.feed('请简单说一说')
        qa.tick()
        self.assertTrue(events.empty())
        now[0] += 0.4
        qa.feed('你负责的项目。')
        now[0] += 2
        qa.tick()
        self.assertEqual(events.get(timeout=2)[1][1:], ('thinking', '请简单说一说\n你负责的项目。'))
        self.assertEqual(events.get(timeout=2)[1][1], 'answer')

    def test_question_detection_and_debounce_with_context(self):
        now = [10.0]
        events = queue.Queue()
        prompts = []
        def run(prompt, cancel):
            prompts.append(prompt)
            return "示例答案"
        qa = QAWorker(events, run, lambda: now[0])
        qa.feed("关闭时不能发送什么？")
        self.assertFalse(qa.context)
        qa.set_enabled(True)
        qa.feed("我们正在讨论机器学习。")
        qa.feed("什么是")
        qa.tick()
        self.assertTrue(events.empty())
        now[0] += 0.5
        qa.feed("大模型？")
        now[0] += 2
        qa.tick()
        self.assertEqual(events.get(timeout=2)[1][1], "thinking")
        result = events.get(timeout=2)[1]
        self.assertEqual(result[1], "answer")
        self.assertEqual(result[2], ("什么是\n大模型？", "示例答案"))
        data = json.loads(prompts[0].split("\n", 1)[1])
        self.assertIn("机器学习", data["背景转写"])
        self.assertEqual(data["当前问题"], "什么是\n大模型？")
        self.assertTrue(is_question("Whisper 是免费的吗。"))
        self.assertFalse(is_question("这是一个语音模型。"))

    def test_cancellation_drops_answer_and_new_generation(self):
        events, started, release = queue.Queue(), threading.Event(), threading.Event()
        def run(prompt, cancel):
            started.set()
            release.wait(2)
            return "旧答案"
        qa = QAWorker(events, run)
        qa.set_enabled(True)
        qa.feed("为什么？")
        qa.tick(force=True)
        self.assertTrue(started.wait(2))
        old = qa.generation
        qa.set_enabled(False)
        release.set()
        self.assertEqual(events.get(timeout=2)[1][1], "thinking")
        result = events.get(timeout=2)[1]
        self.assertEqual(result[:2], (old, "done"))
        self.assertNotEqual(old, qa.generation)
        self.assertFalse(qa.active)
        self.assertFalse(qa.pending)

    def test_duplicate_question_and_bounded_context(self):
        events = queue.Queue()
        qa = QAWorker(events, lambda p, c: "答案")
        qa.set_enabled(True)
        qa.feed("为什么天空是蓝色？")
        qa.tick(force=True)
        for _ in range(3):
            events.get(timeout=2)
        qa.active = False
        qa.feed("为什么天空是蓝色？")
        qa.pending['system'].updated -= 10
        qa.tick()
        self.assertTrue(events.empty())
        for _ in range(30):
            qa.feed("背景。" * 1000)
        self.assertEqual(len(qa.context), 12)
        self.assertLessEqual(len(qa.context[-1]), 1200)


if __name__ == "__main__":
    unittest.main()
