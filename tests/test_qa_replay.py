import gc
import json
import queue
from pathlib import Path
import tempfile
import threading
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from app import App
from qa_records import load_qa
from qa_worker import QAWorker


class QAReplayTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.folder=tempfile.TemporaryDirectory()
        self.path=Path(self.folder.name)/'interview.jsonl'
        self.path.write_text(json.dumps(dict(start=0,end=1,text='什么是幂等性？',source='system'),ensure_ascii=False)+'\n',encoding='utf-8')
        self.journal=self.path.with_suffix('.qa.jsonl')
        for name,options in (
            ('app.GlobalHotkey',{}),('app.App.start_tray',{}),('app.ClipboardWatcher',{}),
            ('qa_connection.QAConnection.check',{}),('app.save_desktop',{}),
            ('app.preferences',{'return_value':{}}),
            ('app.load_settings',{'return_value':{'provider':'compatible','profiles':{}}})):
            item=patch(name,**options)
            item.start()
            self.addCleanup(item.stop)
        self.root=tk.Tk()
        self.root.withdraw()
        self.app=App(self.root)

    def tearDown(self):
        self.app.qa.set_enabled(False)
        for worker in threading.enumerate():
            if worker.name=='qa-worker': worker.join(timeout=3)
        self.app.qa.context_provider=self.app.qa.voice_context_provider=None
        self.app.qa_connection.close()
        self.app.gpu_check.close()
        self.app.hotkey.close()
        for timer in self.root.tk.call('after','info'): self.root.after_cancel(timer)
        self.root.destroy()
        self.folder.cleanup()
        self.app=self.root=None
        gc.collect()

    def seed(self, pairs):
        self.journal.write_text(''.join(json.dumps(dict(question=q,answer=a,time='旧时间'),ensure_ascii=False)+'\n'
                                        for q,a in pairs),encoding='utf-8')

    def replace(self,index,question,answer):
        with patch.object(self.app.qa,'ask',side_effect=lambda *_:self.app.qa.reset()):
            self.app.replace_question(index,question,[])
        generation=self.app.qa.generation
        for state,payload in [('thinking',question),('partial',(question,answer[:1])),('answer',(question,answer)),('done',None)]:
            self.app.handle_qa((generation,state,payload))

    def test_restore_legacy_history_without_resubmission_or_file_migration(self):
        pairs=[('什么是幂等性？','同一操作重复执行结果一致。'),('举个例子？','支付回调使用唯一约束。')]
        self.seed(pairs)
        before=(self.path.read_bytes(),self.journal.read_bytes())
        with patch.object(self.app.qa,'ask') as ask,patch.object(self.app.qa,'runner') as runner:
            self.app.restore_session(self.path)
            self.app.poll()
            ask.assert_not_called();runner.assert_not_called()
        self.assertEqual(self.app.qa_history,pairs)
        self.assertEqual(self.app.qa_display_history,pairs[:-1])
        self.assertEqual((self.app.qa_question,self.app.qa_answer),pairs[-1])
        self.assertTrue(self.app.qa_enabled.get())
        self.assertFalse(self.app.auto_qa.get())
        self.assertFalse(self.app.qa.voice_queue)
        self.assertIn('支付回调使用唯一约束',self.app.qa_text.get('1.0','end'))
        self.assertEqual((self.path.read_bytes(),self.journal.read_bytes()),before)
        self.assertIn('2 条历史问答',self.app.qa_status.get())
        self.assertEqual(self.app.session_context_snapshot()[1],pairs)

    def test_duplicate_legacy_questions_and_answers_update_only_clicked_turn(self):
        self.seed([('为什么？','相同回答'),('为什么？','相同回答')])
        self.app.restore_session(self.path)
        original=self.path.read_bytes()
        self.replace(0,'第一条为什么？','只修改第一条回答')
        self.assertEqual(self.app.qa_history,[('第一条为什么？','只修改第一条回答'),('为什么？','相同回答')])
        self.assertEqual(self.app.qa_display_history,[('第一条为什么？','只修改第一条回答')])
        self.assertEqual((self.app.qa_question,self.app.qa_answer),('为什么？','相同回答'))
        self.assertEqual([(r.question,r.answer) for r in load_qa(self.journal)],self.app.qa_history)
        self.assertEqual(self.path.read_bytes(),original)
        self.app.restore_session(self.path)
        self.assertEqual(self.app.qa_history[0],('第一条为什么？','只修改第一条回答'))

    def test_edit_old_recording_after_new_recording_writes_to_original_journal(self):
        self.seed([('原录音的问题','原录音的回答')])
        self.app.restore_session(self.path)
        other=self.path.with_name('new-recording.jsonl')
        other.write_text('',encoding='utf-8')
        self.app.session=other
        self.replace(0,'原问题修订','修订后的旧回答')
        self.assertEqual(load_qa(self.journal)[0].answer,'修订后的旧回答')
        self.assertFalse(other.with_suffix('.qa.jsonl').exists())

    def test_new_answers_have_distinct_ids_and_can_replay_and_edit_repeatedly(self):
        self.app.session=self.path
        self.app.open_qa()
        generation=self.app.qa.generation
        for _ in range(2):
            self.app.handle_qa((generation,'thinking','为什么？'))
            self.app.handle_qa((generation,'answer',('为什么？','相同回答')))
        first,second=load_qa(self.journal)
        self.assertNotEqual(first.key,second.key)
        self.replace(0,'改第一条','新一答')
        self.replace(0,'再次改第一条','又一答')
        self.assertEqual([r.answer for r in load_qa(self.journal)],['又一答','相同回答'])
        self.assertEqual(self.app.qa_history,[('再次改第一条','又一答'),('为什么？','相同回答')])

    def test_external_change_is_not_overwritten_or_appended_as_a_new_turn(self):
        self.seed([('旧问题','旧答案')])
        self.app.restore_session(self.path)
        self.seed([('外部更新的问题','外部更新的答案')])
        newer=self.journal.read_bytes()
        self.replace(0,'本机修订问题','本机新答案')
        self.assertEqual(self.journal.read_bytes(),newer)
        self.assertIn('保存失败',self.app.qa_status.get())
        self.assertIn('本机新答案',self.app.qa_text.get('1.0','end'))

    def test_corrupt_qa_keeps_transcript_available_and_blocks_further_journal_writes(self):
        self.journal.write_bytes(b'{bad json')
        self.app.restore_session(self.path)
        self.assertEqual(self.app.rows[0]['text'],'什么是幂等性？')
        self.assertFalse(self.app.qa_history)
        self.assertIn('历史问答未载入',self.app.status.get())
        self.app.clear(preserve_recording=True)
        self.app.open_qa()
        self.app.handle_qa((self.app.qa.generation,'answer',('新问题','新回答')))
        self.assertEqual(self.journal.read_bytes(),b'{bad json')
        self.assertIn('保存失败',self.app.qa_status.get())
        self.seed([('修复的问题','修复的答案')])
        self.app.restore_session(self.path)
        self.assertFalse(self.app.qa_unreadable_paths)

    def test_opening_transcript_without_qa_clears_previous_question_ids_and_images(self):
        self.seed([('旧问题','旧回答')])
        self.app.restore_session(self.path)
        other=self.path.with_name('empty-qa.jsonl')
        other.write_bytes(self.path.read_bytes())
        self.app.restore_session(other)
        self.assertFalse(self.app.qa_history)
        self.assertFalse(self.app.qa_turn_ids)
        self.assertFalse(self.app.qa_history_ids)
        self.assertFalse(self.app.qa_origins)
        self.assertFalse(self.app.qa_display_images)
        self.assertEqual(self.app.qa_question,'')

    def test_failed_question_revision_can_complete_as_a_new_identified_exchange(self):
        self.app.session=self.path
        self.app.open_qa()
        self.app.qa_question,self.app.qa_answer='之前未完成的问题','部分答案'
        self.replace(0,'重新提问','完整答案')
        self.assertEqual(self.app.qa_history,[('重新提问','完整答案')])
        self.assertEqual(load_qa(self.journal)[0].answer,'完整答案')

    def test_new_transcript_selection_preserves_previous_answer_and_has_its_own_identity(self):
        self.seed([('旧问题','旧答案')])
        self.app.restore_session(self.path)
        self.app.toggle_question_selection(0)
        self.assertEqual(self.app.qa_display_history,[('旧问题','旧答案')])
        self.assertNotIn(1,self.app.qa_turn_ids)
        self.replace(1,'转录里的新问题','新的回答')
        self.assertEqual(self.app.qa_history,[('旧问题','旧答案'),('转录里的新问题','新的回答')])
        self.assertEqual([r.answer for r in load_qa(self.journal)],['旧答案','新的回答'])

    def test_cancelled_revision_cannot_write_a_late_answer(self):
        self.seed([('旧问题','旧答案')])
        self.app.restore_session(self.path)
        before=self.journal.read_bytes()
        with patch.object(self.app.qa,'ask',side_effect=lambda *_:self.app.qa.reset()):
            self.app.replace_question(0,'未完成修订',[])
        stale=self.app.qa.generation
        self.app.qa.cancel_request()
        self.app.handle_qa((stale,'answer',('未完成修订','迟到的回答')))
        self.assertEqual(self.journal.read_bytes(),before)
        self.app.restore_session(self.path)
        self.assertEqual(self.app.qa_history,[('旧问题','旧答案')])

    def test_restored_history_supports_a_short_voice_followup_before_any_new_request(self):
        self.seed([('什么是幂等性？','相同订单的回调只处理一次。')])
        self.app.restore_session(self.path)
        self.app.auto_qa.set(True)
        self.app.toggle_auto_qa()
        self.app.qa_connection.state='verified'
        now=[0.]
        self.app.qa.clock=lambda:now[0]
        self.app.qa.stream_runner=None
        prompts=[]
        self.app.qa.runner=lambda prompt,cancel:prompts.append(json.loads(prompt.split('\n',1)[1])) or '可以用唯一约束实现。'
        self.assertIsNone(self.app.qa.source)
        self.app.events.put(('segment',dict(start=2,end=3,text='再具体一点。',source='system')))
        self.app.poll()
        now[0]=2
        self.app.poll()
        self.assertTrue(self.app.qa.active)
        while True:
            kind,value=self.app.events.get(timeout=3)
            if kind=='qa':
                self.app.handle_qa(value)
                if value[1]=='done': break
        self.assertEqual(prompts[0]['当前问题'],'再具体一点。')
        self.assertIn('相同订单的回调只处理一次',json.dumps(prompts[0],ensure_ascii=False))
        self.assertEqual(self.app.qa_history[-1],('再具体一点。','可以用唯一约束实现。'))


class QueuedFollowupTests(unittest.TestCase):
    def test_short_followup_queues_behind_a_question_before_service_has_started(self):
        now=[0.]
        worker=QAWorker(queue.Queue(),clock=lambda:now[0])
        worker.set_enabled(True)
        worker.feed('介绍一下幂等性。')
        now[0]=2
        worker.collect_voice()
        self.assertIsNone(worker.source)
        worker.feed('再具体一点。')
        now[0]=4
        worker.collect_voice()
        self.assertEqual([turn.question.text() for turn in worker.voice_queue],['介绍一下幂等性。','再具体一点。'])
