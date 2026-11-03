"""Interrupted transcript stays reviewable and cannot become an automatic half-question."""
import json
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import patch
from app import export_text
from paragraphs import ParagraphAssembler
from qa_worker import QAWorker
from transcript_corrections import load_recording,read_rows
from tests import test_voice_queue as fixtures


class IncompleteVoiceTests(unittest.TestCase):
    def worker(self):
        events=queue.Queue();worker=QAWorker(events,clock=lambda:10.)
        worker.set_enabled(True)
        return worker,events

    def test_partial_cancels_only_same_source_pending_and_keeps_complete_queue(self):
        worker,events=self.worker()
        worker.feed('什么是RAG？',record_key=0);worker.collect_voice(force=True)
        worker.feed('请比较一下微调在',record_key=1)
        worker.feed('为什么数据库','microphone',record_key=2)
        worker.feed('一个场景中',record_key=3,complete=False)
        self.assertEqual(set(worker.pending),{'microphone'})
        self.assertEqual(worker.voice_queue[0].question.text(),'什么是RAG？')
        self.assertNotIn('一个场景中',worker.context)
        self.assertEqual(events.get_nowait()[0],'qa_notice')

    def test_interruption_keeps_queued_turns_and_active_manual_request_state(self):
        worker,_=self.worker()
        worker.feed('什么是RAG？');worker.collect_voice(force=True)
        worker.feed('为什么向量','system');worker.feed('如何微调','microphone')
        worker.active=True;worker.source='manual';generation=worker.generation;cancel=worker.cancel
        worker.interrupt_voice()
        self.assertFalse(worker.pending);self.assertEqual(len(worker.voice_queue),1)
        self.assertTrue(worker.active);self.assertIs(worker.cancel,cancel)
        self.assertFalse(cancel.is_set());self.assertEqual(worker.generation,generation)

    def test_partial_cannot_be_resurrected_by_force_or_correction(self):
        worker,events=self.worker()
        worker.feed('为什么RAG在',record_key=1,complete=False)
        worker.correct_transcript(1,'为什么微调在')
        worker.tick(force=True)
        self.assertFalse(worker.pending or worker.voice_queue or worker.active or worker.context)
        self.assertFalse(any(k=='qa' for k,v in events.queue))

    def test_candidate_partial_does_not_cancel_interviewer_question_or_emit_notice(self):
        worker,events=self.worker();worker.set_question_source('system')
        worker.feed('为什么RAG在','system')
        worker.feed('我使用了','microphone',complete=False)
        self.assertEqual(set(worker.pending),{'system'});self.assertTrue(events.empty())

    def test_disabled_partial_does_not_mutate_worker_or_emit_notice(self):
        worker,events=self.worker();worker.set_enabled(False);revision=worker.context_revision
        worker.feed('为什么RAG在',complete=False)
        self.assertEqual(worker.context_revision,revision);self.assertTrue(events.empty())

    def test_marker_survives_paragraph_merge_source_switch_and_recording_reload(self):
        paragraphs=ParagraphAssembler()
        paragraphs.consume(dict(start=0,end=1,text='前半句',source='system'))
        paragraphs.consume(dict(start=1,end=2,text='后半句',source='system',incomplete=True))
        emitted=paragraphs.consume(dict(start=2,end=3,text='另一方',source='microphone'))
        self.assertEqual(emitted,[dict(start=0,end=2,text='前半句后半句',source='system',incomplete=True)])
        self.assertNotIn('incomplete',paragraphs.finish()[0])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.jsonl';path.write_text(json.dumps(emitted[0],ensure_ascii=False)+'\n',encoding='utf-8')
            self.assertEqual(load_recording(path).rows,emitted)
            self.assertIn('转录中断',export_text(emitted))
            self.assertNotIn('转录中断',export_text(emitted,True))

    def test_invalid_interruption_markers_are_rejected_but_legacy_and_false_are_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.jsonl'
            row=dict(start=0,end=1,text='旧记录')
            path.write_text(json.dumps(row),encoding='utf-8');self.assertEqual(read_rows(path),[row])
            for value in (False,True):
                row['incomplete']=value;path.write_text(json.dumps(row),encoding='utf-8')
                self.assertIs(read_rows(path)[0]['incomplete'],value)
            for value in (0,1,'false',None,[]):
                row['incomplete']=value;path.write_text(json.dumps(row),encoding='utf-8')
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'中断标记'):read_rows(path)


class IncompleteVoiceAppTests(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.VoiceQueueAppTests();self.case.setUp();self.app=self.case.app
    def tearDown(self):
        self.case.tearDown();self.case.doCleanups()
    def partial(self):
        self.app.events.put(('segment',dict(start=0,end=1,text='请比较一下 RAG 和微调在',source='system',incomplete=True)))
        self.app.poll()

    def test_partial_is_saved_labelled_and_never_automatically_requested(self):
        self.partial();self.case.settle()
        self.assertFalse(self.case.requests)
        self.assertIs(json.loads(self.app.session.read_text('utf-8'))['incomplete'],True)
        self.assertIn('转录中断',self.app.chat.bubbles[0].label.cget('text'))
        self.assertIn('转录中断',self.app.session.with_suffix('.txt').read_text('utf-8-sig'))
        self.assertIn('核对',self.app.qa_status.get())
        rows,_=self.app.session_context_snapshot();self.assertIn('转录中断',rows[0][0])

    def test_capture_error_clears_unsealed_half_question_but_dispatches_prior_complete_turn(self):
        self.app.qa_connection.state='unavailable'
        self.case.add('什么是RAG？');self.case.settle()
        self.app.qa.feed('请比较一下微调在')
        self.app.events.put(('error','模型响应失败'))
        with patch('app.messagebox.showerror'):self.app.poll()
        self.assertFalse(self.app.qa.pending);self.assertEqual(len(self.app.qa.voice_queue),1)
        self.app.qa_connection.state='verified';self.app.poll();self.case.finish()
        self.assertEqual([r[0]['当前问题'] for r in self.case.requests],['什么是RAG？'])

    def test_user_can_manually_ask_after_reviewing_interrupted_text(self):
        self.partial();self.case.settle();self.assertFalse(self.case.requests)
        self.app.bubble_click(0,False);self.case.finish()
        self.assertEqual([r[0]['当前问题'] for r in self.case.requests],['请比较一下 RAG 和微调在'])
        self.assertEqual(self.app.qa.source,'manual')

    def test_reload_and_correction_keep_original_interruption_marker_without_auto_requeue(self):
        self.partial();path=self.app.session;original=path.read_bytes()
        self.app.correct_transcript(0,'请比较一下 RAG 和微调在检索场景下的区别。')
        self.app.restore_session(path);self.case.settle()
        self.assertEqual(path.read_bytes(),original)
        self.assertIs(self.app.rows[0]['incomplete'],True)
        self.assertIn('转录中断',self.app.chat.bubbles[0].label.cget('text'))
        self.assertIn('已修正',self.app.chat.bubbles[0].label.cget('text'))
        self.assertFalse(self.case.requests)

    def test_next_complete_question_still_dispatches_and_marks_partial_background(self):
        self.partial();self.case.settle()
        self.case.add('为什么需要向量索引？');self.case.settle();self.case.finish()
        self.assertEqual([r[0]['当前问题'] for r in self.case.requests],['为什么需要向量索引？'])
        self.assertIn('转录中断',json.dumps(self.case.requests[0][0]['会话资料'],ensure_ascii=False))
