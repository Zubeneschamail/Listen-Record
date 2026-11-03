"""Storage-error stops require review before new automatic spoken questions."""
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
from tests import test_voice_queue as fixtures

PARTIAL='请比较一下 RAG 和微调在'
FULL='什么是 RAG？'


class SaveStopQATests(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.VoiceQueueAppTests();self.case.setUp();self.app=self.case.app
        self.app.qa_connection.state='unavailable'

    def tearDown(self):self.case.tearDown();self.case.doCleanups()

    def save_error(self,text=PARTIAL,source='system'):
        with patch('app.save_segment',side_effect=OSError('injected save failure')):
            self.case.add(text,source)
        self.assertTrue(self.app.session_save_failed)
        self.assertIn('核对尾句后手动提问',self.app.qa_status.get())

    def stop_and_settle(self):
        self.app.events.put(('done',None));self.app.poll();self.case.settle()

    def drain(self):
        self.app.qa_connection.state='verified';self.app.poll()
        for _ in range(4):
            if not self.app.qa.active:break
            self.case.finish();self.app.poll()
        self.assertFalse(self.app.qa.active)
        return [row[0]['当前问题'] for row in self.case.requests]

    def test_error_clears_unsealed_question_and_does_not_requeue_late_tail(self):
        self.case.add(PARTIAL)
        self.assertTrue(self.app.qa.pending)
        self.save_error('部署时')
        self.case.add('中文与专业术语的')
        self.stop_and_settle()
        self.assertEqual(self.drain(),[])
        self.assertEqual([row['text'] for row in self.app.rows],[PARTIAL,'部署时','中文与专业术语的'])
        self.assertFalse(self.app.qa.pending or self.app.qa.voice_queue)

    def test_failed_row_itself_cannot_start_an_automatic_question(self):
        self.save_error();self.case.add('后续尾句');self.stop_and_settle()
        self.assertEqual(self.drain(),[])
        self.assertTrue(all('incomplete' not in row for row in self.app.rows))

    def test_prior_completed_queue_is_preserved_and_dispatches_once(self):
        self.case.add(FULL);self.case.settle()
        self.case.add(PARTIAL);self.save_error('部署时');self.stop_and_settle()
        self.assertEqual(self.drain(),[FULL])

    def test_both_source_pending_questions_and_post_stop_input_are_suppressed(self):
        self.app.qa.set_question_source('both')
        self.case.add(PARTIAL);self.case.add('为什么 LoRA 在','microphone')
        self.assertEqual(set(self.app.qa.pending),{'system','microphone'})
        self.save_error('部署时','microphone')
        self.case.add('两个音源的尾句','microphone');self.stop_and_settle()
        self.assertEqual(self.drain(),[])

    def test_active_manual_request_is_not_cancelled_by_save_failure(self):
        entered,release=threading.Event(),threading.Event()
        def stream(*args,**kwargs):
            entered.set();self.assertTrue(release.wait(5));return 'local answer'
        self.app.qa.stream_runner=stream
        self.app.qa.ask('手动问题',[])
        self.assertTrue(entered.wait(3))
        generation,cancel=self.app.qa.generation,self.app.qa.cancel
        try:
            self.save_error();self.stop_and_settle()
            self.assertTrue(self.app.qa.active)
            self.assertEqual(self.app.qa.generation,generation)
            self.assertIs(self.app.qa.cancel,cancel)
            self.assertFalse(cancel.is_set())
        finally:release.set();self.case.finish()

    def test_manual_review_remains_available_for_an_unsaved_row(self):
        self.save_error();self.stop_and_settle();self.assertEqual(self.drain(),[])
        self.app.bubble_click(0,False);self.case.finish()
        self.assertEqual([row[0]['当前问题'] for row in self.case.requests],[PARTIAL])
        self.assertEqual(self.app.qa.source,'manual')

    def test_explicit_normal_stop_keeps_automatic_voice_behavior(self):
        self.case.add(FULL);self.stop_and_settle()
        self.assertFalse(self.app.session_save_failed)
        self.assertEqual(self.drain(),[FULL])

    def test_new_recording_restores_automatic_question_collection(self):
        self.save_error();self.stop_and_settle();self.assertEqual(self.drain(),[])
        self.app.capture_mode.set('仅系统声音')
        self.app.devices=[dict(index=123,name='fixture')]
        self.app.device.configure(values=['fixture']);self.app.device.current(0)
        with patch('app.RECORDINGS',Path(self.case.folder.name)),patch('app.load_capture_preferences',return_value=''),patch.object(self.app.engine,'start'):
            self.app.start()
        self.assertFalse(self.app.session_save_failed)
        self.case.add(FULL);self.stop_and_settle()
        self.assertEqual(self.drain(),[FULL])

    def test_packaged_guard_uses_actual_save_error_and_feed_policies(self):
        import app
        from launcher import verify_storage_stop_qa
        self.assertEqual(verify_storage_stop_qa(app),{'storage_stop_qa_guard':True})
