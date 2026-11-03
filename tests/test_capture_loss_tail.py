"""Known capture loss marks every unfinished source tail before source switches."""
import json
import queue
from types import SimpleNamespace as Item
import unittest
from unittest.mock import Mock,patch
import numpy as np
from app import Transcriber,export_text
from tests import test_voice_queue as voice_fixtures

PARTIAL={'system':'请比较一下 RAG 和微调在','microphone':'我使用 LoRA 训练时'}


def inject_capture(engine,mode,*,sources=('system','microphone'),pending_from_push=False,idle_stall=False):
    streams=[];segmenters=[];echoes=[]
    model=Item(model=Item(device='cpu'))
    class Stream:
        def __init__(self,options):self.options=options;self.closed=False
        def feed(self,flags=0):
            sound=np.full(1600,self.options['input_device_index']/10,np.float32)
            return self.options['stream_callback'](sound.tobytes(),len(sound),{},flags)
        def start_stream(self):self.feed()
        def is_active(self):return True
        def close(self):self.closed=True
    api=Mock();api.get_device_info_by_index.return_value=dict(defaultSampleRate=16000,maxInputChannels=1)
    def open_stream(**options):
        stream=Stream(options);streams.append(stream);return stream
    api.open.side_effect=open_stream
    class Echo:
        def __init__(self,output,stop,rates,convert):
            self.output,self.stop=output,stop;self.done=__import__('threading').Event()
            self.error=None;self.closed=False;echoes.append(self)
        def start(self):pass
        def submit(self,item):
            source,offset,samples,epoch=item
            self.output.put_nowait((source,offset,samples[:,0] if source=='microphone' else samples,epoch))
        def close(self):self.closed=True;self.done.set()
    def load(*args,**kwargs):
        if mode=='overflow':streams[0].feed()
        elif mode=='driver_flag':streams[0].feed(flags=1)
        elif mode=='echo_error':
            echoes[0].error='模拟回声处理失败';echoes[0].done.set();engine.stop()
        elif mode=='normal_stop':engine.stop()
        else:raise ValueError(mode)
        return model
    class Segmenter:
        def __init__(self):
            self.source=sources[len(segmenters)];self.parts=[];self.last_speech_end=None
            self.start=0.;self.epoch=1800000000.;segmenters.append(self)
        @property
        def pending_speech(self):return bool(self.parts)
        def push(self,samples,offset,epoch):
            self.last_speech_end=offset+len(samples)/16000
            if pending_from_push and self.source==sources[0]:
                return [Item(audio=samples,start=offset,epoch=epoch,source=self.source)]
            if not self.parts:self.start,self.epoch=offset,epoch
            self.parts.append(samples)
            if idle_stall and self.source==sources[-1]:
                __import__('time').sleep(1.)
            return []
        def finish(self):
            if not self.parts:return []
            sound=np.concatenate(self.parts);self.parts=[]
            return [Item(audio=sound,start=self.start,epoch=self.epoch,source=self.source)]
    def decode(actual,assembler,chunk,language,guidance,backlog):
        assert actual is model
        return [(chunk.start,chunk.start+len(chunk.audio)/16000,PARTIAL[chunk.source],chunk.epoch)]
    real_queue=queue.Queue
    with patch('app.pa.PyAudio',return_value=api),patch('app.load_model',side_effect=load), \
            patch('app.PauseSegmenter',Segmenter),patch('app.decode_chunk',side_effect=decode), \
            patch('app.DraftPreview') as preview,patch('echo_cancellation.EchoCapture',Echo), \
            patch('app.queue.Queue',side_effect=lambda **kw:real_queue(maxsize=len(sources) if kw.get('maxsize')==1200 else kw.get('maxsize',0))):
        preview.return_value.render.return_value=None
        engine.run([dict(index=1 if s=='system' else 2,source=s) for s in sources],
                   'capture-loss-fixture','zh',echo_cancellation=mode=='echo_error')
    assert all(s.closed for s in streams) and all(e.closed for e in echoes)
    api.terminate.assert_called_once()


class CaptureLossTailTests(unittest.TestCase):
    def exercise(self,mode,**options):
        events=queue.Queue();inject_capture(Transcriber(events),mode,**options)
        messages=list(events.queue);rows=[r for k,r in messages if k=='segment']
        self.assertEqual({r['source'] for r in rows},set(options.get('sources',('system','microphone'))))
        self.assertEqual(sum(k=='done' for k,_ in messages),1)
        if mode=='normal_stop':
            self.assertFalse(any(r.get('incomplete') for r in rows))
            self.assertFalse(any(k=='error' for k,_ in messages))
        else:
            self.assertTrue(all(r.get('incomplete') is True for r in rows),rows)
            self.assertEqual(sum(k=='error' for k,_ in messages),1)
        return rows

    def test_queue_overflow_marks_both_final_sources(self):self.exercise('overflow')
    def test_driver_discontinuity_marks_both_final_sources(self):self.exercise('driver_flag')
    def test_echo_failure_marks_both_final_sources(self):self.exercise('echo_error')
    def test_reverse_source_order_keeps_both_markers(self):
        self.exercise('overflow',sources=('microphone','system'))
    def test_existing_pending_paragraph_is_marked_before_the_next_source_flushes(self):
        self.exercise('driver_flag',pending_from_push=True)
    def test_normal_stop_does_not_add_interruption_markers(self):self.exercise('normal_stop')
    def test_known_loss_during_slow_processing_marks_idle_source_flushes(self):
        for mode in ('overflow','driver_flag','echo_error'):
            with self.subTest(mode=mode):self.exercise(mode,idle_stall=True)
    def test_normal_stop_idle_source_flush_keeps_complete_markers(self):
        self.exercise('normal_stop',idle_stall=True)
    def test_single_source_loss_and_stop_keep_the_existing_behavior(self):
        self.exercise('overflow',sources=('system',));self.exercise('normal_stop',sources=('system',))
    def test_packaged_guard_checks_error_and_normal_source_finalization(self):
        from launcher import verify_capture_loss_tails
        self.assertEqual(verify_capture_loss_tails(),{'capture_loss_tail_guard':True})


class CaptureLossTailAppTests(unittest.TestCase):
    def setUp(self):
        self.case=voice_fixtures.VoiceQueueAppTests();self.case.setUp();self.app=self.case.app
        self.app.qa_connection.state='unavailable'
        self.case.add('什么是 RAG？');self.case.settle()
        self.assertEqual(len(self.app.qa.voice_queue),1)
    def tearDown(self):self.case.tearDown();self.case.doCleanups()
    def loss(self):
        inject_capture(self.app.engine,'overflow')
        with patch('app.messagebox.showerror'):self.app.poll()
        self.case.settle()
        saved=[json.loads(line) for line in self.app.session.read_text('utf-8').splitlines()]
        partials=[r for r in saved if r['text'] in PARTIAL.values()]
        self.assertEqual(len(partials),2);self.assertTrue(all(r.get('incomplete') is True for r in partials))
        self.assertNotIn('incomplete',saved[0])
        self.assertEqual(export_text(partials).count('转录中断'),2)
        context,_=self.app.session_context_snapshot()
        self.assertEqual(json.dumps(context,ensure_ascii=False).count('原转录中断'),2)
        self.assertEqual(len(self.app.qa.voice_queue),1);self.assertFalse(self.case.requests)
        self.app.qa_connection.state='verified';self.app.poll();self.case.finish()
        self.assertEqual([r[0]['当前问题'] for r in self.case.requests],['什么是 RAG？'])
    def test_saved_exported_context_keeps_both_markers_and_prior_complete_question(self):self.loss()
    def test_user_can_manually_ask_first_source_after_review(self):
        self.loss();self.app.bubble_click(1,False);self.case.finish()
        self.assertEqual([r[0]['当前问题'] for r in self.case.requests],['什么是 RAG？',PARTIAL['system']])
        self.assertEqual(self.app.qa.source,'manual')
