import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from app import Transcriber
from manual_cut import SideButtonHotkey
from segmentation import PauseSegmenter, WordAssembler, DraftPreview
from recognition import RecognitionContext
from paragraphs import ParagraphAssembler

class ManualCutTests(unittest.TestCase):
    def test_boundary_is_fifo_coalesced_and_rejects_full_or_stopped(self):
        e=Transcriber(queue.Queue())
        self.assertFalse(e.request_cut())
        e.capture_queue=queue.Queue(maxsize=2)
        e.capture_queue.put('before')
        self.assertTrue(e.request_cut())
        self.assertFalse(e.request_cut())
        self.assertEqual(e.capture_queue.get(),'before')
        self.assertIs(e.capture_queue.get(),e.cut_marker)
        e.cut_pending=False
        e.capture_queue.put(1);e.capture_queue.put(2)
        self.assertFalse(e.request_cut())
        self.assertEqual(e.capture_queue.qsize(),2)
        e.stop();self.assertFalse(e.request_cut())

    def test_both_sources_finish_and_continue_without_lost_audio(self):
        events=queue.Queue();e=Transcriber(events);streams=[];decoded=[]
        class Stream:
            def __init__(self,cb,value):self.cb=cb;self.value=value;self.closed=False
            def feed(self,value=None):
                a=np.full(1600,self.value if value is None else value,dtype=np.float32)
                self.cb(a.tobytes(),1600,{},0)
            def start_stream(self):self.feed()
            def is_active(self):return True
            def close(self):self.closed=True
        api=Mock();api.get_device_info_by_index.return_value={'defaultSampleRate':16000,'maxInputChannels':1}
        def open_stream(**kw):
            st=Stream(kw['stream_callback'],kw['input_device_index']/10);streams.append(st);return st
        api.open.side_effect=open_stream
        def load(*args,**kw):
            self.assertTrue(e.request_cut());self.assertFalse(e.request_cut())
            streams[0].feed(.3);streams[1].feed(.4)
            e.stop()
            return SimpleNamespace(model=SimpleNamespace(device='cuda'))
        def decode(model,assembler,chunk,*args):
            decoded.append(chunk.audio.copy())
            return [(chunk.start,chunk.start+len(chunk.audio)/16000,'段'+str(len(decoded)),chunk.epoch)]
        def segmenter(_):
            return PauseSegmenter(detector=lambda a:[{'start':0,'end':len(a)}] if len(a) else [])
        with patch('app.pa.PyAudio',return_value=api),patch('app.load_model',side_effect=load),patch('app.create_segmenter',side_effect=segmenter),patch('app.decode_chunk',side_effect=decode),patch('app.AudioLevelNormalizer') as levels,patch('app.DraftPreview') as previews:
            levels.return_value.process.side_effect=lambda x:x
            previews.return_value.render.return_value=None
            e.run([dict(index=1,source='system'),dict(index=2,source='microphone')],'test','zh')
        self.assertEqual([len(a) for a in decoded],[1600]*4)
        np.testing.assert_allclose([a[0] for a in decoded],[.1,.2,.3,.4])
        rows=list(events.queue);segments=[v for k,v in rows if k=='segment']
        self.assertEqual([v['text'] for v in segments],['段1','段2','段3','段4'])
        self.assertFalse(any(v.get('incomplete') for v in segments))
        boundary=next(i for i,(k,v) in enumerate(rows) if k=='manual_cut_done')
        self.assertEqual(sum(k=='segment' for k,v in rows[:boundary]),2)
        self.assertFalse(any(k=='error' for k,v in rows))
        self.assertTrue(all(x.closed for x in streams));self.assertFalse(e.cut_active.is_set())
        self.assertIsNone(e.capture_queue)

    def test_held_stable_words_flush_once_without_failure_flag(self):
        e=Transcriber(queue.Queue());a=WordAssembler();a.pending=(0.,.5,'保留词',1800000000.)
        seg=Mock();seg.finish.return_value=[]
        states={'system':dict(segmenter=seg,assembler=a,guidance=RecognitionContext(),preview=DraftPreview())}
        p=ParagraphAssembler();notify=Mock()
        e._manual_boundary(states,Mock(),p,notify)
        e._manual_boundary(states,Mock(),p,notify)
        rows=[v for k,v in e.events.queue if k=='segment']
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['text'],'保留词')
        self.assertNotIn('incomplete',rows[0]);self.assertIsNone(a.pending)

class SideButtonTests(unittest.TestCase):
    def test_only_selected_side_button_is_consumed_while_recording(self):
        events=queue.Queue();active=threading.Event();h=SideButtonHotkey(events,active,1)
        self.assertFalse(h.handle(0x20B,1))
        active.set();self.assertFalse(h.handle(0x20B,2));self.assertFalse(h.handle(0x20B,1,True))
        self.assertTrue(h.handle(0x20B,1));self.assertTrue(h.handle(0x20B,1))
        self.assertEqual(events.qsize(),1)
        active.clear();self.assertTrue(h.handle(0x20C,1));self.assertFalse(h.handle(0x20C,1))
        self.assertFalse(h.handle(0x20B,1))

    def test_native_mouse_hook_installs_and_releases_without_input(self):
        q=queue.Queue();h=SideButtonHotkey(q,threading.Event())
        try:
            h.start();kind,(ok,reason)=q.get(timeout=4)
            self.assertEqual(kind,'manual_mouse_status');self.assertTrue(ok,reason)
        finally:h.close()
        self.assertFalse(h.thread.is_alive())

if __name__=='__main__':unittest.main()
