import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

from tokenizers import Tokenizer, models

from whisper_runtime import WhisperCaptureModel


class WhisperOwnershipTests(unittest.TestCase):
    def make_model(self, *, fail_load=False, fail_decode=False, fail_close=False,
                   decode_started=None, allow_decode=None):
        events=[]
        refs=[]
        class Native:
            def __init__(self):
                events.append(('load',threading.get_ident()))
                self.model=SimpleNamespace(device='cuda',unload_model=self.unload)
                self.max_length=448
                self.hf_tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1},unk_token='[UNK]'))
                refs.append(weakref.ref(self))
            def transcribe(self, audio, **options):
                events.append(('transcribe',threading.get_ident()))
                def segments():
                    events.append(('iterate',threading.get_ident()))
                    if decode_started is not None:
                        decode_started.set()
                        if not allow_decode.wait(timeout=3):
                            raise RuntimeError('Test decode was not released')
                    if fail_decode:
                        raise ValueError('decode failed')
                    yield SimpleNamespace(text='hello',start=0.,end=1.,words=[SimpleNamespace(word='hello',start=0.,end=1.)])
                return segments(),SimpleNamespace(language='en',duration=1.)
            def unload(self):
                events.append(('unload',threading.get_ident()))
                # Break this fake's bound-method cycle, matching the real native
                # backend which does not point back to its Python WhisperModel.
                self.model.unload_model=None
                if fail_close:
                    raise ValueError('unload failed')
            def __del__(self):
                events.append(('destroy',threading.get_ident()))
        def factory():
            model=Native()
            if fail_load:
                model.model.unload_model=None
                raise ValueError('load failed')
            return model
        return factory,events,refs

    def test_model_and_lazy_decode_are_created_used_and_destroyed_on_one_owner(self):
        factory,events,refs=self.make_model()
        adapter=WhisperCaptureModel(factory)
        self.addCleanup(adapter.close)
        owner=adapter._thread.ident
        self.assertNotEqual(owner,threading.get_ident())
        self.assertEqual(adapter.model.device,'cuda')
        self.assertEqual(adapter.max_length,448)
        self.assertIsInstance(adapter.hf_tokenizer,Tokenizer)
        self.assertEqual(adapter.hf_tokenizer.encode('hello',add_special_tokens=False).ids,[1])
        segments,info=adapter.transcribe([0.],language='en')
        self.assertEqual(next(segments).words[0].word,'hello')
        self.assertEqual(info.language,'en')
        adapter.close()
        self.assertIsNone(refs[0]())
        self.assertEqual([kind for kind,_ in events],['load','transcribe','iterate','unload','destroy'])
        self.assertEqual({ident for _,ident in events},{owner})
        self.assertFalse(adapter._thread.is_alive())

    def test_separate_capture_threads_reuse_the_same_warm_model(self):
        factory,events,refs=self.make_model()
        holders=[]
        thread=threading.Thread(target=lambda: holders.append(WhisperCaptureModel(factory)))
        thread.start();thread.join(timeout=3)
        adapter=holders.pop()
        self.addCleanup(adapter.close)
        outputs=[]
        def capture():
            outputs.append(''.join(segment.text for segment in adapter.transcribe([0.])[0]))
        for _ in range(2):
            thread=threading.Thread(target=capture)
            thread.start();thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        adapter.close()
        self.assertEqual(outputs,['hello','hello'])
        self.assertEqual(sum(kind=='load' for kind,_ in events),1)
        self.assertEqual({ident for _,ident in events},{adapter._thread.ident})
        self.assertIsNone(refs[0]())

    def test_decode_error_does_not_export_native_worker_frames_to_caller(self):
        factory,events,refs=self.make_model(fail_decode=True)
        adapter=WhisperCaptureModel(factory)
        self.addCleanup(adapter.close)
        with self.assertRaisesRegex(RuntimeError,'^decode failed$') as caught:
            adapter.transcribe([0.])
        adapter.close()
        self.assertIsNone(refs[0]())
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(caught.exception.worker_error_type,'ValueError')
        self.assertEqual({ident for _,ident in events},{adapter._thread.ident})

    def test_failed_startup_drops_partial_model_on_loading_thread(self):
        factory,events,refs=self.make_model(fail_load=True)
        with self.assertRaisesRegex(RuntimeError,'^load failed$'):
            WhisperCaptureModel(factory)
        self.assertIsNone(refs[0]())
        self.assertEqual([kind for kind,_ in events],['load','destroy'])
        self.assertNotEqual(events[0][1],threading.get_ident())
        self.assertEqual(events[0][1],events[-1][1])

    def test_close_failure_still_destroys_model_and_joins_owner(self):
        factory,events,refs=self.make_model(fail_close=True)
        adapter=WhisperCaptureModel(factory)
        with self.assertRaisesRegex(RuntimeError,'^unload failed$'):
            adapter.close()
        self.assertTrue(adapter.closed)
        self.assertFalse(adapter._thread.is_alive())
        self.assertIsNone(refs[0]())
        adapter.close()
        self.assertEqual(sum(kind=='unload' for kind,_ in events),1)
        with self.assertRaisesRegex(RuntimeError,'已关闭'):
            adapter.transcribe([0.])

    def test_close_waits_for_active_lazy_inference_before_releasing_model(self):
        entered,release,closed=threading.Event(),threading.Event(),threading.Event()
        factory,events,refs=self.make_model(decode_started=entered,allow_decode=release)
        adapter=WhisperCaptureModel(factory)
        outputs=[]
        client=threading.Thread(target=lambda: outputs.extend(adapter.transcribe([0.])[0]))
        client.start()
        self.assertTrue(entered.wait(timeout=3))
        closer=threading.Thread(target=lambda: (adapter.close(),closed.set()))
        closer.start()
        self.assertFalse(closed.wait(timeout=.05))
        self.assertFalse(any(kind=='unload' for kind,_ in events))
        release.set()
        client.join(timeout=3);closer.join(timeout=3)
        self.assertFalse(client.is_alive());self.assertFalse(closer.is_alive())
        self.assertTrue(closed.is_set())
        self.assertEqual([row.text for row in outputs],['hello'])
        self.assertIsNone(refs[0]())

    def test_interrupted_startup_schedules_cleanup_before_join(self):
        from concurrent.futures import Future
        class InterruptedReady(Future):
            def result(self, *args, **kwargs):
                raise KeyboardInterrupt()
        factory,events,refs=self.make_model()
        with patch('whisper_runtime.Future',InterruptedReady):
            with self.assertRaises(KeyboardInterrupt):
                WhisperCaptureModel(factory)
        self.assertIsNone(refs[0]())
        self.assertEqual([kind for kind,_ in events],['load','unload','destroy'])
        self.assertEqual(len({ident for _,ident in events}),1)

    def test_capture_factory_keeps_qwen_on_its_existing_process_adapter(self):
        from model_runtime import load_capture_model
        from qwen_engine import QWEN_MODEL
        with patch('model_runtime.load_model') as load,patch('whisper_runtime.WhisperCaptureModel') as owner:
            self.assertIs(load_capture_model(QWEN_MODEL),load.return_value)
        load.assert_called_once_with(QWEN_MODEL,device='auto',status=None)
        owner.assert_not_called()

    def test_packaged_owner_probe_uses_distinct_clients_and_finishes_cleanup(self):
        from launcher import verify_capture_owner
        factory,events,refs=self.make_model()
        with patch('model_runtime.load_capture_model',side_effect=lambda *args,**kwargs: WhisperCaptureModel(factory)):
            result=verify_capture_owner('cuda')
        self.assertEqual(result,dict(backend='cuda',distinct_capture_clients=True,persistent_owner=True,
            owner_thread_exited=True,tokenizer_budget_available=True))
        self.assertIsNone(refs[0]())
        self.assertEqual(len({ident for _,ident in events}),1)

    def test_exit_finishes_window_cleanup_if_model_release_reports_an_error(self):
        from app import App
        from unittest.mock import Mock
        app=App.__new__(App)
        app.busy=False
        app.root=Mock()
        app.engine=Mock()
        app.engine.unload_model.side_effect=RuntimeError('release failed')
        with patch('app.logging.exception') as log:
            app.close_when_done()
        app.engine.unload_model.assert_called_once()
        app.root.destroy.assert_called_once()
        log.assert_called_once()

    def test_retained_adapter_is_closed_before_normal_interpreter_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder)/'closed.json'
            code='''import json,sys,threading
from pathlib import Path
from types import SimpleNamespace
from whisper_runtime import WhisperCaptureModel
events=[]
class Native:
    def __init__(self):
        events.append(threading.get_ident())
        self.model=SimpleNamespace(device='cuda',unload_model=self.unload)
    def unload(self):
        events.append(threading.get_ident())
        Path(sys.argv[1]).write_text(json.dumps(events))
        self.model.unload_model=None
adapter=WhisperCaptureModel(Native)
'''
            result=subprocess.run([sys.executable,'-c',code,str(report)],capture_output=True,text=True,
                cwd=Path(__file__).resolve().parents[1],timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            events=json.loads(report.read_text())
            self.assertEqual(len(events),2)
            self.assertEqual(events[0],events[1])


if __name__=='__main__':
    unittest.main()
