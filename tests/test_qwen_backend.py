import gc
import json
import sys
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from qwen_alignment import aligned_words
from qwen_runtime import QwenModel


READY = dict(type='ready', protocol=1, device='cuda', warmup=True)


def command(body):
    return [sys.executable, '-u', '-c',
            'import os,sys,json,time,base64,hashlib\n'+body]


def fake_engine(body):
    return command('print(json.dumps(dict(type="starting",protocol=1,pid=os.getpid())),flush=True)\n'
                   +'print('+repr(json.dumps(READY))+',flush=True)\n'+body)


class QwenAlignmentTests(unittest.TestCase):
    def test_code_symbols_and_word_boundaries_preserve_original_text(self):
        text = '用 C++ 和 C#，再讲 .NET、Node.js test-time。'
        tokens = ['用','C','和','C','再','讲','NET','Nodejstesttime']
        alignment = [dict(text=t, start_time=i*.1, end_time=(i+1)*.1) for i,t in enumerate(tokens)]
        words = aligned_words(text, alignment, 1.)
        self.assertEqual(''.join(w['word'] for w in words), text)
        self.assertIn('C++', words[1]['word'])
        self.assertIn('C#', words[3]['word'])
        self.assertTrue(words[6]['word'].startswith('.NET'))

    def test_mismatched_or_truncated_alignment_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'match'):
            aligned_words('C#', [dict(text='Csharp',start_time=0.,end_time=.2)], 1.)

    def test_small_quantization_overshoot_is_bounded_but_large_error_is_rejected(self):
        word = dict(text='好',start_time=.1,end_time=1.04)
        self.assertEqual(aligned_words('好。',[word],1.)[0]['end'],1.)
        with self.assertRaisesRegex(ValueError, 'timeline'):
            aligned_words('好。',[dict(word,end_time=2.)],1.)
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            aligned_words('好。',[dict(word,start_time=float('nan'))],1.)


class QwenProcessTests(unittest.TestCase):
    def test_silence_never_reaches_a_hallucinating_engine(self):
        worker = fake_engine("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps(dict(type='error',id=r['id'],message='静音进入了模型')),flush=True)")
        with QwenModel(command=worker) as model:
            segments, _ = model.transcribe(np.zeros(16000,dtype=np.float32), vad_filter=False)
            self.assertEqual(list(segments), [])
            self.assertEqual(model._sequence, 0)

    def test_audio_is_verified_and_aligned_words_reach_the_existing_decoder(self):
        from recognition import RecognitionContext
        from segmentation import AudioChunk, WordAssembler, decode_chunk
        worker = fake_engine("for line in sys.stdin:\n r=json.loads(line)\n data=base64.b64decode(r['audio'])\n assert len(data)==r['samples']*4 and hashlib.sha256(data).hexdigest()==r['sha256']\n assert 'hotwords' not in r\n print(json.dumps(dict(type='result',id=r['id'],sha256=r['sha256'],text='密等性',words=[dict(word='密等性',start=.1,end=.4)])),flush=True)")
        with QwenModel(command=worker) as model, patch('faster_whisper.vad.get_speech_timestamps',return_value=[dict(start=0,end=8000)]):
            rows = decode_chunk(model, WordAssembler(), AudioChunk(np.ones(16000,dtype=np.float32)*.02,2.,100.),'zh',
                                RecognitionContext('密等性=幂等性'))
        self.assertEqual(rows, [(2.1,2.4,'幂等性',100.1)])

    def test_wrong_response_or_lost_text_closes_the_child(self):
        for response in ("dict(type='result',id=r['id']+1,sha256=r['sha256'],text='好',words=[])",
                         "dict(type='result',id=r['id'],sha256=r['sha256'],text='C#',words=[dict(word='C',start=0.,end=.1)])"):
            with self.subTest(response=response):
                model = QwenModel(command=fake_engine("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps("+response+"),flush=True)"))
                with patch('faster_whisper.vad.get_speech_timestamps',return_value=[dict(start=0,end=8000)]):
                    with self.assertRaises(RuntimeError):
                        model.transcribe(np.ones(16000,dtype=np.float32)*.02)
                self.assertIsNotNone(model.process.poll())

    def test_inference_timeout_terminates_the_child(self):
        model = QwenModel(command=fake_engine("for line in sys.stdin: time.sleep(10)"),inference_timeout=.1)
        handle = kernel = None
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
            kernel.WaitForSingleObject.restype=wintypes.DWORD
            kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
            handle=kernel.OpenProcess(0x00100000,False,model.worker_pid)
            self.assertTrue(handle)
        with patch('faster_whisper.vad.get_speech_timestamps',return_value=[dict(start=0,end=8000)]):
            with self.assertRaisesRegex(RuntimeError,'超时'):
                model.transcribe(np.ones(16000,dtype=np.float32)*.02)
        self.assertIsNotNone(model.process.poll())
        if handle:
            try:self.assertEqual(kernel.WaitForSingleObject(handle,0),0)
            finally:kernel.CloseHandle(handle)

    def test_engine_cannot_claim_the_desktop_pid(self):
        worker=command('print(json.dumps(dict(type="starting",protocol=1,pid='+str(__import__('os').getpid())+')),flush=True);time.sleep(10)')
        with self.assertRaisesRegex(RuntimeError,'标识'):
            QwenModel(command=worker)

    def test_unloaded_model_does_not_leave_a_child_running(self):
        model = QwenModel(command=fake_engine('for line in sys.stdin: pass'))
        process = model.process
        del model
        gc.collect()
        self.assertIsNotNone(process.poll())

    def test_uninitialized_or_crashed_child_is_not_accepted(self):
        for body in ('pass', "print('not JSON',flush=True)", "print('{}',flush=True)"):
            with self.subTest(body=body):
                with self.assertRaises(RuntimeError):
                    QwenModel(command=command(body),startup_timeout=.2)

    def test_nonfinite_audio_is_rejected_before_submission(self):
        with QwenModel(command=fake_engine('for line in sys.stdin: pass')) as model:
            with self.assertRaises(ValueError):
                model.transcribe(np.array([float('nan')],dtype=np.float32))
            self.assertEqual(model._sequence,0)

    @unittest.skipUnless(sys.platform=='win32','Windows parent-process lifetime check')
    def test_hard_killed_parent_does_not_leave_an_engine_running(self):
        import ctypes
        from ctypes import wintypes
        child_script = ('import os,sys,json,ctypes;'
            'print(json.dumps(dict(type="starting",protocol=1,pid=os.getpid())),flush=True);'
            'print(json.dumps(dict(type="ready",protocol=1,device="cuda",warmup=True)),flush=True);'
            'sys.stdin.readline();print(json.dumps(dict(type="blocking")),flush=True);'
            'ctypes.PyDLL("kernel32").Sleep(30000)')
        parent_script = ('import subprocess,sys,os;'
            'from qwen_runtime import QwenModel;'
            f'model=QwenModel(command=[sys.executable,"-u","-c",{child_script!r}]);'
            'model.process.stdin.write("block\\n");model.process.stdin.flush();'
            'assert model._receive(2)["type"]=="blocking";print(model.worker_pid,flush=True);'
            'sys.stdin.readline();os._exit(0)')
        parent = subprocess.Popen([sys.executable,'-u','-c',parent_script],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        handle=None
        try:
            pid=int(parent.stdout.readline())
            handle=kernel.OpenProcess(0x00100000,False,pid)
            self.assertTrue(handle)
            parent.stdin.write('exit\n');parent.stdin.flush()
            parent.wait(timeout=2)
            self.assertEqual(kernel.WaitForSingleObject(handle,3000),0)
        finally:
            if handle:kernel.CloseHandle(handle)
            if parent.poll() is None:parent.terminate();parent.wait(timeout=2)
            parent.stdin.close();parent.stdout.close()


if __name__=='__main__':
    unittest.main()
