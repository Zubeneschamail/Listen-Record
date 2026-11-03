"""Explicit saved rules reach active decoding without changing acoustic hints."""
from copy import deepcopy
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace as Item
import unittest
from unittest.mock import Mock,patch
import numpy as np

from app import App,Transcriber
from recognition import CorrectionUpdates,RecognitionContext
from recognition_rules import load_rule_preferences
from segmentation import AudioChunk,WordAssembler,decode_chunk,RATE,DraftPreview,PauseSegmenter


class LiveCorrectionTests(unittest.TestCase):
    def test_live_and_restarted_rules_have_priority_over_conflicting_hint_case(self):
        updates=CorrectionUpdates()
        live=RecognitionContext('agent',correction_source=updates.snapshot)
        updates.publish('a gent=Agent, agent')
        restarted=RecognitionContext('a gent=Agent, agent')
        self.assertEqual(live.normalize('a gent'),'Agent')
        self.assertEqual(restarted.normalize('a gent'),'Agent')
        self.assertEqual(live.options(0),restarted.options(0))

    def test_two_sources_follow_latest_complete_rules_and_keep_hints_and_rtf(self):
        updates=CorrectionUpdates()
        contexts=[RecognitionContext('RAG, old=OldName',correction_source=updates.snapshot) for _ in range(2)]
        for context in contexts:context.rtf=.8
        options=[c.options(0) for c in contexts]
        updates.publish('密等性=幂等性, old=NewName, PostgreSQL')
        for index,context in enumerate(contexts):
            self.assertEqual(context.normalize('密等性 old postgre sql r a g'), '幂等性 NewName postgre sql RAG')
            self.assertEqual(context.options(0),options[index])
            self.assertEqual(context.rtf,.8)
        updates.publish('old=Latest, RAG')
        for context in contexts:self.assertEqual(context.normalize('old 密等性'),'Latest 密等性')
        updates.publish('')
        for context in contexts:self.assertEqual(context.normalize('old r a g'),'old RAG')

    def test_rule_published_during_inference_applies_before_result_is_emitted(self):
        updates=CorrectionUpdates()
        context=RecognitionContext('RAG',correction_source=updates.snapshot)
        entered,released=threading.Event(),threading.Event()
        captured={}
        def infer(audio,**options):
            captured.update(options)
            entered.set()
            if not released.wait(3):raise RuntimeError('Test release not received')
            return [Item(text='密等性',words=[Item(start=.2,end=.6,word='密等性')])],None
        result,errors=[],[]
        chunk=AudioChunk(np.ones(RATE,np.float32),0,100)
        def decode():
            try:result.extend(decode_chunk(Item(transcribe=infer),WordAssembler(),chunk,'zh',context))
            except Exception as exc:errors.append(exc)
        worker=threading.Thread(target=decode)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            updates.publish('密等性=幂等性, PostgreSQL')
        finally:
            released.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(result,[(.2,.6,'幂等性',100.2)])
        self.assertEqual(captured['hotwords'],'RAG')
        self.assertNotIn('initial_prompt',captured)

    def test_overlap_preview_copy_shares_updates_without_mutating_original_rtf(self):
        updates=CorrectionUpdates()
        context=RecognitionContext('RAG',correction_source=updates.snapshot)
        clone=deepcopy(context)
        self.assertIs(clone.correction_source.__self__,updates)
        clone.rtf=1
        updates.publish('密等性=幂等性')
        self.assertEqual(clone.normalize('密等性'),'幂等性')
        self.assertEqual(context.rtf,0)
        self.assertFalse(context.corrections)
        segmenter=PauseSegmenter(lambda a:[{'start':0,'end':len(a)}])
        segmenter.push(np.ones(RATE*2,np.float32),0,100)
        assembler=WordAssembler()
        assembler.pending=(0,.2,'密等性',100)
        model=Mock()
        model.transcribe.return_value=([Item(text='密等性',words=[Item(start=.3,end=.7,word='密等性')])],None)
        text=DraftPreview().render(model,assembler,segmenter,'zh',guidance=context)
        self.assertIn('幂等性',text)
        self.assertEqual(context.rtf,0)
        self.assertEqual(assembler.pending,(0,.2,'密等性',100))

    def test_live_aliases_preserve_symbols_and_qualified_name_boundaries(self):
        updates=CorrectionUpdates()
        context=RecognitionContext('C++, .NET',correction_source=updates.snapshot)
        updates.publish('net=Network, .NET=DotNet, C++=Cpp')
        self.assertEqual(context.normalize('file.NET net.io .NET.dll C++17 file.C++'),
                         'file.NET net.io .NET.dll C++17 file.C++')
        self.assertEqual(context.normalize('net. .NET。C++.'),'Network. DotNet。Cpp.')

    def test_next_start_clears_capture_snapshot_and_uses_its_own_preferences(self):
        engine=Transcriber(queue.Queue())
        engine.update_corrections('old=OldCapture')
        with patch('app.threading.Thread') as thread:
            engine.start(1,'small','zh','old=NewCapture')
            thread.return_value.start.assert_called_once()
        self.assertIsNone(engine.correction_updates.snapshot())
        fresh=RecognitionContext('old=NewCapture',correction_source=engine.correction_updates.snapshot)
        self.assertEqual(fresh.normalize('old'),'NewCapture')

    def test_only_successfully_persisted_rules_are_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            engine=Transcriber(queue.Queue())
            app=Item(engine=engine,hotwords=Mock(),status=Mock())
            with patch('app.DATA',root):
                App.remember_correction_rule(app,'密等性','幂等性',None)
                saved=load_rule_preferences(root/'recognition-settings.json')
                self.assertEqual(engine.correction_updates.snapshot(),saved.hotwords)
                app.hotwords.set.reset_mock()
                with self.assertRaises(ValueError):App.remember_correction_rule(app,'新错词','新术语',None)
                app.hotwords.set.assert_not_called()
                self.assertEqual(engine.correction_updates.snapshot(),saved.hotwords)
                with patch('recognition_rules.save_rule',side_effect=OSError('disk full')):
                    with self.assertRaises(OSError):App.remember_correction_rule(app,'新错词','新术语',saved.fingerprint)
                self.assertEqual(engine.correction_updates.snapshot(),saved.hotwords)

    def test_saved_rule_during_cold_load_reaches_both_source_contexts_and_stop_flush(self):
        events=queue.Queue()
        engine=Transcriber(events)
        streams=[]
        def open_stream(**options):
            stream=Mock()
            def begin():
                samples=np.full((1600,1),.1,np.float32)
                options['stream_callback'](samples.tobytes(),1600,{},0)
            stream.start_stream.side_effect=begin
            streams.append(stream)
            return stream
        api=Mock()
        api.get_device_info_by_index.return_value={'maxInputChannels':1,'defaultSampleRate':RATE}
        api.open.side_effect=open_stream
        model=Mock()
        model.model.device='cpu'
        model.transcribe.return_value=([Item(text='密等性',words=[Item(start=0,end=.05,word='密等性')])],None)
        class Segmenter:
            last_speech_end=.1
            def push(self,audio,start,epoch):return [AudioChunk(audio,start,epoch)]
            def finish(self):return []
        with tempfile.TemporaryDirectory() as folder:
            app=Item(engine=engine,hotwords=Mock(),status=Mock())
            def load(*args,**kwargs):
                App.remember_correction_rule(app,'密等性','幂等性',None)
                engine.stop()
                return model
            with patch('app.DATA',Path(folder)),patch('app.pa.PyAudio',return_value=api), \
                    patch('app.create_segmenter',side_effect=lambda name:Segmenter()),patch('app.load_model',side_effect=load):
                engine.run([{'index':1,'source':'system'},{'index':2,'source':'microphone'}],'small','zh','RAG')
        messages=list(events.queue)
        self.assertFalse([v for k,v in messages if k=='error'])
        rows=[v for k,v in messages if k=='segment']
        self.assertEqual({v['source'] for v in rows},{'system','microphone'})
        self.assertTrue(all(v['text']=='幂等性' for v in rows))
        self.assertEqual(model.transcribe.call_count,2)
        self.assertTrue(all(c.kwargs['hotwords']=='RAG' for c in model.transcribe.call_args_list))
        self.assertTrue(all(s.close.call_count==1 for s in streams))


if __name__=='__main__':unittest.main()
