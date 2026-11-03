"""Keep selected hard-cut words when the following decoder/device fails."""
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace as Item
import unittest
from unittest.mock import Mock,patch
import numpy as np
from app import Transcriber,save_segment
from paragraphs import ParagraphAssembler
from recognition import RecognitionContext
from segmentation import AudioChunk,PauseSegmenter,WordAssembler,RATE


class PendingWordsTests(unittest.TestCase):
    def assembler_with_prefix(self):
        assembler=WordAssembler()
        output=assembler.consume(AudioChunk(np.zeros(15*RATE,np.float32),0.,1800000000.,forced=True),
            [Item(words=[Item(start=.2,end=14.3,word='已识别'),Item(start=14.7,end=14.9,word='边界词')])])
        self.assertEqual(output,[])
        return assembler

    def test_drain_preserves_stable_text_timestamps_and_is_once_only(self):
        assembler=self.assembler_with_prefix()
        self.assertEqual(assembler.finish(),[(.2,14.3,'已识别',1800000000.2)])
        self.assertIsNone(assembler.pending)
        self.assertEqual(assembler.finish(),[])
        self.assertEqual(assembler.last_end,14.3)

    def test_empty_and_already_published_words_are_not_reemitted(self):
        assembler=WordAssembler()
        self.assertEqual(assembler.finish(),[])
        assembler.consume(AudioChunk(np.zeros(RATE),0.,1000.),[Item(words=[Item(start=0.,end=.5,word='已显示')])])
        self.assertEqual(assembler.finish(),[])

    def capture(self,*,dual=False,device_failure=False,normal=False,live_rule=False,ordinary=False):
        events=queue.Queue();engine=Transcriber(events);streams=[];calls=[]
        sources=['system','microphone'] if dual else ['system']
        blocks=[(source,np.full(15*RATE,.1,np.float32),0.) for source in sources]
        if not dual:blocks.append(('system',np.full(RATE,.1,np.float32),15.))
        class CaptureQueue:
            def __init__(self,**kwargs):self.index=0
            def get(self,timeout):
                if self.index==len(blocks):engine.stop();raise queue.Empty
                source,samples,start=blocks[self.index];self.index+=1
                return source,start,samples,1800000000.+start
            def empty(self):return self.index==len(blocks)
            def qsize(self):return len(blocks)-self.index
        class Stream:
            closed=False
            def start_stream(self):pass
            def is_active(self):return not(device_failure and calls)
            def close(self):self.closed=True
        class API:
            terminated=False
            def get_device_info_by_index(self,index):return dict(defaultSampleRate=RATE,maxInputChannels=1)
            def open(self,**kwargs):s=Stream();streams.append(s);return s
            def terminate(self):self.terminated=True
        class Segmenter(PauseSegmenter):
            def __init__(self):super().__init__(lambda a:[dict(start=0,end=len(a))])
            def push(self,samples,start,epoch):
                if ordinary:return [AudioChunk(samples,start,epoch)]
                return super().push(samples,start,epoch)
            def finish(self):return [] if ordinary else super().finish()
        def recognize(sound,**options):
            calls.append(options)
            if len(calls)>len(sources):
                if live_rule:engine.update_corrections('密等性=最新规则')
                if not normal:raise RuntimeError('Decoder unavailable')
                words=[Item(start=.5,end=.7,word='密等性'),Item(start=.7,end=1.5,word='继续')]
            else:
                words=[Item(start=.2,end=1.,word='我们使用'),Item(start=13.8,end=14.3,word='密等性'),Item(start=14.7,end=14.9,word='继续')]
            return iter([Item(text=''.join(w.word for w in words),words=words)]),None
        engine.model_name='small';engine.model=Item(model=Item(device='cuda'),transcribe=recognize)
        api=API()
        with patch('app.queue.Queue',CaptureQueue),patch('app.pa.PyAudio',return_value=api),patch('app.PauseSegmenter',Segmenter),patch('app.DraftPreview') as preview:
            preview.return_value.render.return_value=None
            engine.run([dict(index=i+1,source=s) for i,s in enumerate(sources)],'small','zh','密等性=幂等性')
        messages=list(events.queue)
        self.assertTrue(api.terminated and all(s.closed for s in streams))
        self.assertEqual(sum(k=='done' for k,v in messages),1)
        return [v for k,v in messages if k=='segment'],[v for k,v in messages if k=='error'],calls

    def test_model_failure_retains_corrected_prefix_in_real_capture_and_save_path(self):
        rows,errors,calls=self.capture()
        self.assertEqual(errors,['Decoder unavailable'])
        self.assertEqual(len(calls),2)
        self.assertEqual([(r['source'],r['text'],r['start'],r['end']) for r in rows],[('system','我们使用幂等性',.2,14.3)])
        self.assertIs(rows[0]['incomplete'],True)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recording.jsonl';save_segment(path,rows[0])
            self.assertEqual(json.loads(path.read_text('utf-8')),rows[0])
            self.assertIn('我们使用幂等性',path.with_suffix('.txt').read_text('utf-8-sig'))
            self.assertNotIn('继续',path.with_suffix('.txt').read_text('utf-8-sig'))

    def test_failure_on_one_source_retains_both_sources_pending_words(self):
        rows,errors,calls=self.capture(dual=True)
        self.assertEqual(errors,['Decoder unavailable'])
        self.assertEqual([(r['source'],r['text']) for r in rows],[('system','我们使用幂等性'),('microphone','我们使用幂等性')])
        self.assertEqual(len(calls),3)

    def test_device_disconnect_retains_prefix_without_retrying_model(self):
        rows,errors,calls=self.capture(device_failure=True)
        self.assertEqual([r['text'] for r in rows],['我们使用幂等性'])
        self.assertEqual(len(calls),1)
        self.assertEqual(len(errors),1);self.assertIn('音频设备已断开',errors[0])

    def test_live_saved_rule_is_used_once_without_changing_acoustic_hints(self):
        rows,errors,calls=self.capture(live_rule=True)
        self.assertEqual([r['text'] for r in rows],['我们使用最新规则'])
        self.assertEqual(errors,['Decoder unavailable'])
        self.assertTrue(all(not options.get('hotwords') for options in calls))

    def test_successful_tail_is_committed_once_without_cleanup_duplicate(self):
        rows,errors,calls=self.capture(normal=True)
        self.assertEqual(errors,[])
        self.assertEqual(len(calls),2)
        self.assertEqual([r['text'] for r in rows],['我们使用幂等性继续'])
        self.assertNotIn('incomplete',rows[0])

    def test_failure_marks_unfinished_paragraph_without_a_hard_cut(self):
        rows,errors,calls=self.capture(ordinary=True)
        self.assertEqual(errors,['Decoder unavailable'])
        self.assertEqual(len(calls),2)
        self.assertEqual([r['text'] for r in rows],['我们使用幂等性继续'])
        self.assertIs(rows[0]['incomplete'],True)

    def test_recovery_error_on_one_source_does_not_block_other_source(self):
        events=queue.Queue();engine=Transcriber(events);paragraphs=ParagraphAssembler()
        broken=Mock();broken.normalize.side_effect=ValueError('bad normalization')
        states=dict(system=dict(assembler=self.assembler_with_prefix(),guidance=broken),
                    microphone=dict(assembler=self.assembler_with_prefix(),guidance=RecognitionContext('已识别=保留')))
        engine._finish_pending_words(states,paragraphs)
        engine._finish_pending_words(states,paragraphs)
        for row in paragraphs.finish():engine.emit('segment',row)
        self.assertEqual([v['text'] for k,v in events.queue if k=='segment'],['保留'])
        self.assertEqual(len([v for k,v in events.queue if k=='error']),1)
