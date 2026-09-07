"""Dual capture: a shared model with independent segmenters and contexts."""
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from app import Transcriber


class DualAudioTests(unittest.TestCase):
    def test_both_inputs_are_tagged_and_cleanup_waits_for_both(self):
        events = queue.Queue()
        engine = Transcriber(events)
        engine.model_name, engine.model = 'small', object()
        opened, contexts = [], []

        class Stream:
            def __init__(self, index, callback):
                self.index, self.callback, self.closed = index, callback, False
            def start_stream(self):
                self.callback(np.full(1600, self.index, dtype=np.float32).tobytes(), 1600, {}, 0)
                if self.index == 2:
                    engine.stop()
            def close(self):
                self.closed = True
            def is_active(self):
                return True

        class API:
            terminated = False
            def get_device_info_by_index(self, index):
                return {'defaultSampleRate': 16000, 'maxInputChannels': 1, 'isLoopbackDevice': index == 1}
            def open(self, **kw):
                stream = Stream(kw['input_device_index'], kw['stream_callback'])
                opened.append(stream)
                return stream
            def terminate(self):
                self.terminated = True

        class Segmenter:
            def push(self, samples, offset, epoch):
                return [SimpleNamespace(audio=samples, start=offset, epoch=epoch)]
            def finish(self):
                return []

        def decode(model, assembler, chunk, language, guidance, backlog):
            contexts.append(guidance)
            self.assertFalse(guidance.history)
            guidance.history.append('独立上下文')
            return [(chunk.start, chunk.start+.1, '我说话' if chunk.audio[0] == 2 else '电脑播放', chunk.epoch)]

        api = API()
        with patch('app.pa.PyAudio', return_value=api), patch('app.PauseSegmenter', Segmenter), \
             patch('app.decode_chunk', side_effect=decode), patch('app.DraftPreview'):
            engine.run([{'index': 1, 'source': 'system'}, {'index': 2, 'source': 'microphone'}], 'small', 'zh')
        messages = list(events.queue)
        self.assertFalse([v for k,v in messages if k == 'error'])
        rows = [v for k,v in messages if k == 'segment']
        self.assertEqual([(r['source'], r['text']) for r in rows], [('system', '电脑播放'), ('microphone', '我说话')])
        self.assertIsNot(contexts[0], contexts[1])
        self.assertTrue(all(s.closed for s in opened))
        self.assertTrue(api.terminated)
        self.assertEqual(sum(k == 'done' for k,v in messages), 1)

    def test_second_device_failure_closes_first_stream(self):
        events = queue.Queue()
        engine = Transcriber(events)
        engine.model_name, engine.model = 'small', object()
        from unittest.mock import Mock
        api, stream = Mock(), Mock()
        api.get_device_info_by_index.return_value = {'defaultSampleRate': 16000, 'maxInputChannels': 1}
        api.open.side_effect = [stream, OSError('麦克风不可用')]
        with patch('app.pa.PyAudio', return_value=api):
            engine.run([{'index':1, 'source':'system'}, {'index':2, 'source':'microphone'}], 'small', 'zh')
        stream.close.assert_called_once()
        api.terminate.assert_called_once()
        self.assertTrue(any(k == 'error' and '麦克风不可用' in v for k,v in events.queue))
