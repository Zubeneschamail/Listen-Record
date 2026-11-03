"""Speech arriving during model loading must survive startup and stop/drain."""
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app import Transcriber


class CaptureStartupTests(unittest.TestCase):
    def exercise(self, *, fail=False, overflow=False, cached=False, already_stopped=False):
        events = queue.Queue()
        engine = Transcriber(events)
        timeline, streams, decoded = [], [], []
        model = SimpleNamespace(model=SimpleNamespace(device='cuda'))
        if cached:
            engine.model_name, engine.model = 'new-model', model
        else:
            old = Mock()
            old.close.side_effect = lambda: timeline.append('unload-old')
            engine.model_name, engine.model = 'old-model', old
        if already_stopped:
            engine.stop()

        class Stream:
            def __init__(self, index, callback):
                self.index, self.callback = index, callback
                self.closed = False
            def feed(self):
                audio = np.full(1600, self.index/10, dtype=np.float32)
                return self.callback(audio.tobytes(), len(audio), {}, 0)
            def start_stream(self):
                timeline.append(f'capture-{self.index}')
                self.feed()
                if cached and self.index == 2:
                    engine.stop()
            def is_active(self):
                return True
            def close(self):
                self.closed = True

        api = Mock()
        api.get_device_info_by_index.return_value = dict(defaultSampleRate=16000,maxInputChannels=1)
        def open_stream(**options):
            stream = Stream(options['input_device_index'],options['stream_callback'])
            streams.append(stream)
            return stream
        api.open.side_effect = open_stream

        def load(*args, **kwargs):
            timeline.append('load-model')
            self.assertEqual(timeline[:2], ['capture-1','capture-2'])
            if fail:
                raise RuntimeError('Model unavailable')
            if overflow:
                streams[0].feed()
                streams[0].feed()
            engine.stop()  # User presses Stop while the model is still loading.
            return model

        class Segmenter:
            pending_speech = False
            last_speech_end = None
            def push(self, samples, offset, epoch):
                return [SimpleNamespace(audio=samples,start=offset,epoch=epoch)]
            def finish(self):
                return []

        def decode(actual, assembler, chunk, language, guidance, backlog):
            self.assertIs(actual,model)
            decoded.append(float(chunk.audio[0]))
            return [(chunk.start,chunk.start+.1,'开头的声音。',chunk.epoch)]

        real_queue = queue.Queue
        with patch('app.pa.PyAudio',return_value=api) as capture, \
                patch('app.load_model',side_effect=load) as loader, \
                patch('app.PauseSegmenter',Segmenter), patch('app.decode_chunk',side_effect=decode), \
                patch('app.DraftPreview') as preview, \
                patch('app.queue.Queue',side_effect=lambda **options: real_queue(maxsize=3 if overflow else options['maxsize'])):
            preview.return_value.render.return_value = None
            engine.run([dict(index=1,source='system'),dict(index=2,source='microphone')],'new-model','zh')
        if cached or already_stopped:
            loader.assert_not_called()
        else:
            loader.assert_called_once()
        if already_stopped:
            capture.assert_not_called()
        else:
            api.terminate.assert_called_once()
            self.assertTrue(all(stream.closed for stream in streams))
        messages = list(events.queue)
        self.assertEqual(sum(kind=='done' for kind,_ in messages),1)
        return engine,timeline,decoded,messages

    def test_cold_load_captures_both_openings_before_loading_and_drains_after_stop(self):
        engine,timeline,decoded,messages = self.exercise()
        self.assertEqual(timeline,['capture-1','capture-2','unload-old','load-model'])
        np.testing.assert_allclose(decoded,[.1,.2])
        self.assertFalse([v for k,v in messages if k=='error'])
        self.assertEqual({v['source'] for k,v in messages if k=='segment'},{'system','microphone'})
        self.assertTrue(any(k=='status' and '音频暂存' in v for k,v in messages))

    def test_warm_model_reuse_keeps_capture_and_stop_drain(self):
        _,timeline,decoded,messages = self.exercise(cached=True)
        self.assertEqual(timeline,['capture-1','capture-2'])
        np.testing.assert_allclose(decoded,[.1,.2])
        self.assertFalse([v for k,v in messages if k=='error'])

    def test_model_failure_closes_early_capture_and_reports_error(self):
        _,_,decoded,messages = self.exercise(fail=True)
        self.assertFalse(decoded)
        self.assertEqual([v for k,v in messages if k=='error'],['Model unavailable'])
        self.assertFalse([v for k,v in messages if k=='recognition_backend'])

    def test_startup_overflow_reports_loss_and_drains_bounded_buffer(self):
        _,_,decoded,messages = self.exercise(overflow=True)
        self.assertEqual(len(decoded),3)
        self.assertTrue(any(k=='error' and '缓冲溢出' in v for k,v in messages))

    def test_stop_before_capture_skips_devices_and_model_loading(self):
        _,timeline,decoded,messages = self.exercise(already_stopped=True)
        self.assertEqual(timeline,[])
        self.assertEqual(decoded,[])
        self.assertFalse([v for k,v in messages if k=='error'])
