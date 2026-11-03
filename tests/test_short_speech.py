"""Backend-specific context without delaying normal speech or losing short replies."""
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

import app
from qwen_engine import QWEN_MODEL
from segmentation import PauseSegmenter, RATE


def detector(audio):
    edges = np.flatnonzero(np.diff(np.pad((audio > .5).astype(np.int8), (1, 1))))
    return [dict(start=int(start), end=int(end)) for start, end in zip(edges[::2], edges[1::2])
            if end - start >= round(.16 * RATE)]


def signal(seconds, spans):
    audio = np.zeros(round(seconds * RATE), dtype=np.float32)
    for start, end in spans:
        audio[round(start * RATE):round(end * RATE)] = 1.
    return audio


def stream(segmenter, audio):
    segmenter.detector = detector
    rows = []
    for offset in range(0, len(audio), RATE // 10):
        part = audio[offset:offset + RATE // 10]
        rows.extend((chunk, (offset + len(part)) / RATE) for chunk in
                    segmenter.push(part, offset / RATE, 1000. + offset / RATE))
    rows.extend((chunk, len(audio) / RATE) for chunk in segmenter.finish())
    return rows


def covered(rows):
    return {round(chunk.start * RATE) + int(index) for chunk, _ in rows
            for index in np.flatnonzero(chunk.audio > .5)}


def signature(rows):
    return [(chunk.start, chunk.epoch, chunk.forced, chunk.overlap, available, chunk.audio.tobytes())
            for chunk, available in rows]


class ShortSpeechTests(unittest.TestCase):
    def test_qwen_joins_brief_onset_with_nearby_continuation(self):
        audio = signal(3., [(0., .35), (.99, 1.8)])
        rows = stream(app.create_segmenter(QWEN_MODEL), audio)
        self.assertEqual(len(rows), 1)
        self.assertEqual(covered(rows), set(np.flatnonzero(audio > .5)))

    def test_whisper_and_default_segmenter_keep_legacy_pause_behavior(self):
        audio = signal(3., [(0., .35), (.99, 1.8)])
        expected = stream(PauseSegmenter(), audio)
        self.assertEqual(len(expected), 2)
        for name in ('small', 'large-v3-turbo', 'base', 'tiny'):
            with self.subTest(model=name):
                self.assertEqual(signature(stream(app.create_segmenter(name), audio)), signature(expected))

    def test_normal_qwen_utterance_has_unchanged_timing_and_audio(self):
        audio = signal(2.4, [(0., 1.2)])
        self.assertEqual(signature(stream(app.create_segmenter(QWEN_MODEL), audio)),
                         signature(stream(PauseSegmenter(), audio)))

    def test_isolated_short_reply_stays_bounded_and_keeps_same_model_input(self):
        audio = signal(3., [(0., .35)])
        before = stream(PauseSegmenter(), audio)
        after = stream(app.create_segmenter(QWEN_MODEL), audio)
        self.assertEqual(len(after), 1)
        np.testing.assert_array_equal(after[0][0].audio, before[0][0].audio)
        self.assertGreater(after[0][1], before[0][1])
        self.assertLessEqual(after[0][1] - before[0][1], .8)
        self.assertLess(after[0][1], 2.)

    def test_stop_flushes_short_reply_without_waiting_for_confirmation(self):
        audio = signal(.35, [(0., .35)])
        rows = stream(app.create_segmenter(QWEN_MODEL), audio)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], .35)
        np.testing.assert_array_equal(rows[0][0].audio, audio)

    def test_long_pause_keeps_distinct_short_replies(self):
        audio = signal(4.7, [(0., .35), (2.35, 2.7)])
        rows = stream(app.create_segmenter(QWEN_MODEL), audio)
        self.assertEqual(len(rows), 2)
        self.assertEqual(covered(rows), set(np.flatnonzero(audio > .5)))

    def test_hard_cut_and_missing_callbacks_preserve_samples_and_clock(self):
        audio = signal(18., [(0., 17.)])
        before = stream(PauseSegmenter(), audio)
        after = stream(app.create_segmenter(QWEN_MODEL), audio)
        self.assertEqual(signature(after), signature(before))
        self.assertTrue(any(chunk.forced for chunk, _ in after))
        short = signal(.35, [(0., .35)])
        state = app.create_segmenter(QWEN_MODEL)
        state.detector = detector
        chunks = state.push(short, 0., 1000.) + state.push(short, 5., 1005.) + state.finish()
        self.assertEqual([(chunk.start, chunk.epoch) for chunk in chunks], [(0., 1000.), (5., 1005.)])

    def test_capture_configures_both_sources_for_the_requested_model(self):
        for name in (QWEN_MODEL, 'large-v3-turbo'):
            with self.subTest(model=name):
                events = queue.Queue()
                engine = app.Transcriber(events)
                api = Mock()
                api.get_device_info_by_index.return_value = dict(defaultSampleRate=16000, maxInputChannels=1)
                model = SimpleNamespace(model=SimpleNamespace(device='cuda'))
                def load(*args, **options):
                    engine.stop()
                    return model
                with patch('app.pa.PyAudio', return_value=api), \
                        patch('app.load_model', side_effect=load), \
                        patch('app.create_segmenter', wraps=app.create_segmenter) as factory:
                    engine.run([dict(index=1, source='system'), dict(index=2, source='microphone')], name, 'zh')
                self.assertEqual(factory.call_args_list, [call(name), call(name)])
                self.assertFalse([value for kind, value in list(events.queue) if kind == 'error'])
                api.terminate.assert_called_once()
                engine.unload_model()


if __name__ == '__main__':
    unittest.main()
