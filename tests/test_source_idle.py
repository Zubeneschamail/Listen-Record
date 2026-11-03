"""One capture can stop callbacks while the other keeps the shared queue busy."""
import queue
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app import Transcriber
from segmentation import PauseSegmenter, RATE


def activity(audio):
    edges = np.diff(np.pad((audio > 0).astype(int), (1, 1)))
    return [dict(start=int(a), end=int(b))
            for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


class SourceIdleTests(unittest.TestCase):
    def run_capture(self, *, quiet_source='system', forced=False, backlog=False, other_speech=False,
                    speech_seconds=None, keep_silent_callbacks=False, align_to_chunk_end=False):
        clock = [100.]
        timeline = []
        class CaptureEvents(queue.Queue):
            def put(self, item, *args, **kwargs):
                timeline.append((clock[0], item))
                return super().put(item, *args, **kwargs)
        events = CaptureEvents()
        engine = Transcriber(events)
        engine.model_name, engine.model = 'small', object()
        observed = {}
        chunks_decoded = []
        duration = speech_seconds if speech_seconds is not None else 15 if forced else 2
        other = 'microphone' if quiet_source == 'system' else 'system'
        blocks = []
        for index in range(math.ceil((duration+3)*10)):
            offset = index/10
            count = max(0, min(1600, round(duration*RATE)-index*1600))
            if count or keep_silent_callbacks:
                audio = np.zeros(1600, dtype=np.float32)
                audio[:count] = .1
                blocks.append((quiet_source, offset, audio))
            blocks.append((other, offset, np.full(1600, .1 if other_speech else 0., dtype=np.float32)))

        class CaptureQueue:
            def __init__(self, **kwargs):
                self.index = 0
            def get(self, timeout):
                if self.index == len(blocks):
                    # Observe before stop/final cleanup. The queue never raised
                    # Empty while callbacks from the other source continued.
                    observed['before_stop'] = list(events.queue)
                    observed['timeline_before_stop'] = list(timeline)
                    observed['decoded_before_stop'] = list(chunks_decoded)
                    engine.stop()
                    raise queue.Empty
                source, offset, audio = blocks[self.index]
                self.index += 1
                clock[0] = 100. + offset
                return source, offset, audio, 1800000000.+offset
            def qsize(self):
                return int(backlog and self.index < len(blocks))
            def empty(self):
                return not self.qsize()

        class Stream:
            def start_stream(self):
                pass
            def is_active(self):
                return True
            def close(self):
                pass

        class API:
            def get_device_info_by_index(self, index):
                return dict(defaultSampleRate=RATE, maxInputChannels=1)
            def open(self, **kwargs):
                return Stream()
            def terminate(self):
                pass

        class ActivitySegmenter(PauseSegmenter):
            def __init__(self):
                super().__init__(activity)

        def decode(model, assembler, chunk, language, guidance, backlog):
            chunks_decoded.append(dict(start=chunk.start, end=chunk.start+len(chunk.audio)/RATE,
                                       forced=chunk.forced, overlap=chunk.overlap,
                                       stopped=engine.stop_event.is_set(), decoded_at=clock[0]))
            if chunk.forced:
                values = [(13.8, 14.3, '什么是'), (14.6, 14.9, '幂等性？')]
            elif forced:
                values = [(14.-chunk.start, 14.3-chunk.start, '什么是'),
                          (14.6-chunk.start, 14.9-chunk.start, '幂等性？')]
            else:
                values = [(0., .8, '什么是'),
                          (.9, len(chunk.audio)/RATE if align_to_chunk_end else 1.9, '幂等性？')]
            segments = [SimpleNamespace(words=[SimpleNamespace(start=a, end=b, word=text)
                                               for a, b, text in values])]
            return assembler.consume(chunk, segments)

        with patch('app.queue.Queue', CaptureQueue), patch('app.pa.PyAudio', API), \
                patch('app.PauseSegmenter', ActivitySegmenter), \
                patch('app.time.monotonic', lambda: clock[0]), \
                patch('app.decode_chunk', side_effect=decode), \
                patch('app.DraftPreview') as preview:
            preview.return_value.render.return_value = None
            engine.run([dict(index=1, source='system'), dict(index=2, source='microphone')],
                       'small', 'zh')
        self.assertFalse([value for kind, value in events.queue if kind == 'error'])
        return observed, list(events.queue)

    def verify_idle_finalization(self, quiet_source, forced):
        observed, all_events = self.run_capture(quiet_source=quiet_source, forced=forced)
        rows = [value for kind, value in observed['before_stop'] if kind == 'segment']
        self.assertEqual([(row['source'], row['text']) for row in rows],
                         [(quiet_source, '什么是幂等性？')])
        self.assertEqual(sum(kind == 'segment' for kind, value in all_events), 1)
        active = next(index for index, (kind, value) in enumerate(observed['before_stop'])
                      if kind == 'voice_activity' and value == dict(source=quiet_source, active=True))
        idle = next(index for index, (kind, value) in enumerate(observed['before_stop'])
                    if index > active and kind == 'voice_activity' and
                    value == dict(source=quiet_source, active=False))
        segment = next(index for index, (kind, _) in enumerate(observed['before_stop']) if kind == 'segment')
        self.assertGreater(idle, segment)
        final = observed['decoded_before_stop'][-1]
        self.assertFalse(final['stopped'])
        self.assertAlmostEqual(final['end'], 15. if forced else 2.)
        if forced:
            self.assertTrue(final['overlap'])
            self.assertAlmostEqual(final['start'], 12.6)

    def test_system_question_finishes_while_mic_silence_continues(self):
        self.verify_idle_finalization('system', False)

    def test_pending_hard_cut_question_finishes_while_mic_silence_continues(self):
        self.verify_idle_finalization('system', True)

    def test_microphone_question_finishes_while_system_silence_continues(self):
        self.verify_idle_finalization('microphone', False)

    def test_busy_queue_defers_idle_finalization_until_audio_is_drained(self):
        observed, all_events = self.run_capture(forced=True, backlog=True)
        final = observed['decoded_before_stop'][-1]
        self.assertFalse(final['stopped'])
        self.assertTrue(final['overlap'])
        self.assertAlmostEqual(final['decoded_at'], 117.9)
        self.assertEqual(sum(kind == 'segment' for kind, value in all_events), 1)

    def test_idle_question_reaches_auto_qa_before_recording_stops(self):
        from qa_worker import QAWorker
        observed, _ = self.run_capture(forced=True)
        clock = [100.]
        events = queue.Queue()
        qa = QAWorker(events, lambda prompt, cancel: '同一订单只处理一次。', lambda: clock[0])
        qa.set_enabled(True)
        qa.set_question_source('system')
        try:
            for moment, (kind, value) in observed['timeline_before_stop']:
                clock[0] = moment
                if kind == 'segment':
                    qa.feed(value['text'], value['source'])
                elif kind == 'voice_activity':
                    qa.observe_voice(value['source'], value['active'])
                qa.tick()
            self.assertTrue(qa.active)
            states = [events.get(timeout=3)[1] for _ in range(3)]
            self.assertEqual(states[0][1:], ('thinking', '什么是幂等性？'))
            self.assertEqual(states[1][1:], ('answer', ('什么是幂等性？', '同一订单只处理一次。')))
            self.assertEqual(states[2][1], 'done')
        finally:
            qa.reset()

    def test_other_speaker_keeps_recording_while_idle_question_is_finalized(self):
        observed, all_events = self.run_capture(other_speech=True)
        rows = [row for kind, row in observed['before_stop'] if kind == 'segment']
        self.assertEqual([(row['source'], row['text']) for row in rows], [('system', '什么是幂等性？')])
        all_rows = [row for kind, row in all_events if kind == 'segment']
        self.assertEqual([row['source'] for row in all_rows], ['system', 'microphone'])
        self.assertFalse(observed['decoded_before_stop'][-1]['stopped'])

    def test_continuous_silence_waits_for_final_text_before_voice_idle(self):
        observed, _ = self.run_capture(speech_seconds=2.05, keep_silent_callbacks=True,
                                       align_to_chunk_end=True)
        events = observed['before_stop']
        active = next(i for i, (kind, value) in enumerate(events)
                      if kind == 'voice_activity' and value == dict(source='system', active=True))
        idle = next(i for i, (kind, value) in enumerate(events)
                    if i > active and kind == 'voice_activity' and
                    value == dict(source='system', active=False))
        final = next(i for i, (kind, value) in enumerate(events) if kind == 'segment')
        self.assertGreater(idle, final)
        self.assertEqual(events[final][1]['text'], '什么是幂等性？')


if __name__ == '__main__':
    unittest.main()
