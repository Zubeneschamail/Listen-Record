from types import SimpleNamespace as Item
from unittest.mock import Mock
import unittest
import numpy as np
from segmentation import PauseSegmenter, WordAssembler, AudioChunk, DraftPreview, RATE, decode_chunk


def detector(audio):
    """Deterministic activity labels to test splitting independently of VAD accuracy."""
    mask = audio > 0
    edges = np.diff(np.pad(mask.astype(int), (1, 1)))
    return [dict(start=int(a), end=int(b)) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


def feed(segmenter, signal):
    result = []
    for start in range(0, len(signal), 1600):
        result.extend(segmenter.push(signal[start:start + 1600], start / RATE, 1000 + start / RATE))
    return result


def words(*values):
    return [Item(text="".join(value[2] for value in values),
                 words=[Item(start=a, end=b, word=text) for a, b, text in values])]


class SegmentationTests(unittest.TestCase):
    def test_preview_does_not_commit_words_and_final_can_correct_them(self):
        clock = [0.0]
        preview = DraftPreview(clock=lambda: clock[0])
        segmenter, assembler = PauseSegmenter(detector), WordAssembler()
        segmenter.push(np.ones(2 * RATE, dtype=np.float32), 0, 1000)
        model = Mock()
        model.transcribe.return_value = (words((0, 1, "临时错词")), None)
        self.assertEqual(preview.render(model, assembler, segmenter, "zh"), "临时错词")
        self.assertEqual(assembler.last_end, -1)
        self.assertIsNone(assembler.pending)
        self.assertEqual(len(segmenter.audio), 2 * RATE)
        model.transcribe.return_value = (words((0, 1, "最终正确")), None)
        rows = decode_chunk(model, assembler, segmenter.finish()[0], "zh")
        self.assertEqual(rows[0][2], "最终正确")

    def test_preview_yields_to_backlog_stop_and_cooldown(self):
        clock = [0.0]
        preview = DraftPreview(clock=lambda: clock[0])
        segmenter, assembler = PauseSegmenter(detector), WordAssembler()
        segmenter.push(np.ones(2 * RATE, dtype=np.float32), 0, 1000)
        model = Mock()
        model.transcribe.return_value = (words((0, 1, "草稿")), None)
        self.assertIsNone(preview.render(model, assembler, segmenter, "zh", backlog=1))
        self.assertIsNone(preview.render(model, assembler, segmenter, "zh", stopping=True))
        model.transcribe.assert_not_called()
        self.assertEqual(preview.render(model, assembler, segmenter, "zh"), "草稿")
        segmenter.push(np.ones(2 * RATE, dtype=np.float32), 2, 1002)
        clock[0] = 1
        self.assertIsNone(preview.render(model, assembler, segmenter, "zh"))
        clock[0] = 3
        self.assertEqual(preview.render(model, assembler, segmenter, "zh"), "草稿")
        self.assertEqual(model.transcribe.call_count, 2)

    def test_speech_runs_past_five_seconds_and_ends_on_pause(self):
        segmenter = PauseSegmenter(detector)
        signal = np.r_[np.ones(8 * RATE), np.zeros(int(0.8 * RATE))].astype(np.float32)
        chunks = feed(segmenter, signal)
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(len(chunks[0].audio) / RATE, 8.2)
        self.assertFalse(chunks[0].forced)
        self.assertEqual(segmenter.finish(), [])

    def test_continuous_speech_overlaps_at_limit_and_flushes_tail(self):
        segmenter = PauseSegmenter(detector)
        chunks = feed(segmenter, np.ones(18 * RATE, dtype=np.float32))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].forced)
        self.assertEqual(len(chunks[0].audio), 15 * RATE)
        tail = segmenter.finish()[0]
        self.assertTrue(tail.overlap)
        self.assertAlmostEqual(tail.start, 14.2)
        self.assertAlmostEqual(tail.epoch, 1014.2)
        self.assertAlmostEqual(len(tail.audio) / RATE, 3.8)

    def test_limit_prefers_recent_short_pause(self):
        signal = np.ones(15 * RATE, dtype=np.float32)
        signal[12 * RATE:int(12.2 * RATE)] = 0
        chunks = feed(PauseSegmenter(detector), signal)
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].forced)
        self.assertAlmostEqual(len(chunks[0].audio) / RATE, 12.06)

    def test_silence_stays_bounded_and_gap_preserves_wall_clock(self):
        segmenter = PauseSegmenter(detector)
        self.assertEqual(feed(segmenter, np.zeros(40 * RATE, dtype=np.float32)), [])
        self.assertLessEqual(len(segmenter.audio), int(0.4 * RATE))
        segmenter = PauseSegmenter(detector)
        self.assertEqual(segmenter.push(np.ones(RATE, dtype=np.float32), 0, 1000), [])
        before_gap = segmenter.push(np.ones(RATE, dtype=np.float32), 10, 1010)
        self.assertEqual(before_gap[0].start, 0)
        after_gap = segmenter.finish()[0]
        self.assertEqual(after_gap.start, 10)
        self.assertEqual(after_gap.epoch, 1010)

    def test_overlap_does_not_repeat_words_or_drop_deferred_tail(self):
        assembler = WordAssembler()
        first = AudioChunk(np.ones(15 * RATE), 0, 1000, forced=True)
        out = assembler.consume(first, words((13.8, 14.3, "系统"), (14.4, 14.9, "声音")))
        self.assertEqual(out, [])  # Hold the partial phrase until its tail arrives.
        second = AudioChunk(np.ones(2 * RATE), 14.2, 1014.2, overlap=True)
        out = assembler.consume(second, words((0, 0.1, "系统"), (0.2, 0.7, "声音"),
                                             (0.8, 1.1, "很好"), (1.1, 1.4, "很好")))
        self.assertEqual(out[0][2], "系统声音很好很好")
        self.assertAlmostEqual(out[0][3], 1013.8)

    def test_stopping_exactly_at_hard_cut_still_flushes_overlap(self):
        segmenter = PauseSegmenter(detector)
        feed(segmenter, np.ones(15 * RATE, dtype=np.float32))
        tail = segmenter.finish()
        self.assertEqual(len(tail), 1)
        self.assertEqual(len(tail[0].audio), int(0.8 * RATE))
        self.assertFalse(tail[0].forced)
        self.assertEqual(segmenter.finish(), [])
