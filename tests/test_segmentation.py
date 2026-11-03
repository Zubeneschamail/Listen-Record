from types import SimpleNamespace as Item
from unittest.mock import Mock
import unittest
import numpy as np
from segmentation import (PauseSegmenter, WordAssembler, AudioChunk, DraftPreview,
                          RATE, decode_chunk, plausible_segment)


def detector(audio):
    """Deterministic activity labels to test splitting independently of VAD accuracy."""
    mask = audio > 0
    edges = np.diff(np.pad(mask.astype(int), (1, 1)))
    return [dict(start=int(a), end=int(b)) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


def onset_filtered_detector(audio):
    # The real VAD withholds a newly started speech run until it is long enough.
    return [speech for speech in detector(audio) if speech['end']-speech['start'] > .16*RATE]


def feed(segmenter, signal):
    result = []
    for start in range(0, len(signal), 1600):
        result.extend(segmenter.push(signal[start:start + 1600], start / RATE, 1000 + start / RATE))
    return result


def words(*values):
    return [Item(text="".join(value[2] for value in values),
                 words=[Item(start=a, end=b, word=text) for a, b, text in values])]


class SegmentationTests(unittest.TestCase):
    def test_short_onset_is_not_counted_as_terminal_silence(self):
        signal = np.r_[np.ones(int(.5*RATE)), np.zeros(int(.58*RATE)),
                       np.ones(int(1.42*RATE))].astype(np.float32)
        segmenter = PauseSegmenter(onset_filtered_detector)
        chunks = feed(segmenter, signal) + segmenter.finish()
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0].audio, signal)

    def test_observed_long_gap_keeps_the_original_pause_threshold(self):
        signal = np.r_[np.ones(int(.5*RATE)), np.zeros(int(.7*RATE)),
                       np.ones(RATE)].astype(np.float32)
        segmenter = PauseSegmenter(onset_filtered_detector)
        chunks = feed(segmenter, signal) + segmenter.finish()
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(len(chunks[0].audio)/RATE, .7)
        self.assertFalse(chunks[0].forced)

    def test_short_onset_is_not_counted_as_long_utterance_silence(self):
        signal = np.r_[np.ones(6*RATE), np.zeros(int(.3*RATE)),
                       np.ones(int(1.2*RATE))].astype(np.float32)
        segmenter = PauseSegmenter(onset_filtered_detector)
        chunks = feed(segmenter, signal) + segmenter.finish()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].audio), len(signal))

    def test_short_response_still_finishes_with_bounded_confirmation(self):
        signal = np.r_[np.ones(int(.3*RATE)), np.zeros(RATE)].astype(np.float32)
        segmenter = PauseSegmenter(onset_filtered_detector)
        chunks = feed(segmenter, signal)
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(len(chunks[0].audio)/RATE, .5)
        self.assertEqual(np.count_nonzero(chunks[0].audio), int(.3*RATE))
        self.assertEqual(segmenter.finish(), [])

    def test_pending_speech_tracks_uncommitted_audio_and_finish(self):
        segmenter = PauseSegmenter(detector)
        self.assertFalse(segmenter.pending_speech)
        segmenter.push(np.ones(2*RATE, dtype=np.float32), 0, 1000)
        self.assertTrue(segmenter.pending_speech)
        segmenter.push(np.zeros(RATE, dtype=np.float32), 2, 1002)
        self.assertFalse(segmenter.pending_speech)
        self.assertEqual(segmenter.finish(), [])
        segmenter = PauseSegmenter(detector)
        feed(segmenter, np.ones(15*RATE, dtype=np.float32))
        self.assertTrue(segmenter.pending_speech)
        segmenter.finish()
        self.assertFalse(segmenter.pending_speech)

    def test_overlap_keeps_distinct_technical_names_with_close_timestamps(self):
        for previous, current in (('C++', ' C#'), ('C#', ' C'),
                                  ('.NET', ' NET'), ('Node.js', ' Nodejs')):
            with self.subTest(previous=previous, current=current):
                assembler = WordAssembler()
                first = AudioChunk(np.ones(15 * RATE), 0, 1000, forced=True)
                before = assembler.consume(first, words((14., 14.5, previous)))
                second = AudioChunk(np.ones(2 * RATE), 14.2, 1014.2, overlap=True)
                after = assembler.consume(second, words((.2, .8, current)))
                self.assertEqual(''.join(row[2] for row in before+after), previous+current)

    def test_overlap_still_drops_replayed_technical_name_with_cosmetic_changes(self):
        assembler = WordAssembler()
        first = AudioChunk(np.ones(15 * RATE), 0, 1000, forced=True)
        before = assembler.consume(first, words((14., 14.5, 'C++')))
        second = AudioChunk(np.ones(2 * RATE), 14.2, 1014.2, overlap=True)
        after = assembler.consume(second, words((.2, .8, ' ｃ＋＋，'), (.8, 1.2, '是什么')))
        self.assertEqual(''.join(row[2] for row in before+after), 'C++是什么')

    def test_hallucination_filter_keeps_ordinary_speech(self):
        self.assertTrue(plausible_segment(Item(text="大模型和小模型有什么区别？", no_speech_prob=.1)))
        self.assertTrue(plausible_segment(Item(text="好好好，继续说", no_speech_prob=.1)))
        self.assertFalse(plausible_segment(Item(text="字幕by索兰娅", no_speech_prob=.9)))
        self.assertFalse(plausible_segment(Item(text="普通文字", compression_ratio=2.5)))
        self.assertFalse(plausible_segment(Item(text="大模型，" * 12)))
        self.assertFalse(plausible_segment(Item(text="先说一点。" + "盖子 " * 16)))

    def test_rejected_segments_never_reach_assembler_or_preview(self):
        chunk = AudioChunk(np.ones(2 * RATE, dtype=np.float32), 0, 1000)
        model = Mock()
        bad = Item(text="大模型，" * 12,
                   words=[Item(start=0, end=1, word="大模型，" * 12)])
        model.transcribe.return_value = ([bad], None)
        assembler = WordAssembler()
        self.assertEqual(decode_chunk(model, assembler, chunk, "zh"), [])
        self.assertEqual(assembler.last_end, -1)
        self.assertGreaterEqual(model.transcribe.call_args.kwargs["vad_parameters"]["min_speech_duration_ms"], 160)
        segmenter = PauseSegmenter(detector)
        segmenter.push(chunk.audio, 0, 1000)
        self.assertEqual(DraftPreview(clock=lambda: 0).render(model, assembler, segmenter, "zh"), "")
        good = Item(text="现在继续讨论", words=[Item(start=0, end=1, word="现在继续讨论")],
                    no_speech_prob=.1, compression_ratio=1.1)
        model.transcribe.return_value = ([good], None)
        self.assertEqual(decode_chunk(model, assembler, chunk, "zh")[0][2], "现在继续讨论")

    def test_confident_short_speech_survives_high_silence_probability(self):
        model = Mock()
        segment = words((0, .3, "对。"))[0]
        segment.no_speech_prob, segment.avg_logprob = .95, -.25
        model.transcribe.return_value = ([segment], None)
        chunk = AudioChunk(np.ones(RATE, dtype=np.float32), 0, 1000)
        self.assertEqual(decode_chunk(model, WordAssembler(), chunk, "zh")[0][2], "对。")
        segment.avg_logprob = -1.2
        self.assertEqual(decode_chunk(model, WordAssembler(), chunk, "zh"), [])
        segment.avg_logprob, segment.compression_ratio = -.25, 3.
        self.assertFalse(plausible_segment(segment))

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
        self.assertIsNone(preview.render(model, assembler, segmenter, "zh", backlog=0.2))
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
        signal = np.arange(1, 15 * RATE + 1, dtype=np.float32)
        feed(segmenter, signal)
        tail = segmenter.finish()
        self.assertEqual(len(tail), 1)
        self.assertEqual(len(tail[0].audio), int(2.4 * RATE))
        np.testing.assert_array_equal(tail[0].audio, signal[-int(2.4 * RATE):])
        self.assertAlmostEqual(tail[0].start, 12.6)
        self.assertAlmostEqual(tail[0].epoch, 1012.6)
        self.assertTrue(tail[0].overlap)
        self.assertFalse(tail[0].forced)
        self.assertEqual(segmenter.finish(), [])

    def test_short_tail_uses_only_needed_context_after_new_speech(self):
        segmenter = PauseSegmenter(detector)
        signal = np.arange(1, int(15.9 * RATE) + 1, dtype=np.float32)
        feed(segmenter, signal)
        tail = segmenter.finish()[0]
        np.testing.assert_array_equal(tail.audio, signal[-int(2.4 * RATE):])
        self.assertAlmostEqual(tail.start, 13.5)
        self.assertAlmostEqual(tail.epoch, 1013.5)

    def test_short_overlap_ending_on_pause_uses_context_without_replaying_next_speech(self):
        segmenter = PauseSegmenter(detector)
        signal = np.r_[np.ones(15 * RATE), np.zeros(RATE)].astype(np.float32)
        chunks = feed(segmenter, signal)
        self.assertEqual(len(chunks), 2)
        tail = chunks[1]
        self.assertTrue(tail.overlap)
        self.assertEqual(len(tail.audio), int(2.4 * RATE))
        self.assertAlmostEqual(tail.start + len(tail.audio) / RATE, 15.2)
        self.assertEqual(np.count_nonzero(tail.audio == 0), int(.2 * RATE))
        self.assertEqual(len(segmenter.tail_prefix), 0)
        fresh = segmenter.push(np.ones(RATE, dtype=np.float32), 20, 1020)
        self.assertEqual(fresh, [])
        next_chunk = segmenter.finish()[0]
        self.assertFalse(next_chunk.overlap)
        self.assertAlmostEqual(next_chunk.start, 20)
        self.assertEqual(len(next_chunk.audio), RATE)

    def test_gap_flushes_context_then_starts_new_timeline(self):
        segmenter = PauseSegmenter(detector)
        feed(segmenter, np.ones(15 * RATE, dtype=np.float32))
        tail = segmenter.push(np.ones(RATE, dtype=np.float32), 30, 1030)[0]
        self.assertAlmostEqual(tail.start, 12.6)
        self.assertAlmostEqual(tail.epoch, 1012.6)
        fresh = segmenter.finish()[0]
        self.assertAlmostEqual(fresh.start, 30)
        self.assertAlmostEqual(fresh.epoch, 1030)
        self.assertEqual(len(fresh.audio), RATE)
        self.assertFalse(fresh.overlap)

    def test_tail_without_new_vad_activity_still_receives_context(self):
        def activity(audio):
            return [dict(start=0, end=len(audio))] if len(audio) >= 2*RATE else []
        segmenter = PauseSegmenter(activity)
        first = segmenter.push(np.ones(15*RATE, dtype=np.float32), 0, 1000)
        self.assertEqual(len(first), 1)
        tail = segmenter.push(np.zeros(int(.6*RATE), dtype=np.float32), 15, 1015)
        self.assertEqual(len(tail), 1)
        self.assertTrue(tail[0].overlap)
        self.assertEqual(len(tail[0].audio), int(2.4*RATE))
        self.assertAlmostEqual(tail[0].start, 13.2)
        self.assertAlmostEqual(tail[0].epoch, 1013.2)
        self.assertEqual(np.count_nonzero(tail[0].audio), int(1.8*RATE))
        self.assertEqual(segmenter.finish(), [])

    def test_later_hard_cut_retains_its_own_context(self):
        segmenter = PauseSegmenter(detector)
        signal = np.arange(1, int(29.2 * RATE) + 1, dtype=np.float32)
        chunks = feed(segmenter, signal)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk.forced for chunk in chunks))
        self.assertEqual(len(chunks[1].audio), 15 * RATE)
        self.assertAlmostEqual(chunks[1].start, 14.2)
        tail = segmenter.finish()[0]
        np.testing.assert_array_equal(tail.audio, signal[-int(2.4 * RATE):])
        self.assertAlmostEqual(tail.start, 26.8)

    def test_context_replay_keeps_deferred_term_once_with_original_time(self):
        segmenter, assembler = PauseSegmenter(detector), WordAssembler()
        first = feed(segmenter, np.ones(15 * RATE, dtype=np.float32))[0]
        before = assembler.consume(first, words((12.8, 13.3, '使用'),
                                               (14., 14.5, '向量'), (14.6, 15., '数据库')))
        tail = segmenter.finish()[0]
        after = assembler.consume(tail, words((.2, .7, '使用'),
                                             (1.4, 1.9, '向量'), (2., 2.4, '数据库')))
        self.assertEqual(''.join(row[2] for row in before + after), '使用向量数据库')
        self.assertAlmostEqual(after[0][0], 12.8)
        self.assertAlmostEqual(after[0][1], 15.)
        self.assertAlmostEqual(after[0][3], 1012.8)
        self.assertIsNone(assembler.pending)
