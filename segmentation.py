"""Pause-first 16 kHz audio segmentation and overlap-aware word assembly."""
from dataclasses import dataclass
from copy import deepcopy
import time
import numpy as np
from faster_whisper.vad import get_speech_timestamps, VadOptions

RATE = 16000


@dataclass
class AudioChunk:
    audio: np.ndarray
    start: float
    epoch: float
    forced: bool = False
    overlap: bool = False


class PauseSegmenter:
    def __init__(self, detector=None, silence=0.6, maximum=15.0, overlap=0.8):
        self.detector = detector or self.detect
        self.silence = round(silence * RATE)
        self.maximum = round(maximum * RATE)
        self.overlap_samples = round(overlap * RATE)
        self.audio = np.empty(0, dtype=np.float32)
        self.start = self.epoch = 0.0
        self.overlapping = False
        self.since_check = 0

    @staticmethod
    def detect(audio):
        return get_speech_timestamps(audio, VadOptions(
            min_silence_duration_ms=100, min_speech_duration_ms=80, speech_pad_ms=0))

    def _advance(self, count):
        self.audio = self.audio[count:].copy()
        self.start += count / RATE
        self.epoch += count / RATE

    def _emit(self, count, forced=False):
        chunk = AudioChunk(self.audio[:count].copy(), self.start, self.epoch,
                           forced=forced, overlap=self.overlapping)
        self._advance(count - self.overlap_samples if forced else count)
        self.overlapping = forced
        return chunk

    def push(self, audio, start, epoch):
        result = []
        if len(self.audio) and start > self.start + len(self.audio) / RATE + 0.3:
            result.extend(self.finish())  # Output device emitted no callbacks across a gap.
        if not len(self.audio):
            self.start, self.epoch = start, epoch
        self.audio = np.concatenate((self.audio, audio))
        self.since_check += len(audio)
        if self.since_check < int(0.2 * RATE) and len(self.audio) < self.maximum:
            return result
        self.since_check = 0
        while len(self.audio):
            ranges = self.detector(self.audio)
            if not ranges:
                if not self.overlapping:
                    # Preserve a little pre-roll for consonants at the next speech onset.
                    self._advance(max(0, len(self.audio) - int(0.25 * RATE)))
                elif len(self.audio) >= self.overlap_samples + self.silence:
                    result.append(self._emit(len(self.audio)))
                break
            # Prefer the first complete utterance, including a short trailing pad.
            cut = None
            for index, speech in enumerate(ranges):
                next_start = ranges[index + 1]["start"] if index + 1 < len(ranges) else len(self.audio)
                if next_start - speech["end"] >= self.silence:
                    cut = speech["end"] + int(0.2 * RATE)
                    break
            if cut is not None and cut <= self.maximum:
                result.append(self._emit(cut))
                continue
            if len(self.audio) >= self.maximum:
                # At the time limit, prefer a short pause in the last five seconds.
                candidates = []
                for index, speech in enumerate(ranges):
                    next_start = ranges[index + 1]["start"] if index + 1 < len(ranges) else len(self.audio)
                    if (10 * RATE <= speech["end"] < self.maximum and
                            next_start - speech["end"] >= int(0.12 * RATE)):
                        candidates.append(min(speech["end"] + int(0.06 * RATE), self.maximum))
                result.append(self._emit(candidates[-1] if candidates else self.maximum,
                                         forced=not candidates))
                continue
            break
        return result

    def finish(self):
        result = []
        if len(self.audio) and (self.overlapping or self.detector(self.audio)):
            result.append(self._emit(len(self.audio)))
        self.audio = np.empty(0, dtype=np.float32)
        self.overlapping = False
        self.since_check = 0
        return result

    def snapshot(self):
        """Read-only copy: previewing must not consume audio awaiting final decoding."""
        if len(self.audio) < int(1.6 * RATE) or not self.detector(self.audio):
            return None
        return AudioChunk(self.audio.copy(), self.start, self.epoch, overlap=self.overlapping)


class WordAssembler:
    """Commit stable words, defer the hard-cut tail and suppress replayed words."""
    def __init__(self):
        self.last_end = -1.0
        self.last_word = ""
        self.pending = None

    def consume(self, chunk, segments):
        output = []
        previous_end, previous_word = self.last_end, self.last_word
        boundary = chunk.start + len(chunk.audio) / RATE - (0.4 if chunk.forced else 0)
        for segment in segments:
            words = segment.words or []
            selected = []
            deferred = False
            for word in words:
                start, end = chunk.start + word.start, chunk.start + word.end
                # End timestamps decide ownership, preserving words straddling a cut.
                if chunk.forced and end > boundary:
                    deferred = True
                    continue
                if chunk.overlap and end <= previous_end + 0.04:
                    continue
                normalized = "".join(c.lower() for c in word.word if c.isalnum())
                if (chunk.overlap and start < previous_end and normalized and
                        normalized == previous_word):
                    continue
                selected.append((start, end, word.word))
                self.last_end = max(self.last_end, end)
                self.last_word = normalized
            if selected:
                row = (selected[0][0], selected[-1][1], "".join(w[2] for w in selected),
                       chunk.epoch + selected[0][0] - chunk.start)
                if self.pending:
                    row = (self.pending[0], row[1], self.pending[2] + row[2], self.pending[3])
                    self.pending = None
                if deferred:
                    # Do not show half of a word/phrase on its own line at a hard cut.
                    self.pending = row
                else:
                    output.append(row)
        if not chunk.forced and self.pending:
            output.append(self.pending)
            self.pending = None
        return output


def decode_chunk(model, assembler, chunk, language):
    segments, _ = model.transcribe(
        chunk.audio, language=language, beam_size=1,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 600},
        word_timestamps=True, condition_on_previous_text=False, temperature=0,
        max_new_tokens=256, repetition_penalty=1.1, no_repeat_ngram_size=3)
    return assembler.consume(chunk, segments)


class DraftPreview:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.last_finished = -float("inf")
        self.last_audio_end = -float("inf")

    def defer(self):
        self.last_finished = self.clock()

    def render(self, model, assembler, segmenter, language, backlog=0, stopping=False):
        end = segmenter.start + len(segmenter.audio) / RATE
        if (stopping or backlog > 0.5 or self.clock() - self.last_finished < 2 or
                end - self.last_audio_end < 1.6):
            return None
        chunk = segmenter.snapshot()
        if chunk is None:
            return None
        self.last_audio_end = end
        try:
            # A disposable assembler prevents provisional recognition from changing
            # final word ownership, deduplication or pending boundary phrases.
            if chunk.overlap or assembler.pending:
                rows = decode_chunk(model, deepcopy(assembler), chunk, language)
                return "".join(row[2] for row in rows)
            # Drafts need no word alignment unless resolving a hard-cut overlap.
            # Omitting that pass reduces CPU work while final decoding stays unchanged.
            segments, _ = model.transcribe(
                chunk.audio, language=language, beam_size=1, vad_filter=True,
                word_timestamps=False, without_timestamps=True,
                condition_on_previous_text=False, temperature=0,
                max_new_tokens=256, repetition_penalty=1.1, no_repeat_ngram_size=3)
            return "".join(segment.text for segment in segments)
        finally:
            self.defer()
