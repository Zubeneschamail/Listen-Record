"""Pause-first 16 kHz audio segmentation and overlap-aware word assembly."""
from dataclasses import dataclass
from copy import deepcopy
import re
import time
import numpy as np
from faster_whisper.vad import get_speech_timestamps, VadOptions
from tokenizers import Tokenizer as TextTokenizer
from transcript_text import transcript_key

RATE = 16000
VAD_PARAMETERS = {"min_silence_duration_ms": 600, "min_speech_duration_ms": 160,
                  "speech_pad_ms": 100}


@dataclass
class AudioChunk:
    audio: np.ndarray
    start: float
    epoch: float
    forced: bool = False
    overlap: bool = False


class PauseSegmenter:
    def __init__(self, detector=None, silence=0.6, maximum=15.0, overlap=0.8,
                 short_speech_pause=None):
        self.detector = detector or self.detect
        self.silence = round(silence * RATE)
        self.short_speech_pause_samples = (None if short_speech_pause is None
                                           else round(short_speech_pause * RATE))
        # A fresh onset can be absent from VAD ranges until it exceeds the
        # 160 ms minimum. Confirm a terminal pause with a little more audio;
        # an already observed gap between two ranges needs no extra wait.
        self.confirmation_samples = round(.24 * RATE)
        self.maximum = round(maximum * RATE)
        self.overlap_samples = round(overlap * RATE)
        # A sub-second tail often lacks enough context to decode the word that
        # was deferred at a hard cut. Retain real preceding audio for a short
        # final chunk, without enlarging every normal streaming overlap.
        self.tail_context_samples = max(self.overlap_samples, round(2.4 * RATE))
        self.tail_prefix = np.empty(0, dtype=np.float32)
        self.audio = np.empty(0, dtype=np.float32)
        self.start = self.epoch = 0.0
        self.overlapping = False
        self.since_check = 0
        self.last_speech_end = None

    @staticmethod
    def detect(audio):
        return get_speech_timestamps(audio, VadOptions(
            min_silence_duration_ms=100, min_speech_duration_ms=160, speech_pad_ms=0))

    @property
    def pending_speech(self):
        """Speech still owned by the buffer, including a hard-cut overlap."""
        return bool(len(self.audio) and self.last_speech_end is not None and
                    self.last_speech_end > self.start)

    def _advance(self, count):
        self.audio = self.audio[count:].copy()
        self.start += count / RATE
        self.epoch += count / RATE

    def _emit(self, count, forced=False):
        prefix = np.empty(0, dtype=np.float32)
        if self.overlapping and not forced and count < self.tail_context_samples:
            needed = self.tail_context_samples - count
            prefix = self.tail_prefix[-needed:]
        audio = np.concatenate((prefix, self.audio[:count]))
        offset = len(prefix) / RATE
        chunk = AudioChunk(audio, self.start - offset, self.epoch - offset,
                           forced=forced, overlap=self.overlapping)
        if forced:
            retained_start = count - self.overlap_samples
            needed = self.tail_context_samples - self.overlap_samples
            self.tail_prefix = self.audio[max(0, retained_start - needed):retained_start].copy()
        else:
            self.tail_prefix = np.empty(0, dtype=np.float32)
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
            if ranges:
                self.last_speech_end = self.start + ranges[-1]['end'] / RATE
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
                # Longer utterances can end at a shorter natural pause; never cut active speech.
                required = min(self.silence, int(0.35 * RATE)) if speech["end"] >= 6 * RATE else self.silence
                # Some backends need following speech to resolve a brief onset.
                # Retain its audio; stopping still flushes a standalone reply.
                if (self.short_speech_pause_samples is not None and
                        speech["end"] - ranges[0]["start"] < int(0.6 * RATE)):
                    required = max(required, self.short_speech_pause_samples)
                if index == len(ranges) - 1:
                    required += self.confirmation_samples
                if next_start - speech["end"] >= required:
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
        self.tail_prefix = np.empty(0, dtype=np.float32)
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
                normalized = transcript_key(word.word)
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

    def finish(self):
        """Drain only the stable prefix held while awaiting a hard-cut tail.

        A failed decoder/device can prevent the next consume call. This keeps
        already selected words without inventing the deferred boundary words.
        """
        row, self.pending = self.pending, None
        return [row] if row else []


def budget_prompt(model, options):
    """Share a token budget between terms and history, reserving 256 output tokens."""
    options = dict(options)
    tokenizer = getattr(model, "hf_tokenizer", None)
    if not isinstance(tokenizer, TextTokenizer):
        return options  # Lightweight model adapters used by segmentation tests.
    budget = max(0, model.max_length - 256 - 8)  # Language/task/timestamp control tokens.
    hotwords = options.get("hotwords") or ""
    history = options.get("initial_prompt")
    # Keep complete UTF-8 text for hotwords; count the actual leading-space encoding.
    limit = min(96, budget) if history else budget
    # A partial technical name is a different hint (PostgreSQL -> PostgreSQ).
    # Keep the user's priority order and remove complete trailing terms only.
    terms = [term.strip() for term in re.split(r'[,，;；\n]+', hotwords) if term.strip()]
    hotwords = ', '.join(terms)
    while hotwords and len(tokenizer.encode(" " + hotwords, add_special_tokens=False).ids) > limit:
        terms.pop()
        hotwords = ', '.join(terms)
    used = len(tokenizer.encode(" " + hotwords.strip(), add_special_tokens=False).ids) if hotwords else 0
    if "hotwords" in options:
        options["hotwords"] = hotwords or None
    if history:
        tokens = tokenizer.encode(" " + history.strip(), add_special_tokens=False).ids if isinstance(history, str) else list(history)
        remaining = budget - used
        options["initial_prompt"] = tokens[-remaining:] if remaining else None
    return options


def plausible_segment(segment):
    """Reject high-confidence silence and runaway decoding before committing words."""
    # Match Whisper's joint silence/confidence decision: a high silence score
    # alone must not erase confidently decoded short or quiet speech.
    confidence = getattr(segment, "avg_logprob", None)
    if ((getattr(segment, "no_speech_prob", 0.) or 0.) >= .8 and
            (confidence is None or confidence <= -1.0)):
        return False
    if (getattr(segment, "compression_ratio", 0.) or 0.) > 2.4:
        return False
    compact = "".join(char.casefold() for char in segment.text if char.isalnum())
    if len(compact) >= 20:
        repeat = re.search(r"(.{2,16}?)\1{4,}", compact)
        if repeat and len(repeat.group()) >= .6 * len(compact):
            return False
    return True


def decode_chunk(model, assembler, chunk, language, guidance=None, backlog=0):
    started = time.monotonic()
    options = budget_prompt(model, guidance.options(chunk.start, backlog) if guidance else {"beam_size": 1})
    segments, _ = model.transcribe(
        chunk.audio, language=language, **options,
        vad_filter=True, vad_parameters=VAD_PARAMETERS,
        word_timestamps=True, condition_on_previous_text=False, temperature=0,
        max_new_tokens=256, repetition_penalty=1.0, no_repeat_ngram_size=0)
    rows = assembler.consume(chunk, (segment for segment in segments if plausible_segment(segment)))
    if guidance:
        rows = [(start, end, guidance.normalize(text), epoch) for start, end, text, epoch in rows]
        guidance.commit(rows, chunk.start + len(chunk.audio) / RATE,
                        time.monotonic() - started, len(chunk.audio) / RATE)
    return rows


class DraftPreview:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.last_finished = -float("inf")
        self.last_audio_end = -float("inf")
        self.cooldown = 2.0

    def defer(self):
        self.last_finished = self.clock()

    def render(self, model, assembler, segmenter, language, backlog=0, stopping=False, guidance=None):
        end = segmenter.start + len(segmenter.audio) / RATE
        # Budget previews more conservatively as the uncommitted utterance grows.
        minimum_new_audio = max(1.6, min(3.0, len(segmenter.audio) / RATE / 4))
        if (stopping or backlog > 0.1 or self.clock() - self.last_finished < self.cooldown or
                end - self.last_audio_end < minimum_new_audio):
            return None
        chunk = segmenter.snapshot()
        if chunk is None:
            return None
        self.last_audio_end = end
        started = self.clock()
        try:
            # A disposable assembler prevents provisional recognition from changing
            # final word ownership, deduplication or pending boundary phrases.
            if chunk.overlap or assembler.pending:
                draft_guidance = deepcopy(guidance)
                if draft_guidance:
                    draft_guidance.rtf = 1  # Preview always uses the inexpensive beam.
                rows = decode_chunk(model, deepcopy(assembler), chunk, language, draft_guidance)
                return "".join(row[2] for row in rows)
            # Drafts need no word alignment unless resolving a hard-cut overlap.
            # Omitting that pass reduces CPU work while final decoding stays unchanged.
            options = budget_prompt(model, guidance.options(chunk.start, draft=True) if guidance else {"beam_size": 1})
            segments, _ = model.transcribe(
                chunk.audio, language=language, **options, vad_filter=True,
                vad_parameters=VAD_PARAMETERS,
                word_timestamps=False, without_timestamps=True,
                condition_on_previous_text=False, temperature=0,
                max_new_tokens=256, repetition_penalty=1.0, no_repeat_ngram_size=0)
            text = "".join(segment.text for segment in segments if plausible_segment(segment))
            return guidance.normalize(text) if guidance else text
        finally:
            self.cooldown = min(6, max(2, (self.clock() - started) * 1.5))
            self.defer()
