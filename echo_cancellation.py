"""Timestamp-aligned AEC before transcription, on a dedicated DSP thread."""
from collections import deque
import math
import queue
import threading
import time

import numpy as np

RATE = 16000
REFERENCE_WAIT = .12


def capture_offset(timing, now, origin, count, rate):
    # PortAudio clocks differ across devices. Translate each ADC timestamp via
    # that callback's current_time into the shared monotonic clock.
    age = count / rate
    current, adc = timing.get('current_time', 0), timing.get('input_buffer_adc_time', 0)
    if current and adc and math.isfinite(current - adc) and 0 <= current - adc < 2:
        age = current - adc
    return max(0., now - origin - age)


def make_processor():
    try:
        from pywebrtc_audio import EchoCanceller
        return EchoCanceller(sample_rate=RATE, num_channels=1, stream_delay_ms=0)
    except Exception as exc:
        raise RuntimeError('回声消除组件无法加载，请重新安装依赖或关闭外放回声消除。') from exc


class EchoAligner:
    def __init__(self, processor=None):
        self.processor = processor if processor is not None else make_processor()
        self.references = deque()
        self.reference_end = -1
        self.mic_end = None

    def add_reference(self, offset, samples):
        start = round(offset * RATE)
        self.references.append((start, samples))
        self.reference_end = start + len(samples)
        # Bounded history: AEC itself maintains the acoustic-path history.
        while self.references and self.references[0][0] + len(self.references[0][1]) < self.reference_end - 2 * RATE:
            self.references.popleft()

    def ready(self, offset, count):
        return self.reference_end >= round(offset * RATE) + count

    def process(self, offset, mic):
        start = round(offset * RATE)
        if self.mic_end is not None and abs(start - self.mic_end) > RATE // 20:
            self.processor.reset()
        self.mic_end = start + len(mic)
        far = np.zeros(len(mic), dtype=np.float32)
        for ref_start, samples in self.references:
            left, right = max(start, ref_start), min(start + len(mic), ref_start + len(samples))
            if left < right:
                far[left-start:right-start] = samples[left-ref_start:right-ref_start]
        # No render packets during WASAPI silence means a zero reference, not
        # a reason to mute the mic or reuse an old speaker frame.
        clean = self.processor.process(np.ascontiguousarray(mic, dtype=np.float32), far)
        if clean.shape != mic.shape or not np.isfinite(clean).all():
            raise RuntimeError('回声消除返回了无效音频。')
        return clean


class EchoCapture:
    def __init__(self, output, stop, rates, convert, processor=None):
        self.output, self.stop, self.rates, self.convert = output, stop, rates, convert
        self.aligner = EchoAligner(processor)
        self.input = queue.Queue(maxsize=100)
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name='echo-cancellation')

    def start(self):
        self.thread.start()

    def submit(self, item):
        self.input.put_nowait((item, time.monotonic()))

    def run(self):
        pending = deque()
        try:
            while not self.stop.is_set() or not self.input.empty() or pending:
                try:
                    (source, offset, samples, epoch), received = self.input.get(timeout=.01)
                except queue.Empty:
                    pass
                else:
                    audio = self.convert(samples, self.rates[source])
                    if source == 'system':
                        self.aligner.add_reference(offset, audio)
                        # Preserve original system samples and their sample rate.
                        self.output.put_nowait((source, offset, samples, epoch))
                    else:
                        pending.append((offset, audio, epoch, received))
                while pending:
                    offset, mic, epoch, received = pending[0]
                    draining = self.stop.is_set() and self.input.empty()
                    if not (draining or self.aligner.ready(offset, len(mic)) or
                            time.monotonic() - received >= REFERENCE_WAIT):
                        break
                    pending.popleft()
                    self.output.put_nowait(('microphone', offset, self.aligner.process(offset, mic), epoch))
        except Exception as exc:
            self.error = '回声消除处理失败，请关闭开关后重试：' + str(exc)
            self.stop.set()
        finally:
            self.done.set()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
