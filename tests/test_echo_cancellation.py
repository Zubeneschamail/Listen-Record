import queue
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.signal import lfilter

from app import mono_16k
from echo_cancellation import EchoAligner, EchoCapture, capture_offset, make_processor


class EchoTests(unittest.TestCase):
    def test_capture_timestamps_share_clock_and_fallback(self):
        timing = {'current_time': 500, 'input_buffer_adc_time': 499.9}
        self.assertAlmostEqual(capture_offset(timing, 20, 10, 4800, 48000), 9.9)
        self.assertAlmostEqual(capture_offset({}, 20, 10, 4800, 48000), 9.9)
        timing['input_buffer_adc_time'] = float('nan')
        self.assertAlmostEqual(capture_offset(timing, 20, 10, 4800, 48000), 9.9)

    def test_reference_alignment_silence_and_discontinuity(self):
        processor = Mock()
        processor.process.side_effect = lambda near, far: near.copy()
        align = EchoAligner(processor)
        align.add_reference(.05, np.ones(1600, np.float32))
        mic = np.full(1600, .1, np.float32)
        np.testing.assert_array_equal(align.process(0, mic), mic)
        far = processor.process.call_args.args[1]
        np.testing.assert_array_equal(far[:800], 0)
        np.testing.assert_array_equal(far[800:], 1)
        align.process(.1, mic)
        processor.reset.assert_not_called()
        align.process(1, mic)
        processor.reset.assert_called_once()
        np.testing.assert_array_equal(processor.process.call_args.args[1], 0)
        for i in range(50):
            align.add_reference(i * .1, mic)
        self.assertLessEqual(len(align.references), 22)

    def test_mic_first_waits_for_reference_and_system_is_untouched(self):
        output, stop = queue.Queue(), threading.Event()
        processor = Mock()
        processor.process.side_effect = lambda near, far: near - far
        worker = EchoCapture(output, stop, {'system': 48000, 'microphone': 16000}, mono_16k, processor)
        worker.start()
        try:
            mic, system = np.ones(1600, np.float32), np.ones((4800, 2), np.float32)
            worker.submit(('microphone', 0, mic, 10))
            worker.submit(('system', 0, system, 10))
            system_item = output.get(timeout=2)
            mic_item = output.get(timeout=2)
            self.assertIs(system_item[2], system)
            self.assertEqual(mic_item[:2], ('microphone', 0))
            self.assertEqual(mic_item[3], 10)
            self.assertLess(np.max(np.abs(mic_item[2][20:-20])), .01)
        finally:
            worker.close()
        self.assertTrue(worker.done.is_set())
        self.assertIsNone(worker.error)

    def test_missing_loopback_never_blocks_mic_and_stop_drains(self):
        output, stop = queue.Queue(), threading.Event()
        processor = Mock()
        processor.process.side_effect = lambda near, far: near.copy()
        worker = EchoCapture(output, stop, {'microphone': 16000}, mono_16k, processor)
        worker.start()
        mic = np.ones(1600, np.float32) * .1
        try:
            worker.submit(('microphone', 0, mic, 10))
            np.testing.assert_array_equal(output.get(timeout=1)[2], mic)
            worker.submit(('microphone', .1, mic, 10.1))
            worker.close()
            np.testing.assert_array_equal(output.get(timeout=1)[2], mic)
            np.testing.assert_array_equal(processor.process.call_args.args[1], 0)
        finally:
            worker.close()

    def test_processor_failure_stops_instead_of_silently_disabling(self):
        output, stop = queue.Queue(), threading.Event()
        processor = Mock()
        processor.process.side_effect = RuntimeError('native error')
        worker = EchoCapture(output, stop, {'microphone': 16000}, mono_16k, processor)
        worker.start()
        worker.submit(('microphone', 0, np.zeros(1600, np.float32), 10))
        self.assertTrue(worker.done.wait(2))
        self.assertTrue(stop.is_set())
        self.assertIn('回声消除处理失败', worker.error)
        self.assertTrue(output.empty())

    def test_native_aec_reduces_delayed_echo_and_preserves_near_only_audio(self):
        rng = np.random.default_rng(42)
        far = lfilter([1, .5], [1, -.7], rng.normal(0, .045, 16000*12)).astype(np.float32)
        mic = np.zeros_like(far)
        mic[800:] = far[:-800] * .6
        processor = make_processor()
        clean = np.concatenate([processor.process(mic[i:i+1600], far[i:i+1600])
                                for i in range(0, len(mic), 1600)])
        attenuation = 10*np.log10(np.mean(mic[-64000:]**2) / max(1e-12, np.mean(clean[-64000:]**2)))
        self.assertGreater(attenuation, 10)
        processor = make_processor()
        near = (.1*np.sin(2*np.pi*250*np.arange(64000)/16000)).astype(np.float32)
        clean = np.concatenate([processor.process(near[i:i+1600], np.zeros(1600, np.float32))
                                for i in range(0, len(near), 1600)])
        self.assertGreater(np.sqrt(np.mean(clean[16000:]**2)), .035)
        self.assertTrue(np.isfinite(clean).all())


if __name__ == '__main__':
    unittest.main()
