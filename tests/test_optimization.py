import unittest
from unittest.mock import patch, Mock
import numpy as np
from model_runtime import load_model
from recognition import RecognitionContext
from segmentation import PauseSegmenter, RATE
from tests.test_segmentation import detector, feed

class OptimizationTests(unittest.TestCase):
    def test_gpu_warmup_failure_falls_back_before_capture(self):
        gpu, cpu = Mock(), Mock()
        gpu.transcribe.side_effect = RuntimeError('missing DLL')
        cpu.transcribe.return_value = (iter([]), None)
        with patch('model_runtime.prepare_cuda'), patch('model_runtime.ctranslate2.get_cuda_device_count', return_value=1), patch('model_runtime.WhisperModel', side_effect=[gpu, cpu]) as create:
            self.assertIs(load_model('small'), cpu)
            self.assertEqual([c.kwargs['device'] for c in create.call_args_list], ['cuda', 'cpu'])
            cpu.transcribe.assert_called_once()

    def test_explicit_corrections_are_non_cascading_and_prompt_uses_target(self):
        context = RecognitionContext('大模形=大模型, 大模型=模型, MCP')
        self.assertEqual(context.normalize('大模形和大模型'), '大模型和模型')
        self.assertNotIn('=', context.options(0)['hotwords'])
        self.assertEqual(RecognitionContext('大模型').normalize('大模形'), '大模形')

    def test_long_phrase_can_finalize_on_short_pause_without_cutting_speech(self):
        chunks = feed(PauseSegmenter(detector), np.r_[np.ones(7*RATE), np.zeros(int(.4*RATE))].astype(np.float32))
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].forced)
        self.assertAlmostEqual(len(chunks[0].audio)/RATE, 7.2)
        self.assertFalse(feed(PauseSegmenter(detector), np.ones(9*RATE, dtype=np.float32)))

    def test_final_keeps_search_quality_during_small_backlog(self):
        context = RecognitionContext()
        self.assertEqual(context.options(0, backlog=.7)['beam_size'], 3)
        self.assertEqual(context.options(0, backlog=2)['beam_size'], 1)
