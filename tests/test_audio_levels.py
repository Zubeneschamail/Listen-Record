import unittest
import numpy as np
from audio_levels import AudioLevelNormalizer


class LevelTests(unittest.TestCase):
    def test_quiet_audio_boost_and_sudden_loud_audio_limit(self):
        normalizer = AudioLevelNormalizer()
        wave = np.sin(np.arange(1600) * .1).astype(np.float32)
        boosted = normalizer.process(wave * .0005)
        self.assertGreater(np.max(np.abs(boosted)), .01)
        louder = normalizer.process(wave * .5)
        np.testing.assert_allclose(louder, wave * .5)

    def test_silence_and_noise_floor_are_not_amplified(self):
        for level in (0., 1e-7):
            audio = np.full(1600, level, dtype=np.float32)
            np.testing.assert_array_equal(AudioLevelNormalizer().process(audio), audio)

    def test_invalid_and_out_of_range_input_stays_finite(self):
        output = AudioLevelNormalizer().process(np.array([np.nan, np.inf, -12., 10.], np.float32))
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(np.max(np.abs(output)), 1.)
