"""Bounded per-source gain for quiet loopback audio before speech detection."""
import numpy as np


class AudioLevelNormalizer:
    def __init__(self):
        self.gain = None

    def process(self, audio):
        audio = np.nan_to_num(audio, nan=0., posinf=0., neginf=0.)
        if not audio.size:
            return audio
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio)))
        # Do not raise silence/the numerical noise floor into audible noise.
        if rms < 1e-5:
            return audio
        target = min(64., max(1., .015 / rms), .98 / max(peak, 1e-9))
        if self.gain is None or target < self.gain:
            self.gain = target  # Immediately protect louder speech from clipping.
        else:
            self.gain += .2 * (target - self.gain)
        return np.clip(audio * self.gain, -1., 1.).astype(np.float32)
