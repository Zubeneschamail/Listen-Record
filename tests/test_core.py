import unittest
from datetime import datetime
import numpy as np
from app import timestamp, export_text, mono_16k, clean_caption, plain_text, row_time, make_segment


class AudioAndExportTests(unittest.TestCase):
    def test_copy_excludes_both_wall_and_relative_timestamps(self):
        rows = [{"start": 2, "end": 4, "text": "第一句。", "captured_at": "2026-09-07T14:35:20+08:00"},
                {"start": 5, "end": 6, "text": "第二句。", "captured_at": "2026-09-07T14:35:23+08:00"}]
        self.assertEqual(plain_text(rows), "第一句。\n第二句。")
        self.assertEqual(row_time(rows[0]), "14:35:20")
        self.assertIn("[14:35:20] 第一句。", export_text(rows))
        self.assertIn("00:00:02,000 --> 00:00:04,000", export_text(rows, True))

    def test_capture_clock_preserves_date_across_midnight(self):
        # Recognition can finish much later; its completion time is never used.
        epoch = datetime(2026, 9, 7, 23, 59, 59).timestamp() + 2
        row = make_segment(2, 4, "跨天。", epoch)
        self.assertEqual(row_time(row), "00:00:01")
        self.assertTrue(row["captured_at"].startswith("2026-09-08T"))
        self.assertAlmostEqual(datetime.fromisoformat(row["captured_at"]).timestamp(), epoch, places=3)

    def test_noise_punctuation_is_not_shown_as_caption(self):
        self.assertEqual(clean_caption("." * 500), "")
        self.assertEqual(clean_caption("你好" + "." * 500), "你好.")

    def test_resampling_stereo_44100_keeps_duration_and_tone(self):
        t = np.arange(44100, dtype=np.float32) / 44100
        tone = np.sin(2 * np.pi * 440 * t)
        result = mono_16k(np.column_stack([tone, tone]), 44100)
        self.assertEqual(result.shape, (16000,))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(np.argmax(np.abs(np.fft.rfft(result))), 440)

    def test_srt_timestamp_carries_and_clamps(self):
        self.assertEqual(timestamp(59.9996, True), "00:01:00,000")
        self.assertEqual(timestamp(-1, True), "00:00:00,000")
        self.assertEqual(timestamp(3661.234, True), "01:01:01,234")

    def test_export_keeps_unicode_and_segment_timing(self):
        rows = [{"start": 1.25, "end": 3.75, "text": "你好，世界。"}]
        self.assertEqual(export_text(rows, True),
                         "1\n00:00:01,250 --> 00:00:03,750\n你好，世界。\n")
        self.assertIn("你好，世界。", export_text(rows))


if __name__ == "__main__":
    unittest.main()
