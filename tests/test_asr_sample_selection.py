from collections import Counter
import unittest

from tools.prepare_code_switch_samples import select


class ASRSampleSelectionTests(unittest.TestCase):
    def test_unknown_annotations_are_excluded_before_even_sampling(self):
        rows = []
        for speaker in (3, 17):
            for index in range(20):
                rows.append({'row': dict(original_speaker_id=speaker, duration=3,
                                         language='mixed', transcription='hello中文'+str(index))})
            rows.insert(0, {'row': dict(original_speaker_id=speaker, duration=3,
                                       language='mixed', transcription='hello [UNK] 中文')})
        selected = select(rows, 'mixed', 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(Counter(item['row']['original_speaker_id'] for item in selected), {3: 4, 17: 4})
        self.assertTrue(all('[UNK]' not in item['row']['transcription'] for item in selected))

    def test_unavailable_language_sample_count_is_reported(self):
        with self.assertRaises(ValueError):
            select([], 'mixed', 8)

    def test_sparse_speakers_return_unused_slots_to_remaining_candidates(self):
        rows = []
        for speaker, count in ((1, 1), (2, 1), (3, 20)):
            for index in range(count):
                rows.append({'row': dict(original_speaker_id=speaker, duration=3,
                                         language='mixed', transcription=f'技术词{index}'),
                             'row_idx': len(rows)})
        selected = select(rows, 'mixed', 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({item['row_idx'] for item in selected}), 8)
        self.assertEqual(Counter(item['row']['original_speaker_id'] for item in selected), {1: 1, 2: 1, 3: 6})
        # The last speaker's quota still covers both ends of the source order.
        self.assertEqual(selected[-1]['row_idx'], rows[-1]['row_idx'])

    def test_zero_and_negative_sample_requests(self):
        self.assertEqual(select([], 'mixed', 0), [])
        with self.assertRaises(ValueError):
            select([], 'mixed', -1)


if __name__ == '__main__':
    unittest.main()
