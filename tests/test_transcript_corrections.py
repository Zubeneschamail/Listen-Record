import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from transcript_corrections import (load_recording, row_origin, save_correction,
                                     corrections_path, original_row)


class TranscriptCorrectionStoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'interview.jsonl'
        self.rows = [dict(start=0, end=1, source='system', text='什么是密等性？'),
                     dict(start=2, end=3, source='microphone', text='什么是密等性？')]
        self.path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in self.rows), encoding='utf-8')
        self.path.with_suffix('.txt').write_text('原始记录', encoding='utf-8-sig')
        self.origin = row_origin(self.path, 0, self.rows[0])

    def test_overlay_restores_only_the_selected_row_and_preserves_both_originals(self):
        original, txt = self.path.read_bytes(), self.path.with_suffix('.txt').read_bytes()
        self.assertTrue(save_correction(self.origin, '什么是幂等性？'))
        result = load_recording(self.path)
        self.assertEqual([row['text'] for row in result.rows], ['什么是幂等性？', '什么是密等性？'])
        self.assertEqual(result.corrected, {0})
        self.assertEqual(result.origins[0], self.origin)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.path.with_suffix('.txt').read_bytes(), txt)
        self.assertEqual(original_row(self.origin), self.rows[0])

    def test_restoring_original_removes_override_without_deleting_recordings(self):
        save_correction(self.origin, '什么是幂等性？')
        self.assertFalse(save_correction(self.origin, self.rows[0]['text']))
        self.assertEqual(load_recording(self.path).corrected, set())
        self.assertEqual(json.loads(corrections_path(self.path).read_text())['rows'], {})
        self.assertTrue(self.path.is_file())

    def test_new_appended_rows_do_not_invalidate_previous_correction(self):
        save_correction(self.origin, '什么是幂等性？')
        with self.path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(dict(start=4, end=5, text='继续录制。'))+'\n')
        result = load_recording(self.path)
        self.assertEqual(len(result.rows), 3)
        self.assertEqual(result.rows[0]['text'], '什么是幂等性？')
        self.assertFalse(result.warnings)

    def test_changed_original_is_not_given_a_stale_correction(self):
        save_correction(self.origin, '什么是幂等性？')
        before = corrections_path(self.path).read_bytes()
        self.path.write_text(json.dumps(dict(start=0, end=1, text='另一个会话'), ensure_ascii=False)+'\n', encoding='utf-8')
        result = load_recording(self.path)
        self.assertEqual(result.rows[0]['text'], '另一个会话')
        self.assertTrue(result.warnings)
        with self.assertRaises(ValueError):
            save_correction(self.origin, '新修正')
        self.assertEqual(corrections_path(self.path).read_bytes(), before)

    def test_corrupt_corrections_preserve_raw_view_and_are_not_overwritten(self):
        file = corrections_path(self.path)
        file.write_text('{incomplete', encoding='utf-8')
        result = load_recording(self.path)
        self.assertEqual(result.rows, self.rows)
        self.assertTrue(result.warnings)
        with self.assertRaises(ValueError):
            save_correction(self.origin, '新修正')
        self.assertEqual(file.read_text(), '{incomplete')

    def test_failed_atomic_write_leaves_previous_correction_usable(self):
        save_correction(self.origin, '什么是幂等性？')
        before = corrections_path(self.path).read_bytes()
        with patch('transcript_corrections.Path.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                save_correction(self.origin, '什么是事务？')
        self.assertEqual(corrections_path(self.path).read_bytes(), before)
        self.assertEqual(load_recording(self.path).rows[0]['text'], '什么是幂等性？')
        self.assertFalse(list(self.path.parent.glob('*.tmp')))

    def test_empty_or_oversized_replacement_never_creates_a_correction(self):
        for text in (' ', '字'*6001):
            with self.subTest(length=len(text)), self.assertRaises(ValueError):
                save_correction(self.origin, text)
        self.assertFalse(corrections_path(self.path).exists())
