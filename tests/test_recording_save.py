import codecs
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import export_text, save_segment


class RecordingSaveTests(unittest.TestCase):
    def test_append_matches_full_export_with_single_bom_and_unicode(self):
        rows = [dict(start=0, end=1, text='中文与C++、.NET', source='system',
                     captured_at='2026-11-02T10:00:01+08:00'),
                dict(start=3, end=4, text='我的补充回答。', source='microphone')]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'interview.jsonl'
            for row in rows:
                save_segment(path, row)
            restored = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(restored, rows)
            txt = path.with_suffix('.txt')
            self.assertEqual(txt.read_text(encoding='utf-8-sig'), export_text(rows))
            self.assertEqual(txt.read_bytes().count(codecs.BOM_UTF8), 1)

    def test_previously_saved_text_remains_untouched_when_more_rows_arrive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'interview.jsonl'
            original = dict(start=0, end=1, text='之前已经保存的第一段。')
            save_segment(path, original)
            prefix = path.with_suffix('.txt').read_bytes()
            for i in range(1, 101):
                save_segment(path, dict(start=i, end=i+1, text=f'继续面试第{i}段。'))
            self.assertTrue(path.with_suffix('.txt').read_bytes().startswith(prefix))
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 101)

    def test_text_save_failure_keeps_structured_original_and_reports_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'interview.jsonl'
            row = dict(start=0, end=1, text='这段结构化原文仍可恢复。')
            real_open = Path.open
            def open_file(target, *args, **kwargs):
                if target.suffix == '.txt':
                    raise OSError('disk write failed')
                return real_open(target, *args, **kwargs)
            with patch.object(Path, 'open', open_file):
                with self.assertRaisesRegex(OSError, 'disk write failed'):
                    save_segment(path, row)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), row)
