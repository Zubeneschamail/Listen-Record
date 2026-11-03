"""A recoverable torn append must not make earlier interview history unreadable."""
import codecs
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import export_text, save_segment
from qa_records import append_qa, load_qa
from recording_storage import append_bytes, append_text
from transcript_corrections import read_rows
from tests import test_voice_queue as fixtures


class FaultFile:
    def __init__(self, stream, mode): self.stream, self.mode = stream, mode
    def __getattr__(self, name): return getattr(self.stream, name)
    def __enter__(self): self.stream.__enter__(); return self
    def __exit__(self, *args): return self.stream.__exit__(*args)
    def write(self, value):
        if self.mode == 'short': return self.stream.write(value[:max(1, len(value)//3)])
        if self.mode == 'zero': return 0
        if self.mode == 'none': return None
        if self.mode == 'flush': return self.stream.write(value)
        self.stream.write(value[:max(1, len(value)//2)])
        raise OSError('injected partial write')
    def flush(self):
        if self.mode == 'flush': raise OSError('injected flush failure')
        return self.stream.flush()
    def truncate(self, size):
        if self.mode == 'rollback': raise OSError('injected truncate failure')
        return self.stream.truncate(size)


@contextmanager
def fault(path, mode='partial'):
    original = Path.open
    def opened(actual, *args, **kwargs):
        stream = original(actual, *args, **kwargs)
        return FaultFile(stream, mode) if actual == path and args and 'a' in args[0] else stream
    with patch.object(Path, 'open', opened):
        yield


class AppendFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'practice.jsonl'
        self.old = dict(start=0, end=1, text='原有完整的中文记录。')
        self.new = dict(start=1, end=2, text='RAG 与微调的区别。', incomplete=True)

    def test_transcript_partial_write_restores_previous_history_and_allows_explicit_retry(self):
        save_segment(self.path, self.old)
        before = self.path.read_bytes()
        with fault(self.path), self.assertRaisesRegex(OSError, 'partial write'):
            save_segment(self.path, self.new)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(read_rows(self.path), [self.old])
        save_segment(self.path, self.new)
        self.assertEqual(read_rows(self.path), [self.old, self.new])

    def test_qa_rollback_preserves_bom_and_missing_terminal_newline_then_retries(self):
        for bom in (b'', codecs.BOM_UTF8):
            with self.subTest(bom=bool(bom)):
                self.path.unlink(missing_ok=True)
                append_qa(self.path, 'old', '原题', '原回答')
                self.path.write_bytes(bom+self.path.read_bytes().rstrip(b'\r\n'))
                before = self.path.read_bytes()
                with fault(self.path), self.assertRaises(OSError):
                    append_qa(self.path, 'new', '新题', '新回答')
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual([row.key for row in load_qa(self.path)], ['old'])
                append_qa(self.path, 'new', '新题', '新回答')
                self.assertEqual([row.key for row in load_qa(self.path)], ['old', 'new'])

    def test_txt_failure_rolls_back_only_txt_and_keeps_new_structured_original(self):
        save_segment(self.path, self.old)
        text = self.path.with_suffix('.txt')
        before = text.read_bytes()
        with fault(text), self.assertRaises(OSError):
            save_segment(self.path, self.new)
        self.assertEqual(text.read_bytes(), before)
        self.assertEqual(read_rows(self.path), [self.old, self.new])

    def test_first_row_failure_leaves_a_readable_empty_recording(self):
        with fault(self.path), self.assertRaises(OSError):
            save_segment(self.path, self.new)
        self.assertEqual(self.path.read_bytes(), b'')
        self.assertEqual(read_rows(self.path), [])
        self.assertFalse(self.path.with_suffix('.txt').exists())

    def test_positive_short_writes_complete_utf8_without_duplicate_bom(self):
        with fault(self.path, 'short'):
            append_text(self.path, '中文与 RAG\n', bom=True)
            append_text(self.path, '后续文字\n', bom=True)
        self.assertEqual(self.path.read_text('utf-8-sig'), '中文与 RAG\n后续文字\n')
        self.assertEqual(self.path.read_bytes().count(codecs.BOM_UTF8), 1)

    def test_nonprogressing_writes_raise_and_restore_original_bytes(self):
        self.path.write_bytes(b'previous\n')
        for mode in ('zero', 'none'):
            with self.subTest(mode=mode), fault(self.path, mode), self.assertRaises(OSError):
                append_bytes(self.path, b'next\n')
            self.assertEqual(self.path.read_bytes(), b'previous\n')

    def test_flush_failure_restores_original_even_after_the_whole_payload_was_written(self):
        self.path.write_bytes(b'previous\n')
        with fault(self.path, 'flush'), self.assertRaisesRegex(OSError, 'flush failure'):
            append_bytes(self.path, b'next\n')
        self.assertEqual(self.path.read_bytes(), b'previous\n')

    def test_failed_rollback_is_reported_instead_of_claiming_the_file_was_recovered(self):
        self.path.write_bytes(b'previous\n')
        with fault(self.path, 'rollback'), self.assertRaisesRegex(OSError, '未能回滚') as raised:
            append_bytes(self.path, b'next\n')
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertTrue(self.path.read_bytes().startswith(b'previous\n'))
        self.assertNotEqual(self.path.read_bytes(), b'previous\n')

    def test_normal_transcript_keeps_native_newlines_bom_and_interruption_marker(self):
        import os
        save_segment(self.path, self.old)
        save_segment(self.path, self.new)
        expected = codecs.BOM_UTF8+export_text([self.old,self.new]).replace('\n', os.linesep).encode('utf-8')
        self.assertEqual(self.path.with_suffix('.txt').read_bytes(), expected)
        self.assertEqual(read_rows(self.path), [self.old,self.new])

    def test_packaged_guard_exercises_both_journals_and_txt(self):
        import app
        from launcher import verify_append_recovery
        self.assertEqual(verify_append_recovery(app), {'record_append_rollback':True})


class AppendFailureAppTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.VoiceQueueAppTests()
        self.case.setUp()
        self.app = self.case.app
        self.app.auto_qa.set(False)
        self.app.qa.set_enabled(False)

    def tearDown(self):
        self.case.tearDown()
        self.case.doCleanups()

    def add(self, text):
        self.case.add(text)

    def interrupted(self, extension):
        self.add('已经保存的原文。')
        target = self.app.session if extension == 'jsonl' else self.app.session.with_suffix('.txt')
        previous = target.read_bytes()
        with fault(target): self.add('写入失败但仍可见。')
        with patch('app.save_segment') as append:
            self.add('停止时到达的最后一句。')
        append.assert_not_called()
        self.app.events.put(('backlog', 0));self.app.events.put(('done', None));self.app.poll()
        self.assertEqual(target.read_bytes(), previous)
        self.assertEqual(len(read_rows(self.app.session)), 1 if extension == 'jsonl' else 2)
        self.assertEqual(len(self.app.rows), 3)
        self.assertIn('停止时到达的最后一句', export_text(self.app.rows))
        self.assertEqual(set(self.app.row_origins), {0})
        self.assertTrue(self.app.failed and self.app.session_save_failed)
        self.assertIn('自动保存失败', self.app.status.get())
        self.assertIn('自动保存中断', self.app.footer.get())

    def test_jsonl_failure_keeps_visible_tails_without_bad_disk_row_origins(self): self.interrupted('jsonl')
    def test_txt_failure_keeps_structured_row_and_visible_tails_without_more_appends(self): self.interrupted('txt')

    def test_starting_a_new_recording_resets_save_failure_and_uses_new_row_indices(self):
        self.interrupted('jsonl')
        self.app.capture_mode.set('仅系统声音')
        self.app.devices = [dict(index=123,name='fixture')]
        self.app.device.configure(values=['fixture'])
        self.app.device.current(0)
        with patch('app.RECORDINGS', Path(self.case.folder.name)), \
                patch('app.load_capture_preferences', return_value=''), patch.object(self.app.engine,'start'):
            self.app.start()
        self.assertFalse(self.app.session_save_failed or self.app.failed)
        self.add('新一轮已经保存。')
        self.assertEqual(len(read_rows(self.app.session)), 1)
        self.assertEqual(self.app.row_origins[3].index, 0)

    def test_clear_during_same_recording_preserves_failure_but_clear_practice_resets_it(self):
        self.interrupted('jsonl')
        self.app.clear(preserve_recording=True)
        self.assertTrue(self.app.session_save_failed)
        self.app.clear()
        self.assertFalse(self.app.session_save_failed)
