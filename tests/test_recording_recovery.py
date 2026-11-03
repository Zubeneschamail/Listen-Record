"""A failed open/export must not damage the practice being reviewed."""
import codecs
import gc
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from app import App, export_text
from transcript_corrections import corrections_path, save_correction


class RecordingRecoveryTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        for name, options in (
                ('app.GlobalHotkey', {}), ('app.App.start_tray', {}),
                ('app.ClipboardWatcher', {}), ('qa_connection.QAConnection.check', {}),
                ('app.save_desktop', {}), ('app.preferences', {'return_value': {}}),
                ('app.load_settings', {'return_value': {'provider': 'compatible', 'profiles': {}}})):
            item = patch(name, **options)
            item.start()
            self.addCleanup(item.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = App(self.root)
        self.path = Path(self.folder.name)/'current.jsonl'
        self.row = dict(start=0, end=2, text='什么是密等性？', source='system',
                        captured_at='2026-11-02T10:00:00+08:00')
        self.write(self.path, [self.row])
        self.path.with_suffix('.qa.jsonl').write_text(json.dumps(
            dict(id='old', question='为什么？', answer='原有完整回答'), ensure_ascii=False)+'\n', encoding='utf-8')
        self.app.restore_session(self.path)
        self.app.composer.input.insert('1.0', '还没有发送的草稿')

    def tearDown(self):
        self.app.qa.set_enabled(False)
        self.app.qa.context_provider = self.app.qa.voice_context_provider = None
        self.app.qa_connection.close()
        self.app.gpu_check.close()
        self.app.hotkey.close()
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.root.destroy()
        self.app = self.root = None
        gc.collect()

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')

    def assert_current_practice(self):
        self.assertEqual(self.app.session, self.path)
        self.assertEqual(self.app.rows, [self.row])
        self.assertEqual(self.app.qa_history, [('为什么？', '原有完整回答')])
        self.assertEqual(self.app.composer.get(), '还没有发送的草稿')
        self.assertEqual(self.app.qa_origins['old'].path, self.path.with_suffix('.qa.jsonl'))
        self.assertIn(self.row['text'], self.app.text.get('1.0', 'end'))

    def test_invalid_source_is_rejected_before_the_current_practice_is_cleared(self):
        bad = self.path.with_name('bad.jsonl')
        for source in (3, ['system'], {'name': 'system'}, 'unknown', None):
            with self.subTest(source=source):
                self.write(bad, [self.row, {**self.row, 'source': source}])
                before = bad.read_bytes()
                with self.assertRaisesRegex(ValueError, '第 2 行'):
                    self.app.restore_session(bad)
                self.assertEqual(bad.read_bytes(), before)
                self.assert_current_practice()

    def test_unexportable_timestamps_are_rejected_even_with_a_valid_capture_clock(self):
        bad = self.path.with_name('bad-time.jsonl')
        changes = [dict(end=None), dict(end='2'), dict(end=True), dict(end=-1),
                   dict(start=float('nan')), dict(end=float('inf')), dict(start=1e308),
                   dict(start=10**500), dict(start=False), dict(captured_at=10),
                   dict(captured_at='not-a-date')]
        for change in changes:
            with self.subTest(change=change):
                self.write(bad, [{**self.row, **change}])
                with self.assertRaisesRegex(ValueError, '第 1 行'):
                    self.app.restore_session(bad)
                self.assert_current_practice()
        missing = {key: value for key, value in self.row.items() if key != 'end'}
        self.write(bad, [missing])
        with self.assertRaises(ValueError):
            self.app.restore_session(bad)
        self.assert_current_practice()

    def test_parse_error_reports_the_physical_line_and_preserves_the_current_view(self):
        bad = self.path.with_name('truncated.jsonl')
        bad.write_text(json.dumps(self.row)+'\n\n{"text":', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '第 3 行'):
            self.app.restore_session(bad)
        self.assert_current_practice()

    def test_utf8_bom_recording_and_corrections_can_be_reviewed_and_exported(self):
        self.path.write_bytes(codecs.BOM_UTF8+self.path.read_bytes())
        # Fingerprints describe the parsed row, independent of the text BOM.
        save_correction(self.app.row_origins[0], '什么是幂等性？')
        sidecar = corrections_path(self.path)
        sidecar.write_bytes(codecs.BOM_UTF8+sidecar.read_bytes())
        original = self.path.read_bytes()
        self.app.restore_session(self.path)
        self.assertEqual(self.app.rows[0]['text'], '什么是幂等性？')
        self.assertEqual(self.app.corrected_rows, {0})
        self.assertIn('00:00:00,000 --> 00:00:02,000', export_text(self.app.rows, True))
        self.assertEqual(self.path.read_bytes(), original)

    def test_export_cannot_replace_a_recording_or_its_qa_and_correction_files(self):
        save_correction(self.app.row_origins[0], '什么是幂等性？')
        self.path.with_suffix('.txt').write_text('原始自动保存的文字', encoding='utf-8-sig')
        sources = [self.path, self.path.with_suffix('.txt'),
                   self.path.with_suffix('.qa.jsonl'), corrections_path(self.path)]
        for target in sources:
            with self.subTest(target=target):
                before = target.read_bytes()
                with patch('app.filedialog.asksaveasfilename', return_value=str(target)), \
                        patch('app.messagebox.showerror') as error:
                    self.app.export()
                self.assertEqual(target.read_bytes(), before)
                error.assert_called_once()

    def test_export_preserves_earlier_recordings_after_starting_a_new_one(self):
        previous = self.path.with_suffix('.txt')
        previous.write_text('原始自动保存的文字', encoding='utf-8-sig')
        before = previous.read_bytes()
        self.app.session = self.path.with_name('next-recording.jsonl')
        with patch('app.filedialog.asksaveasfilename', return_value=str(previous)), \
                patch('app.messagebox.showerror') as error:
            self.app.export()
        error.assert_called_once()
        self.assertEqual(previous.read_bytes(), before)

    def test_partial_write_failure_preserves_the_destination_and_cleans_up(self):
        target = self.path.with_name('review.txt')
        target.write_bytes(b'previous export')
        create_file = tempfile.NamedTemporaryFile
        def create(*args, **kwargs):
            file = create_file(*args, **kwargs)
            write = file.write
            def fail(text):
                write(text[:4])
                raise OSError('disk full during write')
            file.write = fail
            return file
        with patch('app.filedialog.asksaveasfilename', return_value=str(target)), \
                patch('tempfile.NamedTemporaryFile', side_effect=create), \
                patch('app.messagebox.showerror') as error:
            self.app.export()
        error.assert_called_once()
        self.assertEqual(target.read_bytes(), b'previous export')
        self.assertFalse(list(target.parent.glob('*.tmp')))
        self.assert_current_practice()

    def test_failed_export_replacement_preserves_the_previous_destination(self):
        target = self.path.with_name('review.txt')
        target.write_bytes(b'previous exported review')
        with patch('app.filedialog.asksaveasfilename', return_value=str(target)), \
                patch('pathlib.Path.replace', side_effect=OSError('disk full')), \
                patch('app.messagebox.showerror') as error:
            self.app.export()
        error.assert_called_once()
        self.assertEqual(target.read_bytes(), b'previous exported review')
        self.assertFalse(list(target.parent.glob('*.tmp')))
        self.assert_current_practice()

    def test_text_and_subtitle_exports_use_corrected_words_without_changing_sources(self):
        self.app.correct_transcript(0, '什么是幂等性？')
        sources = {path: path.read_bytes() for path in
                   (self.path, self.path.with_suffix('.qa.jsonl'), corrections_path(self.path))}
        for extension in ('txt', 'SRT'):
            target = self.path.with_name('review.'+extension)
            target.write_text('old export', encoding='utf-8')
            with patch('app.filedialog.asksaveasfilename', return_value=str(target)), \
                    patch('app.messagebox.showerror') as error:
                self.app.export()
            error.assert_not_called()
            self.assertEqual(target.read_text(encoding='utf-8-sig'),
                             export_text(self.app.rows, extension == 'SRT'))
            self.assertEqual(target.read_bytes().count(codecs.BOM_UTF8), 1)
        self.assertEqual({path: path.read_bytes() for path in sources}, sources)
        self.assertFalse(list(self.path.parent.glob('*.tmp')))

    def test_legacy_relative_times_and_missing_source_still_open(self):
        legacy = self.path.with_name('legacy.jsonl')
        rows = [dict(start=1.25, end=3.75, text='旧版本的记录', extra='keep')]
        self.write(legacy, rows)
        self.app.restore_session(legacy)
        self.assertEqual(self.app.rows, rows)
        self.assertIn('00:00:01,250 --> 00:00:03,750', export_text(self.app.rows, True))


class PackagedRecordingChecks(unittest.TestCase):
    def test_shipped_recording_checks_cover_validation_bom_and_export_sources(self):
        import app
        from launcher import verify_recording_io
        self.assertEqual(verify_recording_io(app), dict(recording_import_validation=True,
            utf8_bom_recording=True, transcript_export_guard=True, atomic_transcript_export=True))
