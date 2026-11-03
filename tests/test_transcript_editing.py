import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from app import App, export_text, plain_text
from transcript_corrections import load_recording, corrections_path


class TranscriptEditingTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('app.GlobalHotkey'), patch('app.App.start_tray'),
                        patch('qa_connection.QAConnection.check'), patch('app.ClipboardWatcher'),
                        patch('app.preferences', return_value={}), patch('app.save_desktop'),
                        patch('app.load_settings', return_value={'provider':'compatible', 'profiles':{}})]
        for item in self.patches:
            item.start()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = App(self.root)
        self.folder = tempfile.TemporaryDirectory()
        self.app.session = Path(self.folder.name)/'interview.jsonl'

    def tearDown(self):
        if self.app.transcript_editor is not None:
            self.app.transcript_editor.destroy()
        self.app.qa.set_enabled(False)
        self.app.qa_connection.close()
        self.app.gpu_check.close()
        self.app.hotkey.close()
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.root.destroy()
        self.folder.cleanup()
        for item in reversed(self.patches):
            item.stop()

    def add(self, text='什么是密等性？'):
        index = len(self.app.session_rows)
        self.app.events.put(('segment', dict(start=index*2, end=index*2+1, text=text, source='system')))
        self.app.poll()

    def test_correction_updates_bubble_selection_copy_export_and_restore(self):
        self.add()
        original = self.app.session.read_bytes()
        self.app.chat.draft('system', '正在讲话')
        self.app.correct_transcript(0, '什么是幂等性？')
        self.assertEqual(plain_text(self.app.rows), '什么是幂等性？')
        self.assertIn('幂等性', export_text(self.app.rows))
        self.assertEqual(self.app.text.get(*self.app.text.tag_ranges('body:0')), '什么是幂等性？')
        self.assertIn('已修正', self.app.chat.bubbles[0].label.cget('text'))
        self.assertIn('draft:system', self.app.chat.bubbles)
        self.assertEqual(self.app.session_rows[0]['text'], '什么是幂等性？')
        self.assertEqual(self.app.session.read_bytes(), original)
        self.app.restore_session(self.app.session)
        self.assertEqual(self.app.rows[0]['text'], '什么是幂等性？')
        self.assertEqual(self.app.corrected_rows, {0})
        self.app.correct_transcript(0, '什么是密等性？')
        self.assertFalse(self.app.corrected_rows)
        self.assertNotIn('已修正', self.app.chat.bubbles[0].label.cget('text'))

    def test_fresh_question_and_context_use_corrected_text(self):
        self.add()
        self.app.correct_transcript(0, '什么是幂等性？')
        self.app.open_qa()
        self.app.qa_connection.state = 'authenticated'
        self.app.qa.ask = Mock()
        self.app.bubble_click(0, False)
        self.app.qa.ask.assert_called_once_with('什么是幂等性？', ['什么是幂等性？'])
        self.app.auto_qa.set(True)
        rows, _ = self.app.session_context_snapshot()
        self.assertIn('用户已修正', rows[0][0])
        self.assertEqual(rows[0][1], '什么是幂等性？')

    def test_old_recording_remains_addressable_after_restart(self):
        self.add()
        first = self.app.session
        with patch('app.RECORDINGS', Path(self.folder.name)), patch('app.save_preferences'), \
                patch.object(self.app.engine, 'start'):
            self.app.capture_mode.set('仅系统声音')
            self.app.devices = [{'index':0}]
            self.app.device.configure(values=['Test'])
            self.app.device.current(0)
            self.app.start()
        second = self.app.session
        self.add('这是第二次录制。')
        self.app.correct_transcript(0, '什么是幂等性？')
        self.assertEqual(load_recording(first).rows[0]['text'], '什么是幂等性？')
        self.assertFalse(corrections_path(second).exists())
        self.assertEqual(load_recording(second).rows[0]['text'], '这是第二次录制。')

    def test_clear_view_preserves_file_row_identity(self):
        self.add('第一段。')
        self.app.clear(preserve_recording=True)
        self.add()
        self.app.correct_transcript(0, '什么是幂等性？')
        result = load_recording(self.app.session)
        self.assertEqual([row['text'] for row in result.rows], ['第一段。', '什么是幂等性？'])
        self.assertEqual(result.corrected, {1})

    def test_failed_save_preserves_ui_and_question_generation(self):
        self.add()
        self.app.qa.active = True
        self.app.qa.source = 'auto'
        generation = self.app.qa.generation
        with patch('transcript_corrections.save_correction', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.app.correct_transcript(0, '什么是幂等性？')
        self.assertEqual(self.app.rows[0]['text'], '什么是密等性？')
        self.assertEqual(self.app.qa.generation, generation)
        self.assertTrue(self.app.qa.active)
        self.assertFalse(self.app.corrected_rows)

    def test_saved_correction_cancels_stale_voice_answer_but_not_clipboard(self):
        self.add()
        self.app.qa.set_enabled(True)
        self.app.qa.feed('什么是密等性？', record_key=0)
        self.app.qa.active = True
        self.app.qa.source = 'auto'
        old = self.app.qa.generation
        cancellation = self.app.qa.cancel
        self.app.correct_transcript(0, '什么是幂等性？')
        self.assertTrue(cancellation.is_set())
        self.assertFalse(self.app.qa.active)
        self.assertGreater(self.app.qa.generation, old)
        self.assertEqual(self.app.qa.pending['system'].parts, ['什么是幂等性？'])
        self.app.handle_qa((old, 'answer', ('旧问题', '迟到的旧答案')))
        self.assertNotIn('迟到', self.app.qa_text.get('1.0','end'))
        self.app.qa.active, self.app.qa.source = True, 'clipboard'
        cancellation = self.app.qa.cancel
        self.app.correct_transcript(0, '什么是事务？')
        self.assertFalse(cancellation.is_set())

    def test_editor_cancel_and_invalid_save_do_not_modify_record(self):
        self.add()
        self.app.open_transcript_editor(0)
        editor = self.app.transcript_editor
        self.assertIsNotNone(editor)
        editor.reset('')
        editor.save()
        self.assertIs(self.app.transcript_editor, editor)
        self.assertIn('未保存', editor.detail.get())
        editor.destroy()
        self.assertEqual(self.app.rows[0]['text'], '什么是密等性？')
        self.assertFalse(corrections_path(self.app.session).exists())

    def test_editor_ime_composition_does_not_commit_partial_input(self):
        self.add()
        self.app.open_transcript_editor(0)
        editor = self.app.transcript_editor
        editor.reset('什么是幂等性？')
        with patch('ime_support.is_composing', return_value=True):
            editor.save()
        self.assertFalse(corrections_path(self.app.session).exists())
        with patch('ime_support.is_composing', return_value=False):
            editor.save()
        self.assertIsNone(self.app.transcript_editor)
        self.assertEqual(self.app.rows[0]['text'], '什么是幂等性？')

    def test_open_history_is_blocked_while_recording(self):
        self.app.busy = True
        with patch('app.filedialog.askopenfilename') as choose:
            self.app.open_recording()
            choose.assert_not_called()

    def test_voice_auto_submission_waits_until_correction_editor_is_closed(self):
        self.add()
        self.app.auto_qa.set(True)
        self.app.toggle_auto_qa()
        self.app.qa_connection.state = 'authenticated'
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel: '本地测试'
        self.app.qa.feed('什么是密等性？', record_key=0)
        self.app.qa.pending['system'].updated -= 10
        self.app.open_transcript_editor(0)
        self.app.poll()
        self.assertFalse(self.app.qa.active)
        self.app.transcript_editor.destroy()
        self.app.poll()
        self.assertTrue(self.app.qa.active)

    def test_editor_select_all_replaces_original_text(self):
        self.add()
        self.root.deiconify()
        self.app.open_transcript_editor(0)
        editor = self.app.transcript_editor.editor
        self.root.update()
        editor.focus_force()
        editor.event_generate('<Control-a>')
        self.root.update()
        editor.tk.call('tk::TextInsert', editor, '什么是幂等性？')
        self.assertEqual(editor.get('1.0', 'end-1c'), '什么是幂等性？')
