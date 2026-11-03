import json
import gc
import re
from pathlib import Path
import tempfile
import threading
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from recognition import RecognitionContext, read_preferences, LEGACY_DEFAULT_HOTWORDS
from recognition_rules import load_rule_preferences, prepare_rule, save_rule
from app import App
from transcript_corrections import load_recording
from transcript_editor import CorrectionRuleEditor, place_editor


class RulePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'settings.json'

    def write(self, hotwords, **extra):
        self.path.write_text(json.dumps(dict(hotwords=hotwords, **extra), ensure_ascii=False), encoding='utf-8')
        return load_rule_preferences(self.path)

    def test_new_explicit_rule_is_active_after_loading_new_context(self):
        state = load_rule_preferences(self.path)
        self.assertEqual(state.hotwords, '')
        self.assertIsNone(state.fingerprint)
        save_rule(self.path, '密等性', '幂等性', state.fingerprint)
        self.assertEqual(RecognitionContext(read_preferences(self.path)).normalize('密等性和密钥'), '幂等性和密钥')

    def test_other_aliases_terms_and_json_fields_are_preserved(self):
        state = self.write('lang chain=LangGraph, C++, C#, .NET', schema_note={'keep': True})
        value = save_rule(self.path, '密等性', '幂等性', state.fingerprint)
        self.assertEqual(value, '密等性=幂等性, lang chain=LangGraph, C++, C#, .NET')
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))['schema_note'], {'keep': True})
        self.assertEqual(RecognitionContext(value).normalize('lang chain与c + +'), 'LangGraph与C++')

    def test_conflicting_alias_requires_explicit_replacement(self):
        state = self.write('foo=Old, FOO=Older, Redis, sql=SQL')
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, '已有规则'):
            save_rule(self.path, 'Foo', 'New', state.fingerprint)
        self.assertEqual(self.path.read_bytes(), original)
        value = save_rule(self.path, 'Foo', 'New', state.fingerprint, replace=True)
        self.assertEqual(value, 'Foo=New, Redis, sql=SQL')

    def test_identical_alias_is_idempotent_without_rewriting_file(self):
        state = self.write('foo=Bar, Redis')
        original = self.path.read_bytes()
        self.assertEqual(save_rule(self.path, 'FOO', 'Bar', state.fingerprint), state.hotwords)
        self.assertEqual(self.path.read_bytes(), original)

    def test_invalid_aliases_do_not_change_preferences(self):
        state = self.write('MCP')
        original = self.path.read_bytes()
        for wrong, right in (('', '词'), ('词', ''), ('词', '词'), ('a=b', 'c'),
                             ('a,b', 'c'), ('a', 'b；c'), ('a\nb', 'c'), ('旧'*150, '新'*150)):
            with self.subTest(wrong=wrong):
                with self.assertRaises(ValueError):
                    save_rule(self.path, wrong, right, state.fingerprint)
                self.assertEqual(self.path.read_bytes(), original)

    def test_full_or_oversized_word_lists_are_never_silently_trimmed(self):
        for value in (', '.join('term'+str(i) for i in range(24)), 'x'*250, 'x'*300):
            with self.subTest(value=value[:30]):
                state = self.write(value)
                original = self.path.read_bytes()
                with self.assertRaisesRegex(ValueError, '词库已满'):
                    save_rule(self.path, '密等性', '幂等性', state.fingerprint)
                self.assertEqual(self.path.read_bytes(), original)

    def test_corrupt_existing_preferences_are_not_replaced(self):
        for raw in (b'{', b'[]', b'{"hotwords":null}', b'\xff'):
            self.path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, '无法读取'):
                load_rule_preferences(self.path)
            with self.assertRaises(ValueError):
                save_rule(self.path, '密等性', '幂等性', None)
            self.assertEqual(self.path.read_bytes(), raw)

    def test_stale_dialog_does_not_overwrite_a_newer_word_list(self):
        state = self.write('Redis')
        self.write('Redis, 用户后来添加的词')
        newer = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, '发生变化'):
            save_rule(self.path, '密等性', '幂等性', state.fingerprint)
        self.assertEqual(self.path.read_bytes(), newer)

    def test_json_formatting_changes_do_not_create_a_false_edit_conflict(self):
        state = self.write('密等性=幂等性, Redis', note=True)
        self.path.write_text(json.dumps({'note':True,'hotwords':state.hotwords},
                                       ensure_ascii=False,indent=2), encoding='utf-8')
        value = save_rule(self.path, 'lang chain', 'LangGraph', state.fingerprint)
        self.assertEqual(value, 'lang chain=LangGraph, 密等性=幂等性, Redis')

    def test_atomic_replace_failure_keeps_previous_file(self):
        state = self.write('Redis')
        original = self.path.read_bytes()
        with patch('recognition_rules.Path.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                save_rule(self.path, '密等性', '幂等性', state.fingerprint)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(list(self.path.parent.glob('*.tmp')))

    def test_legacy_migration_and_case_only_spelling_rules(self):
        state = self.write(LEGACY_DEFAULT_HOTWORDS)
        self.assertEqual(state.hotwords, '')
        value = save_rule(self.path, 'ai', 'AI', state.fingerprint)
        self.assertEqual(value, 'ai=AI')
        self.assertEqual(RecognitionContext(value).normalize('ai和aim'), 'AI和aim')
        self.assertEqual(prepare_rule('a=b', 'long old name', 'LongCanonicalName'),
                         'long old name=LongCanonicalName, a=b')

    def test_editor_stays_inside_parent_monitor_even_with_negative_coordinates(self):
        for bounds, position in (((0,0,1000,700),(900,600)), ((-1000,0,0,700),(-100,600))):
            window, parent = Mock(), Mock()
            window.winfo_fpixels.return_value = 96
            parent.winfo_rootx.return_value, parent.winfo_rooty.return_value = position
            parent.winfo_width.return_value, parent.winfo_height.return_value = 300,200
            with patch('window_effects.work_area', return_value=bounds):
                place_editor(window,parent,520,470)
            match = re.fullmatch(r'520x470\+(-?\d+)\+(-?\d+)',window.geometry.call_args.args[0])
            self.assertIsNotNone(match)
            x,y = map(int,match.groups())
            self.assertGreaterEqual(x,bounds[0])
            self.assertGreaterEqual(y,bounds[1])
            self.assertLessEqual(x+520+24,bounds[2])
            self.assertLessEqual(y+470+48,bounds[3])


class RememberRuleAppTests(unittest.TestCase):
    def setUp(self):
        gc.collect()  # Dispose old Tk cycles on the test/UI thread, never a Q&A thread.
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        for target, options in (
                ('app.GlobalHotkey', {}), ('app.App.start_tray', {}), ('app.ClipboardWatcher', {}),
                ('qa_connection.QAConnection.check', {}), ('app.save_desktop', {}),
                ('app.preferences', {'return_value': {}}),
                ('app.load_settings', {'return_value': {'provider':'compatible','profiles':{}}}),
                ('app.DATA', {'new':self.path}), ('settings_view.DATA', {'new':self.path})):
            mocked = patch(target, **options)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = App(self.root)
        self.app.session = self.path/'recording.jsonl'
        self.prefs = self.path/'recognition-settings.json'
        self.app.events.put(('segment', dict(start=0,end=1,text='什么是密等性？',source='system')))
        self.app.poll()

    def tearDown(self):
        if self.app.transcript_editor is not None:
            self.app.transcript_editor.destroy()
        self.app.qa.set_enabled(False)
        for worker in threading.enumerate():
            if worker.name == 'qa-worker':
                worker.join(timeout=3)
        self.app.qa.context_provider = self.app.qa.voice_context_provider = None
        self.app.qa_connection.close()
        self.app.gpu_check.close()
        self.app.hotkey.close()
        for timer in self.root.tk.call('after','info'):
            self.root.after_cancel(timer)
        self.root.destroy()
        self.folder.cleanup()
        self.app = self.root = None
        gc.collect()

    def open_rule(self):
        self.app.open_transcript_editor(0)
        editor = self.app.transcript_editor
        editor.reset('什么是幂等性？')
        editor.remember.set(True)
        editor.save()
        self.assertIsInstance(self.app.transcript_editor, CorrectionRuleEditor)
        return self.app.transcript_editor

    def test_correction_does_not_create_rules_unless_explicitly_requested(self):
        self.app.open_transcript_editor(0)
        editor = self.app.transcript_editor
        self.assertFalse(editor.remember.get())
        editor.reset('什么是幂等性？')
        editor.save()
        self.assertIsNone(self.app.transcript_editor)
        self.assertFalse(self.prefs.exists())

    def test_rule_fields_are_explicit_and_cancel_preserves_saved_paragraph(self):
        original = self.app.session.read_bytes()
        editor = self.open_rule()
        self.assertEqual(editor.wrong.get(), '')
        self.assertEqual(editor.right.get(), '')
        editor.destroy()
        self.assertEqual(load_recording(self.app.session).rows[0]['text'], '什么是幂等性？')
        self.assertEqual(self.app.session.read_bytes(), original)
        self.assertFalse(self.prefs.exists())

    def test_rule_save_preserves_existing_terms_and_next_capture_uses_it(self):
        self.prefs.write_text(json.dumps({'hotwords':'C++, lang chain=LangGraph'}), encoding='utf-8')
        old = RecognitionContext('C++, lang chain=LangGraph')
        editor = self.open_rule()
        editor.wrong.set('密等性')
        editor.right.set('幂等性')
        editor.save()
        self.assertIsNone(self.app.transcript_editor)
        value = self.app.hotwords.get()
        self.assertEqual(value, '密等性=幂等性, C++, lang chain=LangGraph')
        self.assertEqual(old.normalize('密等性'), '密等性')
        self.assertEqual(RecognitionContext(read_preferences(self.prefs)).normalize('密等性'), '幂等性')
        with patch('app.RECORDINGS',self.path), patch.object(self.app.engine,'start') as start:
            self.app.capture_mode.set('仅系统声音')
            self.app.devices = [{'index':0}]
            self.app.device.configure(values=['Test'])
            self.app.device.current(0)
            self.app.start()
        self.assertEqual(start.call_args.args[3], value)

    def test_rule_save_failure_keeps_draft_and_does_not_change_transcript_again(self):
        editor = self.open_rule()
        editor.wrong.set('密等性')
        editor.right.set('幂等性')
        with patch('recognition_rules.save_rule', side_effect=OSError('disk full')):
            editor.save()
        self.assertIs(self.app.transcript_editor, editor)
        self.assertIn('规则未保存', editor.detail.get())
        self.assertEqual(self.app.rows[0]['text'], '什么是幂等性？')
        self.assertEqual(self.app.hotwords.get(), '')
        self.assertFalse(self.prefs.exists())

    def test_existing_rule_update_is_explicit_and_editing_clears_replace_choice(self):
        self.prefs.write_text(json.dumps({'hotwords':'密等性=旧词, Redis'}), encoding='utf-8')
        editor = self.open_rule()
        editor.wrong.set('密等性')
        editor.right.set('幂等性')
        editor.save()
        self.assertIs(self.app.transcript_editor, editor)
        self.assertIn('已有规则', editor.detail.get())
        editor.replace_existing.set(True)
        editor.right.set('新词')
        self.assertFalse(editor.replace_existing.get())
        editor.right.set('幂等性')
        editor.replace_existing.set(True)
        editor.save()
        self.assertEqual(read_preferences(self.prefs), '密等性=幂等性, Redis')

    def test_stale_rule_failure_survives_ordinary_desktop_save(self):
        self.prefs.write_text(json.dumps({'hotwords':'Redis'}), encoding='utf-8')
        self.app.hotwords.set('Redis')
        editor = self.open_rule()
        newer = json.dumps({'hotwords':'Redis, 用户后来添加的词', 'note':True}, ensure_ascii=False).encode('utf-8')
        self.prefs.write_bytes(newer)
        editor.wrong.set('密等性')
        editor.right.set('幂等性')
        editor.save()
        self.assertIn('发生变化', editor.detail.get())
        editor.destroy()
        self.app.save_desktop_settings()
        self.assertEqual(self.prefs.read_bytes(), newer)

    def test_explicit_settings_save_still_commits_word_list(self):
        self.app.hotwords.set('密等性=幂等性, C++')
        self.app.save_desktop_settings(force=True)
        self.assertEqual(read_preferences(self.prefs), '密等性=幂等性, C++')

    def test_ime_and_clear_do_not_commit_unconfirmed_rule_fields(self):
        editor = self.open_rule()
        editor.wrong.set('密等性')
        editor.right.set('幂等性')
        with patch('ime_support.is_composing', return_value=True):
            editor.save()
        self.assertIs(self.app.transcript_editor, editor)
        self.assertFalse(self.prefs.exists())
        self.app.clear()
        self.assertIsNone(self.app.transcript_editor)
        self.assertFalse(self.prefs.exists())

    def test_failed_paragraph_save_never_opens_or_saves_rule(self):
        self.app.open_transcript_editor(0)
        editor = self.app.transcript_editor
        editor.reset('什么是幂等性？')
        editor.remember.set(True)
        with patch('transcript_corrections.save_correction', side_effect=OSError('disk full')):
            editor.save()
        self.assertIs(self.app.transcript_editor, editor)
        self.assertNotIsInstance(editor, CorrectionRuleEditor)
        self.assertEqual(self.app.rows[0]['text'], '什么是密等性？')
        self.assertFalse(self.prefs.exists())

    def test_voice_auto_submission_waits_until_rule_dialog_is_closed(self):
        editor = self.open_rule()
        self.app.auto_qa.set(True)
        self.app.toggle_auto_qa()
        self.app.qa_connection.state = 'verified'
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel: '本地测试'
        self.app.qa.feed('什么是幂等性？', record_key=0)
        self.app.qa.pending['system'].updated -= 10
        self.app.poll()
        self.assertFalse(self.app.qa.active)
        editor.destroy()
        self.app.poll()
        self.assertTrue(self.app.qa.active)

    def test_closing_rule_window_removes_tcl_variable_callbacks(self):
        editor = self.open_rule()
        self.assertTrue(editor.wrong.trace_info())
        self.assertTrue(editor.right.trace_info())
        editor.destroy()
        self.assertFalse(editor.wrong.trace_info())
        self.assertFalse(editor.right.trace_info())
