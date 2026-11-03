"""Real preference files at capture start and settings transaction boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recognition import (RecognitionContext,load_capture_preferences,save_preferences,
                         LEGACY_DEFAULT_HOTWORDS)
import test_recognition_rules as rule_fixture


class PreferenceFileTests(unittest.TestCase):
    def test_capture_read_is_bounded_read_only_and_migrates_legacy_in_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            self.assertEqual(load_capture_preferences(path),'')
            self.assertFalse(path.exists())
            for text,expected in [('密等性=幂等性, Redis','密等性=幂等性, Redis'),
                                  (LEGACY_DEFAULT_HOTWORDS,''),(', '.join('term'+str(i) for i in range(30)),None)]:
                path.write_text(json.dumps(dict(hotwords=text,note={'keep':True}),ensure_ascii=False),encoding='utf-8')
                before=path.read_bytes()
                with patch('recognition.Path.replace',side_effect=AssertionError('Capture must not write')):
                    value=load_capture_preferences(path)
                self.assertEqual(path.read_bytes(),before)
                if expected is not None:self.assertEqual(value,expected)
                self.assertLessEqual(len(value),256)
                self.assertLessEqual(len(value.split(', ')),24)

    def test_invalid_saved_preferences_raise_without_replacing_or_clearing_them(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            for raw in [b'{',b'[]',b'null',b'{"hotwords":null}',b'\xff']:
                path.write_bytes(raw)
                with self.assertRaises(ValueError):load_capture_preferences(path)
                self.assertEqual(path.read_bytes(),raw)
            with patch('recognition_rules.Path.read_bytes',side_effect=PermissionError('unreadable')):
                with self.assertRaises(PermissionError):load_capture_preferences(path)

    def test_explicit_save_preserves_unrelated_json_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            path.write_text(json.dumps(dict(hotwords='Old',schema=2,note={'keep':[1,2]})),encoding='utf-8')
            save_preferences(path,'密等性=幂等性，RAG')
            self.assertEqual(json.loads(path.read_text('utf-8')),
                             dict(hotwords='密等性=幂等性, RAG',schema=2,note={'keep':[1,2]}))

    def test_failed_explicit_replace_keeps_previous_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            path.write_bytes(b'{"hotwords":"Redis","note":true}')
            before=path.read_bytes()
            with patch('recognition.Path.replace',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):save_preferences(path,'New')
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(folder).iterdir()),[path])


class PreferenceAppTests(unittest.TestCase):
    def setUp(self):
        self.fixture=rule_fixture.RememberRuleAppTests('test_explicit_settings_save_still_commits_word_list')
        self.fixture.setUp()
        self.app,self.path,self.prefs=self.fixture.app,self.fixture.path,self.fixture.prefs
        self.addCleanup(self.cleanup_app)
        for name,options in [('settings_session.DATA',dict(new=self.path)),
                             ('reference_settings.save_settings',{}),('app.messagebox.showerror',{})]:
            mocked=patch(name,**options)
            value=mocked.start()
            self.addCleanup(mocked.stop)
            if name=='app.messagebox.showerror':self.error=value
            if name=='reference_settings.save_settings':self.reference_save=value
        self.app.capture_mode.set('仅系统声音')
        self.app.devices=[{'index':0}]
        self.app.device.configure(values=['Test'])
        self.app.device.current(0)

    def cleanup_app(self):
        self.fixture.tearDown()
        self.fixture.doCleanups()

    def write(self,value,**extra):
        self.prefs.write_text(json.dumps(dict(hotwords=value,**extra),ensure_ascii=False),encoding='utf-8')
        return self.prefs.read_bytes()

    def start(self):
        capture=patch.object(self.app.engine,'start')
        writer=patch('app.save_preferences')
        with capture as begin,writer as save,patch('app.RECORDINGS',self.path/'new-recordings'):
            self.app.start()
        save.assert_not_called()
        return begin

    def test_start_uses_external_latest_rules_and_never_writes_old_ui_value(self):
        self.app.hotwords.set('Redis')
        before=self.write('密等性=幂等性, Redis',note={'keep':'external'})
        begin=self.start()
        self.assertEqual(begin.call_args.args[3],'密等性=幂等性, Redis')
        self.assertEqual(RecognitionContext(begin.call_args.args[3]).normalize('密等性'),'幂等性')
        self.assertEqual(self.prefs.read_bytes(),before)
        self.assertEqual(self.app.hotwords.get(),'密等性=幂等性, Redis')

    def test_missing_word_file_stays_missing_and_does_not_revive_old_rules(self):
        self.app.hotwords.set('old=Stale')
        begin=self.start()
        self.assertEqual(begin.call_args.args[3],'')
        self.assertFalse(self.prefs.exists())

    def test_invalid_or_unreadable_word_file_blocks_before_recording_or_capture(self):
        for raw in [b'{',b'[]',b'{"hotwords":null}',b'\xff']:
            self.prefs.write_bytes(raw)
            begin=self.start()
            begin.assert_not_called()
            self.assertFalse((self.path/'new-recordings').exists())
            self.assertEqual(self.prefs.read_bytes(),raw)
        with patch('recognition_rules.Path.read_bytes',side_effect=PermissionError('read denied')):
            self.start().assert_not_called()
        self.assertIn('无法读取',self.error.call_args.args[0])
        self.assertFalse(self.app.busy)

    def test_settings_refresh_latest_and_cancel_draft_does_not_erase_saved_rule(self):
        self.app.hotwords.set('Old UI')
        before=self.write('密等性=幂等性, Redis')
        self.app.toggle_settings()
        self.assertEqual(self.app.hotwords.get(),'密等性=幂等性, Redis')
        self.app.hotwords.set('unsaved draft')
        self.app.hide_settings()
        self.assertEqual(self.app.hotwords.get(),'密等性=幂等性, Redis')
        self.assertEqual(self.prefs.read_bytes(),before)
        self.assertEqual(self.start().call_args.args[3],'密等性=幂等性, Redis')

    def test_unrelated_setting_save_leaves_newer_external_word_file_untouched(self):
        self.write('Redis')
        self.app.toggle_settings()
        before=self.write('密等性=幂等性, Redis',note=True)
        self.app.font_size.set(12)
        self.app.save_settings_dialog()
        self.assertFalse(self.app.settings_visible)
        self.assertEqual(self.prefs.read_bytes(),before)
        self.assertEqual(self.start().call_args.args[3],'密等性=幂等性, Redis')

    def test_stale_edited_draft_is_rejected_before_other_settings_are_written(self):
        self.write('Redis')
        self.app.toggle_settings()
        self.app.hotwords.set('Redis, RAG')
        self.app.settings_auto_qa.set(True)
        before=self.write('密等性=幂等性, Redis')
        with patch('app.save_desktop') as desktop,patch.object(self.app.clipboard_watcher,'start') as watch:
            self.app.save_settings_dialog()
        self.assertTrue(self.app.settings_visible)
        self.assertEqual(self.app.hotwords.get(),'Redis, RAG')
        self.assertEqual(self.prefs.read_bytes(),before)
        self.assertIn('词库未保存',self.error.call_args.args[0])
        desktop.assert_not_called()
        self.reference_save.assert_not_called()
        watch.assert_not_called()
        self.assertFalse(self.app.auto_qa.get())

    def test_update_between_dialog_read_and_snapshot_creation_is_detected(self):
        from recognition import load_wordlist_snapshot
        for initial in ('Redis',None):
            with self.subTest(initial=initial):
                if initial is None:self.prefs.write_bytes(b'{broken')
                else:self.write(initial)
                latest=None
                def read_then_update(path):
                    nonlocal latest
                    try:return load_wordlist_snapshot(path)
                    finally:latest=self.write('密等性=幂等性, Redis',note='newer')
                with patch('app.load_wordlist_snapshot',side_effect=read_then_update):
                    self.app.toggle_settings()
                self.app.hotwords.set('replacement draft')
                self.app.save_settings_dialog()
                self.assertTrue(self.app.settings_visible)
                self.assertEqual(self.prefs.read_bytes(),latest)
                self.assertEqual(self.app.hotwords.get(),'replacement draft')
                self.app.hide_settings()

    def test_reference_failure_does_not_roll_back_an_untouched_external_word_file(self):
        self.write('Redis')
        desktop=self.path/'desktop-settings.json'
        desktop.write_bytes(b'original desktop')
        self.app.toggle_settings()
        latest=json.dumps(dict(hotwords='密等性=幂等性, Redis',note=True),ensure_ascii=False).encode('utf-8')
        def write_desktop(values):
            desktop.write_bytes(b'new desktop')
            self.prefs.write_bytes(latest)  # A separate writer updates the untouched file.
        with patch('app.save_desktop',side_effect=write_desktop):
            self.reference_save.side_effect=OSError('reference write failed')
            self.app.save_settings_dialog()
        self.assertTrue(self.app.settings_visible)
        self.assertEqual(desktop.read_bytes(),b'original desktop')
        self.assertEqual(self.prefs.read_bytes(),latest)

    def test_explicit_word_edit_preserves_metadata_and_repairs_unchanged_corrupt_file(self):
        self.write('Redis',note={'keep':True})
        self.app.toggle_settings()
        self.app.hotwords.set('密等性=幂等性, Redis')
        self.app.save_settings_dialog()
        self.assertEqual(json.loads(self.prefs.read_text('utf-8'))['note'],{'keep':True})
        self.prefs.write_bytes(b'{broken')
        self.app.toggle_settings()  # Still opens so the user can repair the list.
        self.app.hotwords.set('密等性=幂等性, RAG')
        self.app.save_settings_dialog()
        self.assertFalse(self.app.settings_visible)
        self.assertEqual(self.start().call_args.args[3],'密等性=幂等性, RAG')


if __name__=='__main__':unittest.main()
