import gc
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from app import App
from qa_records import load_qa, save_qa_copy


class ManualPracticeTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.folder=tempfile.TemporaryDirectory()
        self.path=Path(self.folder.name)/'practice.qa.jsonl'
        for name,options in (
            ('app.GlobalHotkey',{}),('app.App.start_tray',{}),('app.ClipboardWatcher',{}),
            ('qa_connection.QAConnection.check',{}),('app.save_desktop',{}),
            ('app.preferences',{'return_value':{}}),
            ('app.load_settings',{'return_value':{'provider':'compatible','profiles':{}}})):
            item=patch(name,**options);item.start();self.addCleanup(item.stop)
        self.root=tk.Tk();self.root.withdraw();self.app=App(self.root)

    def tearDown(self):
        self.app.qa.set_enabled(False)
        self.app.qa.context_provider=self.app.qa.voice_context_provider=None
        self.app.qa_connection.close();self.app.gpu_check.close();self.app.hotkey.close()
        for timer in self.root.tk.call('after','info'):self.root.after_cancel(timer)
        self.root.destroy();self.folder.cleanup();self.root=self.app=None;gc.collect()

    def answer(self,question='为什么？',answer='同一订单只处理一次。'):
        generation=self.app.qa.generation
        self.app.handle_qa((generation,'thinking',question))
        self.app.handle_qa((generation,'answer',(question,answer)))

    def save(self,path=None):
        with patch('app.filedialog.asksaveasfilename',return_value=str(path or self.path)):
            self.app.save_qa_copy()

    def test_manual_answers_save_copy_without_audio_or_changing_active_destination(self):
        self.answer();self.answer()
        self.assertIsNone(self.app.session)
        ids=dict(self.app.qa_history_ids)
        self.save()
        records=load_qa(self.path)
        self.assertEqual([r.key for r in records],list(ids.values()))
        self.assertEqual([r.answer for r in records],['同一订单只处理一次。']*2)
        self.assertTrue(all(r.time for r in records))
        self.assertEqual([r.time for r in records],[self.app.qa_times[r.key] for r in records])
        self.assertIsNone(self.app.session)
        self.assertIsNone(self.app.qa_journal_path)
        self.assertFalse(self.app.qa_origins)
        self.assertEqual(self.app.qa_history_ids,ids)

    def test_open_standalone_copy_and_continue_then_edit_first_duplicate(self):
        self.answer();self.answer();self.save()
        old=self.path.read_bytes()
        with patch.object(self.app.qa,'ask') as ask:
            self.app.restore_session(self.path)
            ask.assert_not_called()
        self.assertEqual(self.path.read_bytes(),old)
        self.assertEqual(self.app.rows,[])
        self.assertEqual(self.app.qa_journal_path,self.path)
        self.assertIsNone(self.app.session)
        self.answer('再具体一点。','使用订单号的唯一约束。')
        self.assertEqual(len(load_qa(self.path)),3)
        with patch.object(self.app.qa,'ask',side_effect=lambda *_:self.app.qa.reset()):
            self.app.replace_question(0,'修改第一条',[])
        generation=self.app.qa.generation
        self.app.handle_qa((generation,'answer',('修改第一条','第一条新回答')))
        self.app.handle_qa((generation,'done',None))
        self.assertEqual([r.question for r in load_qa(self.path)],['修改第一条','为什么？','再具体一点。'])
        self.app.restore_session(self.path)
        self.assertEqual(self.app.qa_history[0],('修改第一条','第一条新回答'))

    def test_copy_excludes_streaming_and_revised_incomplete_answers(self):
        self.answer('完成题','完整答案')
        self.app.qa_history.append(('正在修改',''))
        self.app.qa_question,self.app.qa_answer='生成中问题','尚未完成的部分文字'
        self.save()
        self.assertEqual([(r.question,r.answer) for r in load_qa(self.path)],[('完成题','完整答案')])

    def test_empty_or_cancelled_export_does_not_create_files_or_send_requests(self):
        with patch('app.filedialog.asksaveasfilename') as dialog:
            self.app.save_qa_copy();dialog.assert_not_called()
        self.answer()
        with patch('app.filedialog.asksaveasfilename',return_value=''):
            self.app.save_qa_copy()
        self.assertFalse(self.path.exists())
        self.assertFalse(self.app.qa.active)

    def test_export_cannot_replace_current_practice_or_recording_sources(self):
        self.answer();self.save();self.app.restore_session(self.path)
        original=self.path.read_bytes()
        self.save()
        self.assertEqual(self.path.read_bytes(),original)
        self.assertIn('另一个文件名',self.app.qa_status.get())
        recording=Path(self.folder.name)/'audio.jsonl'
        recording.write_text(json.dumps(dict(start=0,end=1,text='原转录')),encoding='utf-8')
        self.app.session=recording
        raw=recording.read_bytes()
        self.save(recording)
        self.assertEqual(recording.read_bytes(),raw)

    def test_failed_atomic_export_keeps_existing_copy_and_memory(self):
        self.answer();self.save();old=self.path.read_bytes()
        self.answer('新增问题','新增回答')
        with patch('qa_records.Path.replace',side_effect=OSError('disk full')):
            self.save()
        self.assertEqual(self.path.read_bytes(),old)
        self.assertEqual(len(self.app.qa_history),2)
        self.assertIn('未保存',self.app.qa_status.get())
        self.assertFalse(list(self.path.parent.glob('*.tmp')))

    def test_bad_standalone_file_does_not_clear_current_practice(self):
        self.answer('当前问题','当前回答')
        self.path.write_bytes(b'{bad')
        with self.assertRaises(ValueError):self.app.restore_session(self.path)
        self.assertEqual(self.app.qa_history,[('当前问题','当前回答')])
        self.assertIsNone(self.app.qa_journal_path)

    def test_missing_standalone_file_does_not_clear_current_practice(self):
        self.answer('当前问题','当前回答')
        with self.assertRaises(FileNotFoundError):self.app.restore_session(self.path)
        self.assertEqual(self.app.qa_history,[('当前问题','当前回答')])

    def test_externally_deleted_open_file_is_not_silently_recreated(self):
        self.answer();self.save();self.app.restore_session(self.path)
        self.path.unlink()
        self.answer('新问题','可另存的新答案')
        self.assertFalse(self.path.exists())
        self.assertIn('已不存在',self.app.qa_status.get())
        self.assertEqual(len(self.app.qa_history),2)

    def test_clear_view_keeps_open_practice_for_new_answers(self):
        self.answer();self.save();self.app.restore_session(self.path)
        self.app.clear(preserve_recording=True)
        self.answer('清空后新题','仍追加到同一练习')
        self.assertEqual(len(load_qa(self.path)),2)
        self.assertEqual(self.app.qa_journal_path,self.path)

    def test_opening_qa_file_does_not_implicitly_load_a_companion_transcript(self):
        self.answer();self.save()
        companion=self.path.with_name('practice.jsonl')
        companion.write_text(json.dumps(dict(start=0,end=1,text='同名转录')),encoding='utf-8')
        self.app.restore_session(self.path)
        self.assertFalse(self.app.rows)
        self.assertIsNone(self.app.session)

    def test_starting_audio_switches_only_new_answers_to_new_recording(self):
        self.answer();self.save();self.app.restore_session(self.path)
        old=self.path.read_bytes()
        self.app.capture_mode.set('仅系统声音')
        self.app.devices=[{'index':0}];self.app.device.configure(values=['Test']);self.app.device.current(0)
        with patch('app.RECORDINGS',self.path.parent),patch('app.save_preferences'),patch.object(self.app.engine,'start'):
            self.app.start()
        self.assertIsNone(self.app.qa_journal_path)
        self.answer('开始录音后新题','保存到新录音')
        self.assertEqual(self.path.read_bytes(),old)
        self.assertEqual(load_qa(self.app.session.with_suffix('.qa.jsonl'))[0].question,'开始录音后新题')

    def test_failed_audio_start_keeps_the_open_manual_practice_destination(self):
        self.answer();self.save();self.app.restore_session(self.path)
        self.app.capture_mode.set('仅系统声音')
        self.app.devices=[{'index':0}];self.app.device.configure(values=['Test']);self.app.device.current(0)
        with patch('app.RECORDINGS',self.path.parent),patch('app.save_preferences'), \
                patch('app.Path.touch',side_effect=OSError('disk full')),patch('app.messagebox.showerror'):
            self.app.start()
        self.assertIsNone(self.app.session)
        self.assertEqual(self.app.qa_journal_path,self.path)
        self.answer('录音未开始时新题','保留当前问答文件')
        self.assertEqual(len(load_qa(self.path)),2)


class CopyFormatTests(unittest.TestCase):
    def test_invalid_copy_does_not_overwrite_an_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'copy.qa.jsonl';path.write_bytes(b'previous')
            for records in ([],[{'question':'','answer':'a'}],[{'question':'q','answer':None}]):
                with self.assertRaises(ValueError):save_qa_copy(path,records)
                self.assertEqual(path.read_bytes(),b'previous')
