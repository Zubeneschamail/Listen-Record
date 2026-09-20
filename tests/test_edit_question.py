import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from app import App


class EditQuestionTests(unittest.TestCase):
    def test_switch_questions_preserves_drafts_and_sends_selected_turn(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'):
            root = tk.Tk()
            app = App(root)
        try:
            app.open_qa()
            app.qa_connection.state = 'verified'
            app.qa_display_history = [('第一问题', '第一回复')]
            app.qa_question, app.qa_answer = '第二问题', '第二回复'
            app.render_qa()

            def edit(index):
                start = app.qa_text.tag_ranges(f'question_turn:{index}')[0]
                app.qa_text.see(start)
                root.update()
                x, y, _, _ = app.qa_text.bbox(start)
                app.edit_question(SimpleNamespace(x=x+2, y=y+2))
                root.update()
                return app.question_editor

            first = edit(0)
            first.insert('end', '草稿一')
            second = edit(1)
            self.assertFalse(first.winfo_exists())
            self.assertEqual(second.get('1.0', 'end-1c'), '第二问题')
            second.insert('end', '草稿二')
            first = edit(0)
            self.assertFalse(second.winfo_exists())
            self.assertEqual(first.get('1.0', 'end-1c'), '第一问题草稿一')
            second = edit(1)
            self.assertEqual(second.get('1.0', 'end-1c'), '第二问题草稿二')
            with patch.object(app.qa, 'ask') as ask:
                app.ask_selected()
                ask.assert_called_once_with('第二问题草稿二', [])
            self.assertEqual(app.qa_display_history, [('第一问题', '第一回复')])
            self.assertEqual(app.qa_question, '第二问题草稿二')
            self.assertNotIn((1, '第二问题'), app.question_drafts)
        finally:
            app.qa.set_enabled(False)
            app.qa_connection.close()
            app.hotkey.close()
            for timer in root.tk.call('after', 'info'):
                root.after_cancel(timer)
            root.destroy()

    def test_auto_history_replacement_stays_in_original_position(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'):
            root = tk.Tk()
            app = App(root)
        try:
            app.open_qa()
            app.auto_qa.set(True)
            app.qa_display_history = [('旧问题', '旧回答')]
            app.qa_question, app.qa_answer = '最新问题', '最新回答'
            app.qa_history = [('旧问题', '旧回答')]
            with patch.object(app.qa, 'ask', side_effect=lambda *_: app.qa.reset()):
                app.replace_question(0, '修改问题', [])
            generation = app.qa.generation
            app.handle_qa((generation, 'thinking', '修改问题'))
            app.handle_qa((generation, 'partial', ('修改问题', '新')))
            app.handle_qa((generation, 'answer', ('修改问题', '新回答')))
            self.assertEqual(app.qa_display_history, [('修改问题', '新回答')])
            self.assertEqual((app.qa_question, app.qa_answer), ('最新问题', '最新回答'))
            self.assertEqual(app.qa_history, [('修改问题', '新回答')])
            self.assertNotIn('旧回答', app.qa_text.get('1.0', 'end-1c'))
            app.handle_qa((generation, 'done', None))
            app.handle_qa((generation, 'thinking', '下一问题'))
            self.assertEqual(app.qa_question, '下一问题')
            self.assertEqual(len(app.qa_display_history), 2)
        finally:
            app.qa.set_enabled(False)
            app.qa_connection.close()
            app.hotkey.close()
            for timer in root.tk.call('after', 'info'):
                root.after_cancel(timer)
            root.destroy()

    def test_edit_and_resubmit_without_transcript_selection(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'):
            root = tk.Tk()
            app = App(root)
        try:
            app.open_qa()
            app.qa_connection.state = 'verified'
            app.qa_question, app.qa_answer = '原问题', '原回答'
            app.render_qa()
            root.update()
            x, y, _, _ = app.qa_text.bbox('1.0')
            app.edit_question(SimpleNamespace(x=x+2, y=y+2))
            editor = app.question_editor
            self.assertIsNone(root.grab_current())
            self.assertEqual(editor.get('1.0', 'end-1c'), '原问题')
            send = app.ask_selected_button
            editor.delete('1.0', 'end')
            with patch.object(app.qa, 'ask') as ask:
                send.invoke()
                ask.assert_not_called()
                self.assertTrue(editor.winfo_exists())
                app.qa_answer = "后台新增回答"
                app.render_qa(streaming=True)
                self.assertIs(app.question_editor, editor)
                self.assertIn('后台新增回答', app.qa_text.get('1.0', 'end-1c'))
                editor.insert('1.0', '修改后的专业问题')
                root.focus_force()
                editor.focus_set()
                root.update()
                editor.event_generate('<Control-Return>')
                self.assertEqual(editor.get('1.0', 'end-1c'), '修改后的专业问题\n')
                ask.assert_not_called()
                editor.event_generate('<Return>')
                ask.assert_called_once_with('修改后的专业问题', [])
            self.assertEqual(app.qa_question, '修改后的专业问题')
            self.assertIsNone(root.grab_current())
        finally:
            app.qa.set_enabled(False)
            app.qa_connection.close()
            app.hotkey.close()
            for timer in root.tk.call('after', 'info'):
                root.after_cancel(timer)
            root.destroy()

    def test_auto_updates_while_preserving_unsent_editor(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'):
            root = tk.Tk()
            app = App(root)
        try:
            app.open_qa()
            app.auto_qa.set(True)
            app.qa_question, app.qa_answer = '第一问题', '第一回复'
            app.render_qa()
            root.update()
            x, y, _, _ = app.qa_text.bbox('1.0')
            app.edit_question(SimpleNamespace(x=x+2, y=y+2))
            editor = app.question_editor
            editor.insert('end', '，未发送草稿')
            editor.mark_set('insert', '1.2')
            editor.tag_add('sel', '1.0', '1.2')
            generation = app.qa.generation
            app.handle_qa((generation, 'thinking', '第二问题'))
            app.handle_qa((generation, 'partial', ('第二问题', '第二回复正在生成')))
            root.update()
            self.assertIs(app.question_editor, editor)
            self.assertEqual(editor.get('1.0', 'end-1c'), '第一问题，未发送草稿')
            self.assertEqual(editor.index('insert'), '1.2')
            self.assertEqual(tuple(map(str, editor.tag_ranges('sel'))), ('1.0', '1.2'))
            displayed = app.qa_text.get('1.0', 'end-1c')
            self.assertIn('第一回复', displayed)
            self.assertIn('第二问题\n第二回复正在生成', displayed)
            self.assertEqual(app.edit_question_turn, 0)
            app.cancel_question_edit()
            self.assertIn('第一问题\n第一回复', app.qa_text.get('1.0', 'end-1c'))
            self.assertIn('第二问题', app.qa_text.get('1.0', 'end-1c'))
        finally:
            app.qa.set_enabled(False)
            app.qa_connection.close()
            app.hotkey.close()
            for timer in root.tk.call('after', 'info'):
                root.after_cancel(timer)
            root.destroy()
