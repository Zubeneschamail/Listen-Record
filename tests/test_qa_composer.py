import tkinter as tk
import unittest
from unittest.mock import patch

from app import App


class ComposerTests(unittest.TestCase):
    def setUp(self):
        for target in ('app.GlobalHotkey', 'app.App.start_tray', 'app.ClipboardWatcher',
                       'qa_connection.QAConnection.check', 'app.save_desktop'):
            patched = patch(target)
            patched.start()
            self.addCleanup(patched.stop)
        self.root = tk.Tk()
        self.app = App(self.root)
        self.app.qa_connection.state = 'verified'
        self.root.geometry('620x420')
        self.root.update()

    def tearDown(self):
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.app.close()

    def finish(self):
        while True:
            kind, value = self.app.events.get(timeout=3)
            if kind == 'qa':
                self.app.handle_qa(value)
                if value[1] == 'done':
                    return

    def test_collapse_preserves_draft_and_only_exposes_add_and_send(self):
        composer = self.app.composer
        self.assertTrue(composer.expanded)
        composer.input.insert('1.0', '第一行\n第二行')
        self.root.update()
        height = self.app.qa_text.winfo_height()
        composer.collapse()
        self.root.update()
        self.assertFalse(composer.expanded)
        self.assertFalse(composer.input.winfo_ismapped())
        self.assertFalse(self.app.reference_path_label.winfo_ismapped())
        self.assertTrue(self.app.reference_add_button.winfo_viewable())
        self.assertTrue(self.app.ask_selected_button.winfo_viewable())
        self.assertGreater(self.app.qa_text.winfo_height(), height)
        composer.middle.event_generate('<Button-1>', x=5, y=5)
        self.root.update()
        self.assertTrue(composer.expanded)
        self.assertEqual(composer.get(), '第一行\n第二行')
        composer.clear()
        composer.collapse()
        generation = self.app.qa.generation
        self.app.ask_selected_button.invoke()
        self.assertTrue(composer.expanded)
        self.assertEqual(self.app.qa.generation, generation)

    def test_typed_questions_and_keyboard_send_continue_the_same_conversation(self):
        requests = []
        def stream(prompt, cancel, partial, history=None, **kwargs):
            requests.append(history)
            partial('回答中')
            return f'答案 {len(requests)}'
        self.app.qa.stream_runner = stream
        composer = self.app.composer
        composer.input.insert('1.0', '介绍二分查找')
        composer.collapse()
        self.app.ask_selected_button.invoke()
        self.assertEqual(composer.get(), '')
        self.assertFalse(composer.expanded)
        self.finish()
        self.assertEqual(requests[0], [])
        composer.expand()
        composer.input.insert('1.0', '给个例子')
        self.root.focus_force()
        composer.input.focus_set()
        self.root.update()
        composer.input.event_generate('<Return>')
        self.finish()
        self.assertEqual(composer.get(), '')
        self.assertEqual([item['role'] for item in requests[1]], ['user', 'assistant'])
        self.assertIn('介绍二分查找', requests[1][0]['content'])
        self.assertEqual(requests[1][1]['content'], '答案 1')
        self.assertEqual(self.app.qa_history[-1], ('给个例子', '答案 2'))

    def test_invalid_or_disconnected_submission_keeps_draft(self):
        composer = self.app.composer
        for draft, connected in [('  \n', True), ('字' * 2401, True), ('保留问题', False)]:
            composer.clear()
            composer.input.insert('1.0', draft)
            self.app.qa_connection.state = 'verified' if connected else 'unavailable'
            generation = self.app.qa.generation
            self.app.ask_selected_button.invoke()
            self.assertEqual(composer.get(), draft)
            self.assertEqual(self.app.qa.generation, generation)
            self.assertFalse(self.app.qa.active)

    def test_markdown_streaming_reformats_current_answer_and_keeps_history(self):
        app = self.app
        app.qa_display_history = [('早先问题', '**早先回答**')]
        app.qa_question, app.qa_answer = '当前问题', '**正在'
        app.render_qa()
        app.composer.input.insert('1.0', '未发送草稿')
        app.qa_answer = '**正在回答**\n\n```python\nprint(1)\n```'
        app.render_qa(streaming=True)
        visible = app.qa_text.get('1.0', 'end-1c')
        self.assertIn('早先问题\n早先回答', visible)
        self.assertIn('当前问题\n正在回答', visible)
        self.assertIn('print(1)', visible)
        self.assertNotIn('**', visible)
        self.assertNotIn('```', visible)
        self.assertTrue(app.qa_text.tag_ranges('md:codeblock'))
        self.assertEqual(app.composer.get(), '未发送草稿')
        self.assertEqual(app.qa_answer, '**正在回答**\n\n```python\nprint(1)\n```')

    def test_markdown_question_editing_and_copy_keep_original_source(self):
        from types import SimpleNamespace
        app = self.app
        app.qa_question, app.qa_answer = '**原始问题**', '# 标题\n\n`代码`'
        app.render_qa()
        self.root.update()
        start = app.qa_text.tag_ranges('question_turn:0')[0]
        x, y, _, _ = app.qa_text.bbox(start)
        app.edit_question(SimpleNamespace(x=x+1, y=y+1))
        self.assertEqual(app.question_editor.get('1.0', 'end-1c'), '**原始问题**')
        app.cancel_question_edit()
        with patch.object(self.root, 'clipboard_clear'), patch.object(self.root, 'clipboard_append') as copy:
            app.copy_answer()
            copy.assert_called_with('# 标题\n\n`代码`')
            app.qa_display_history = [('**历史问题**', '```python\nx = 1\n```')]
            app.render_qa()
            app.copy_answer()
            self.assertIn('**历史问题**', copy.call_args.args[0])
            self.assertIn('```python', copy.call_args.args[0])

    def test_pasted_image_is_a_removable_draft_and_survives_layout_changes(self):
        from clipboard_watch import image_item
        from PIL import Image
        composer = self.app.composer
        item = image_item(Image.new('RGB', (320, 180), '#007ACC'))
        composer.input.insert('1.0', '解释图片')
        generation = self.app.qa.generation
        with patch('clipboard_watch.WindowsClipboard.read_image', return_value=item):
            composer.input.event_generate('<<Paste>>')
        self.root.update()
        self.assertIs(composer.image_item, item)
        self.assertTrue(composer.attachment.winfo_ismapped())
        self.assertEqual(composer.get(), '解释图片')
        self.assertEqual(self.app.qa.generation, generation)
        composer.collapse()
        self.root.update()
        self.assertFalse(composer.attachment.winfo_ismapped())
        composer.expand()
        self.app.dark_mode.set(True)
        self.app.apply_theme()
        self.root.geometry('420x280')
        self.root.update()
        self.assertIs(composer.image_item, item)
        self.assertTrue(composer.attachment.winfo_ismapped())
        self.assertGreater(composer.input.winfo_width(), 0)
        composer.remove_image_button.invoke()
        self.root.update()
        self.assertIsNone(composer.image_item)
        self.assertFalse(composer.attachment.winfo_ismapped())
        self.assertEqual(composer.get(), '解释图片')

    def test_image_and_text_send_together_and_preview_survives_streaming(self):
        from clipboard_watch import image_item
        from PIL import Image
        requests = []
        def stream(prompt, cancel, partial, **kwargs):
            requests.append((prompt, kwargs))
            if 'image_context' in kwargs:
                kwargs['image_context']('图片里有三个蓝色方块')
            partial('分析中')
            return '分析结果'
        self.app.qa.stream_runner = stream
        self.app.qa_settings['provider'] = 'deepseek'
        self.app.qa_settings['profiles']['deepseek']['model'] = 'deepseek-v4-pro'
        composer = self.app.composer
        item = image_item(Image.new('RGB', (80, 60), 'blue'))
        composer.set_image(item)
        composer.input.insert('1.0', '图中有多少个方块？')
        self.app.ask_selected_button.invoke()
        self.finish()
        self.assertEqual(requests[0][1]['image'], item.image)
        self.assertIn('图中有多少个方块？', requests[0][0])
        self.assertIsNone(composer.image_item)
        self.assertEqual(composer.get(), '')
        self.assertEqual(len(self.app.qa_text.image_names()), 1)
        self.assertIn('图中有多少个方块？', self.app.qa_text.get('1.0', 'end'))
        composer.input.insert('1.0', '它们是什么颜色？')
        self.app.send_question()
        self.finish()
        self.assertIn('图片里有三个蓝色方块', str(requests[1][1]['history']))
        self.assertEqual(len(self.app.qa_text.image_names()), 1)
        self.assertNotIn('image', requests[1][1])

    def test_image_only_submission_and_failed_validation_keep_attachment(self):
        from clipboard_watch import image_item
        from PIL import Image
        composer = self.app.composer
        item = image_item(Image.new('RGB', (80, 60), 'blue'))
        composer.set_image(item)
        self.app.qa_connection.state = 'unavailable'
        self.app.send_question()
        self.assertIs(composer.image_item, item)
        self.app.qa_connection.state = 'verified'
        with patch('app.image_input_error', return_value='模型不支持图片'):
            self.app.send_question()
        self.assertIs(composer.image_item, item)
        with patch.object(self.app.qa, 'ask') as ask, patch('app.image_input_error', return_value=''):
            self.app.send_question()
        self.assertEqual(ask.call_args.args[0], item.text)
        self.assertEqual(ask.call_args.kwargs['image'], item.image)
        self.assertIsNone(composer.image_item)

    def test_paste_failure_preserves_draft_and_text_paste_uses_native_handler(self):
        composer = self.app.composer
        composer.input.insert('1.0', '原有文字')
        with patch('clipboard_watch.WindowsClipboard.read_image', side_effect=OSError('剪贴板被占用')):
            self.assertEqual(self.app.paste_question_image(), 'break')
        self.assertIn('剪贴板被占用', self.app.qa_status.get())
        self.assertEqual(composer.get(), '原有文字')
        with patch('clipboard_watch.WindowsClipboard.read_image', return_value=None):
            self.assertIsNone(self.app.paste_question_image())

    def test_ctrl_enter_replaces_selection_with_newline_without_sending(self):
        composer = self.app.composer
        self.root.focus_force()
        composer.input.focus_set()
        self.root.update()
        composer.input.insert('1.0', '第一行替换第二行')
        composer.input.edit_separator()
        composer.input.tag_add('sel', '1.3', '1.5')
        composer.input.mark_set('insert', '1.5')
        generation = self.app.qa.generation
        composer.input.event_generate('<Control-Return>')
        self.root.update()
        self.assertEqual(composer.get(), '第一行\n第二行')
        self.assertEqual(self.app.qa.generation, generation)
        composer.input.edit_undo()
        self.assertEqual(composer.get(), '第一行替换第二行')

    def test_enter_does_not_send_while_ime_is_composing(self):
        composer = self.app.composer
        composer.input.insert('1.0', '未发送草稿')
        self.root.focus_force()
        composer.input.focus_set()
        self.root.update()
        generation = self.app.qa.generation
        with patch('text_shortcuts.is_composing', return_value=True):
            for key in ('Return', 'Control-Return'):
                composer.input.event_generate(f'<{key}>')
                self.root.update()
        self.assertEqual(composer.get(), '未发送草稿')
        self.assertEqual(self.app.qa.generation, generation)

    def test_incoming_answer_and_theme_changes_preserve_composer_draft(self):
        composer = self.app.composer
        composer.input.insert('1.0', '尚未发送的草稿')
        composer.input.mark_set('insert', '1.3')
        composer.input.tag_add('sel', '1.0', '1.2')
        generation = self.app.qa.generation
        self.app.handle_qa((generation, 'thinking', '自动问题'))
        self.app.handle_qa((generation, 'partial', ('自动问题', '自动回答')))
        self.app.dark_mode.set(True)
        self.app.apply_theme()
        self.root.update()
        self.assertEqual(composer.get(), '尚未发送的草稿')
        self.assertEqual(composer.input.index('insert'), '1.3')
        self.assertEqual(tuple(map(str, composer.input.tag_ranges('sel'))), ('1.0', '1.2'))
        self.assertEqual(composer.input.cget('bg'), self.app.theme_color('white'))
        self.assertEqual(self.app.qa_rows.cget('bg'), self.app.theme_color('#E3E9F0'))

    def test_ctrl_backspace_edits_draft_instead_of_clearing_conversation(self):
        composer = self.app.composer
        self.app.qa_history = [('已有问题', '已有答案')]
        composer.input.insert('1.0', 'hello world')
        composer.input.mark_set('insert', 'end-1c')
        self.root.focus_force()
        composer.input.focus_set()
        self.root.update()
        composer.input.event_generate('<Control-BackSpace>')
        self.root.update()
        self.assertEqual(self.app.qa_history, [('已有问题', '已有答案')])
        self.assertIn('hello', composer.get())
        self.assertNotIn('world', composer.get())

    def test_vertical_drag_resizes_input_and_remembers_height_after_collapse(self):
        composer, rows, handle = self.app.composer, self.app.qa_rows, self.app.composer_splitter
        composer.input.insert('1.0', '拖动时保留输入内容')
        composer.input.mark_set('insert', '1.3')
        composer.input.tag_add('sel', '1.0', '1.2')
        self.root.update()
        original = rows.sash_coord(0)[1]
        text_height = composer.input.winfo_height()
        geometry = self.root.geometry()
        self.assertEqual(handle.winfo_height(), 13)
        self.assertEqual(handle.winfo_width(), rows.winfo_width())
        handle.event_generate('<ButtonPress-1>', x=40, y=6, rootx=500, rooty=300)
        handle.event_generate('<B1-Motion>', x=40, y=-44, rootx=500, rooty=250)
        self.root.update()
        self.assertEqual(rows.sash_coord(0)[1], original)
        handle.event_generate('<ButtonRelease-1>', x=40, y=-44, rootx=500, rooty=250)
        self.root.update()
        self.assertEqual(rows.sash_coord(0)[1], original-50)
        self.assertEqual(composer.input.winfo_height(), text_height+50)
        self.assertEqual(self.root.geometry(), geometry)
        expanded_height = composer.winfo_height()
        composer.expand()
        self.root.update()
        self.assertEqual(composer.winfo_height(), expanded_height)
        composer.collapse()
        self.root.update()
        self.assertEqual(composer.winfo_height(), 40)
        self.assertFalse(handle.winfo_ismapped())
        composer.expand()
        self.root.update()
        self.assertEqual(composer.winfo_height(), expanded_height)
        self.assertTrue(handle.winfo_ismapped())
        self.root.geometry('620x500')
        self.root.update()
        self.assertEqual(composer.winfo_height(), expanded_height)
        self.assertEqual(composer.get(), '拖动时保留输入内容')
        self.assertEqual(composer.input.index('insert'), '1.3')
        self.assertEqual(tuple(map(str, composer.input.tag_ranges('sel'))), ('1.0', '1.2'))

    def test_vertical_drag_respects_minimum_heights_and_can_be_cancelled(self):
        handle, rows = self.app.composer_splitter, self.app.qa_rows
        original = rows.sash_coord(0)[1]
        handle.event_generate('<ButtonPress-1>', x=40, y=6, rootx=500, rooty=300)
        handle.event_generate('<B1-Motion>', x=40, y=-1000, rootx=500, rooty=-1000)
        self.assertEqual(handle.target, 60)
        handle.cancel()
        self.root.update()
        self.assertEqual(rows.sash_coord(0)[1], original)
        self.assertIsNone(self.root.grab_current())
        handle.event_generate('<ButtonPress-1>', x=40, y=6, rootx=500, rooty=300)
        handle.event_generate('<ButtonRelease-1>', x=40, y=1000, rootx=500, rooty=1000)
        self.root.update()
        self.assertEqual(self.app.composer.winfo_height(), 80)
        self.assertGreaterEqual(self.app.qa_text.winfo_height(), 60)

    def test_dividers_meet_after_resize_and_no_collapse_button_remains(self):
        handle = self.app.splitter_handle
        self.assertFalse(hasattr(self.app.composer, 'collapse_button'))
        for height in (100, 180):
            self.app.qa_rows.paneconfigure(self.app.composer, height=height)
            self.root.update()
            expected_y = (self.app.qa_rows.winfo_rooty() - self.app.columns.winfo_rooty()
                          + self.app.qa_rows.sash_coord(0)[1])
            self.assertEqual(handle.coords(handle.junction), [6, expected_y, 13, expected_y])
            self.assertEqual(handle.itemcget(handle.junction, 'state'), 'normal')
        self.app.dark_mode.set(True)
        self.app.apply_theme()
        self.assertEqual(handle.itemcget(handle.junction, 'fill'), handle.itemcget(handle.line, 'fill'))


if __name__ == '__main__':
    unittest.main()
