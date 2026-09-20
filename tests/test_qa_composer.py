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
