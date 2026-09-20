import tkinter as tk
import unittest
from unittest.mock import patch

from app import App


class AnswerSelectionTests(unittest.TestCase):
    def setUp(self):
        for target in ('app.GlobalHotkey', 'app.App.start_tray',
                       'qa_connection.QAConnection.check', 'app.save_desktop'):
            patched = patch(target)
            patched.start()
            self.addCleanup(patched.stop)
        self.root = tk.Tk()
        self.app = App(self.root)
        self.root.geometry('560x370')
        self.root.update()

    def tearDown(self):
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.app.close()

    def test_empty_panel_does_not_select_implicit_newline(self):
        text = self.app.qa_text
        self.assertEqual(text.get('1.0', 'end-1c'), '')
        text.event_generate('<<SelectAll>>')
        self.root.update()
        self.assertFalse(text.tag_ranges('sel'))
        text.event_generate('<ButtonPress-1>', x=20, y=15)
        # Exercise Tk's real triple-click line selection, which includes the
        # implicit newline; <<Selection>> must strip that selection afterward.
        script = text.bind_class('Text', '<Triple-Button-1>')
        text.tk.eval(script.replace('%W', str(text)).replace('%x', '20').replace('%y', '15'))
        self.root.update()
        self.assertFalse(text.tag_ranges('sel'))

    def test_answer_selection_remains_copyable_and_clear_leaves_no_highlight(self):
        self.app.qa_question, self.app.qa_answer = '问题', '正常回答\n第二行'
        self.app.render_qa()
        self.root.update()
        text = self.app.qa_text
        text.event_generate('<<SelectAll>>')
        self.root.update()
        self.assertEqual(text.get('sel.first', 'sel.last'), '问题\n正常回答\n第二行')
        self.assertEqual(text.index('sel.last'), text.index('end-1c'))
        text.tag_remove('sel', '1.0', 'end')
        text.tag_add('sel', '2.0', '2.4')
        self.root.update()
        self.assertEqual(text.get('sel.first', 'sel.last'), '正常回答')
        self.app.clear()
        self.root.update()
        self.assertFalse(text.tag_ranges('sel'))
        self.assertEqual(text.get('1.0', 'end-1c'), '')


if __name__ == '__main__':
    unittest.main()
