import tkinter as tk
import unittest
from types import SimpleNamespace

from app import App


class AnswerScrollTests(unittest.TestCase):
    def test_history_position_and_bottom_follow(self):
        root = tk.Tk()
        root.geometry('360x240')
        text = tk.Text(root, wrap='word')
        text.pack(fill='both', expand=True)
        app = SimpleNamespace(qa_text=text, qa_answer='', qa_question='问题',
                              qa_display_history=[], answer_copy_button=tk.Button(root),
                              qa_placeholder=tk.Label(text))
        try:
            root.update()
            app.qa_answer = '\n'.join(f'历史内容 {i}' for i in range(200))
            App.render_qa(app)
            root.update()
            self.assertAlmostEqual(text.yview()[1], 1.0)

            text.yview('40.0')
            root.update()
            anchor = text.index('@0,0')
            app.qa_answer += '\n' + '\n'.join(f'新增 {i}' for i in range(50))
            App.render_qa(app, streaming=True)
            root.update()
            self.assertEqual(text.index('@0,0'), anchor)

            # Auto adds a new turn by rebuilding the transcript.
            app.qa_display_history.append((app.qa_question, app.qa_answer))
            app.qa_question, app.qa_answer = '新问题', '新回复\n' * 80
            App.render_qa(app)
            root.update()
            self.assertEqual(text.index('@0,0'), anchor)

            text.see('end')
            root.update()
            app.qa_answer += '继续回答\n' * 40
            App.render_qa(app, streaming=True)
            root.update()
            self.assertAlmostEqual(text.yview()[1], 1.0)

            text.yview_scroll(-1, 'units')
            root.update()
            anchor = text.index('@0,0')
            app.qa_answer += '末尾新内容\n' * 20
            App.render_qa(app, streaming=True)
            root.update()
            self.assertEqual(text.index('@0,0'), anchor)
        finally:
            root.destroy()
