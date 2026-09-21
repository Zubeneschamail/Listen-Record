import tkinter as tk
import unittest
from unittest.mock import patch

from app import App


class QAScrollTests(unittest.TestCase):
    def setUp(self):
        for name in ('app.GlobalHotkey', 'app.App.start_tray', 'app.ClipboardWatcher',
                     'qa_connection.QAConnection.check', 'app.save_desktop'):
            target = patch(name)
            target.start()
            self.addCleanup(target.stop)
        self.root = tk.Tk()
        self.app = App(self.root)
        self.root.geometry('640x420')
        self.root.update()
        self.app.qa_question = '长回答的滚动测试'
        self.app.qa_answer = '\n\n'.join(f'第 {i} 段 **重点**。'+('原理与实现细节，方案取舍与边界。'*6) for i in range(45))
        self.app.render_qa()
        self.root.update()

    def tearDown(self):
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.app.close()

    def test_scrolled_up_view_and_selection_stay_fixed_during_stream(self):
        app, text = self.app, self.app.qa_text
        text.yview_moveto(.35)
        self.root.update()
        top = text.index('@0,0')
        y = text.dlineinfo(top)[1]
        text.tag_add('sel', top, top+'+5c')
        selected = text.get('sel.first', 'sel.last')
        with patch.object(text, 'update_idletasks', side_effect=AssertionError('Forced intermediate paint')):
            for i in range(12):
                app.qa_answer += f' 新增片段 {i}。'
                app.render_qa(streaming=True)
                self.root.update()
                self.assertEqual(text.index('@0,0'), top)
                self.assertEqual(text.dlineinfo(top)[1], y)
                self.assertEqual(text.get('sel.first', 'sel.last'), selected)

    def test_bottom_follow_resumes_only_after_user_scrolls_back(self):
        app, text = self.app, self.app.qa_text
        text.yview_moveto(1)
        self.root.update()
        for i in range(8):
            app.qa_answer += '\n\n' + f'新段落 {i} '+('流式回答继续。'*16)
            app.render_qa(streaming=True)
            self.root.update()
            self.assertGreaterEqual(text.yview()[1], .999)
        text.yview_moveto(.2)
        self.root.update()
        top = text.index('@0,0')
        app.qa_answer += '\n\n继续生成。'
        app.render_qa(streaming=True)
        self.root.update()
        self.assertEqual(text.index('@0,0'), top)
        text.yview_moveto(1)
        self.root.update()
        app.qa_answer += '\n\n继续跟随。'*12
        app.render_qa(streaming=True)
        self.root.update()
        self.assertGreaterEqual(text.yview()[1], .999)

    def test_burst_bottom_follow_and_new_turn_preserve_scroll_intent(self):
        app, text = self.app, self.app.qa_text
        text.yview_moveto(1)
        self.root.update()
        for _ in range(5):
            app.qa_answer += '连续生成的长段落需要自动换行并保持底部可见。' * 20
            app.render_qa(streaming=True)
        self.root.update()
        self.assertAlmostEqual(text.yview()[1], 1.0)

        text.yview_moveto(.35)
        self.root.update()
        anchor = text.index('@0,0')
        app.qa_display_history.append((app.qa_question, app.qa_answer))
        app.qa_question, app.qa_answer = '新问题', '新回复\n' * 80
        app.render_qa()
        self.root.update()
        self.assertEqual(text.index('@0,0'), anchor)

        text.yview_moveto(1)
        self.root.update()
        text.yview_scroll(-1, 'units')
        self.root.update()
        anchor = text.index('@0,0')
        app.qa_answer += '末尾新内容\n' * 20
        app.render_qa(streaming=True)
        self.root.update()
        self.assertEqual(text.index('@0,0'), anchor)

    def test_burst_partials_render_once_and_final_cancels_pending_render(self):
        app = self.app
        generation = app.qa.generation
        with patch.object(app, 'render_qa', wraps=app.render_qa) as render:
            for i in range(5):
                app.handle_qa((generation, 'partial', (app.qa_question, f'片段 {i}')))
            self.assertEqual(render.call_count, 0)
            self.root.update()
            self.assertEqual(render.call_count, 1)
            app.handle_qa((generation, 'partial', (app.qa_question, '待刷新')))
            app.handle_qa((generation, 'answer', (app.qa_question, '最终回答')))
            self.assertEqual(render.call_count, 2)
            self.root.update()
            self.assertEqual(render.call_count, 2)
        self.assertTrue(app.qa_text.get('1.0', 'end-1c').endswith('最终回答'))
