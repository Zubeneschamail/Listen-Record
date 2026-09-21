import json
import tkinter as tk
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from app import App


class AutoQATests(unittest.TestCase):
    def test_restart_preserves_conversation_and_f12_toggles_auto(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), \
                patch('qa_connection.QAConnection.check'), patch('app.ClipboardWatcher'):
            root = tk.Tk()
            app = App(root)
            try:
                root.update()
                root.focus_force()
                root.event_generate('<F12>')
                root.update()
                self.assertTrue(app.auto_qa.get())
                self.assertTrue(app.qa.enabled)
                root.event_generate('<F12>')
                root.update()
                self.assertFalse(app.auto_qa.get())
                with tempfile.TemporaryDirectory() as folder, \
                        patch('app.RECORDINGS', Path(folder)), patch('app.save_preferences'), \
                        patch.object(app.engine, 'start'), patch.object(app.engine, 'stop'):
                    app.capture_mode.set('仅系统声音')
                    app.devices = [{'index': 0}]
                    app.device.configure(values=['Test'])
                    app.device.current(0)
                    app.start()
                    first = app.session
                    app.events.put(('segment', {'start': 0, 'end': 1, 'text': '第一段文字'}))
                    app.poll()
                    app.qa_question, app.qa_answer = '旧问题', '旧答案'
                    app.qa_display_history = [('更早的问题', '更早的答案')]
                    app.render_qa()
                    before = app.qa_text.get('1.0', 'end-1c')
                    app.stop()
                    app.events.put(('done', None))
                    app.poll()
                    app.start()
                    self.assertNotEqual(app.session, first)
                    self.assertEqual(len(app.rows), 1)
                    self.assertEqual(app.qa_text.get('1.0', 'end-1c'), before)
                    app.events.put(('segment', {'start': 0, 'end': 1, 'text': '第二段文字'}))
                    app.poll()
                    self.assertEqual(len(app.rows), 2)
                    self.assertEqual(len(app.session_rows), 1)
                    self.assertNotIn('第一段文字', app.session.with_suffix('.txt').read_text(encoding='utf-8-sig'))
                    root.event_generate('<Control-BackSpace>')
                    root.update()
                    self.assertFalse(app.rows)
                    self.assertFalse(app.qa_display_history)
                    self.assertEqual(app.qa_text.get('1.0', 'end-1c'), '')
            finally:
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()

    def test_auto_context_new_questions_and_disable_cancellation(self):
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), \
                patch('qa_connection.QAConnection.check'), patch('app.ClipboardWatcher'):
            root = tk.Tk()
            app = App(root)
            try:
                app.qa_enabled.set(True)
                app.toggle_qa()
                app.qa_connection.state = 'authenticated'
                app.rows = [{'text': '我们正在讨论机器学习。'}, {'text': '旧问题是什么？'}]
                app.auto_qa.set(True)
                app.toggle_auto_qa()
                self.assertFalse(app.qa.pending)
                self.assertFalse(app.qa.active)
                prompts = []
                app.qa.stream_runner = None
                app.qa.runner = lambda prompt, cancel: prompts.append(prompt) or '示例答案'
                app.qa.feed('什么是大模型？')
                app.qa.tick()
                self.assertFalse(app.qa.active)
                app.qa.last_input -= 2
                app.qa.tick()
                # Consume the worker events without running recording poll timers.
                states = []
                while 'done' not in states:
                    kind, value = app.events.get(timeout=3)
                    if kind == 'qa':
                        states.append(value[1])
                        app.handle_qa(value)
                self.assertEqual(app.qa_question, '什么是大模型？')
                self.assertEqual(app.qa_answer, '示例答案')
                payload = json.loads(prompts[0].split('\n', 1)[1])
                self.assertIn('机器学习', '\n'.join(payload['会话资料']['全部转写']))
                self.assertEqual(app.session_context_snapshot()[1], [('什么是大模型？', '示例答案')])
                self.assertNotIn('旧问题', payload['当前问题'])
                app.qa.feed('什么是大模型？')
                app.qa.last_input -= 2
                app.qa.tick()
                self.assertFalse(app.qa.active)
                generation = app.qa.generation
                app.handle_qa((generation, 'thinking', '模型怎么训练？'))
                app.handle_qa((generation, 'partial', ('模型怎么训练？', '先准备')))
                root.update_idletasks()  # Flush the coalesced streaming render.
                visible = app.qa_text.get('1.0', 'end-1c')
                self.assertIn('什么是大模型？\n示例答案', visible)
                self.assertIn('模型怎么训练？\n先准备', visible)
                app.handle_qa((generation, 'answer', ('模型怎么训练？', '先准备训练数据。')))
                app.handle_qa((generation, 'thinking', '需要多少数据？'))
                visible = app.qa_text.get('1.0', 'end-1c')
                self.assertEqual(visible.count('什么是大模型？'), 1)
                self.assertEqual(visible.count('模型怎么训练？'), 1)
                self.assertIn('先准备训练数据。', visible)
                app.qa.feed('为什么？')
                cancellation = app.qa.cancel
                app.auto_qa.set(False)
                app.toggle_auto_qa()
                self.assertTrue(cancellation.is_set())
                self.assertFalse(app.qa.pending)
                app.auto_qa.set(True)
                app.qa_enabled.set(False)
                app.toggle_qa()
                self.assertTrue(app.auto_qa.get())
                app.clear()
                self.assertEqual(app.qa_history, [])
                self.assertEqual(app.rows, [])
                self.assertFalse(app.qa_display_history)
                self.assertEqual(app.qa_text.get('1.0', 'end-1c'), '')
            finally:
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()
