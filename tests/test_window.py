"""Exercise the segment-to-window/save/clipboard path without recording audio."""
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from app import App


class WindowFlowTests(unittest.TestCase):
    def test_clear_conversation_preserves_recording_and_cancels_old_answer(self):
        with patch("app.GlobalHotkey"):
            root = tk.Tk()
            app = App(root)
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                app.session = Path(folder) / "session.jsonl"
                app.qa.set_enabled(True)
                app.events.put(("segment", {"start": 0, "end": 1, "text": "清空前的文字"}))
                app.poll()
                app.show_preview("临时文字")
                old = app.qa.generation
                app.qa_question, app.qa_answer = "旧问题", "旧答案"
                app.render_qa()
                app.set_busy(True)
                before = app.session.read_bytes()
                self.assertEqual(app.clear_conversation(), "break")
                self.assertEqual(app.session.read_bytes(), before)
                self.assertTrue(app.busy)
                self.assertEqual(app.rows, [])
                self.assertEqual(app.draft_text, "")
                self.assertEqual(app.qa_text.get("1.0", "end").strip(), "")
                self.assertTrue(app.qa.enabled)
                app.handle_qa((old, "answer", ("旧问题", "迟到的答案")))
                self.assertEqual(app.qa_answer, "")
                app.events.put(("segment", {"start": 1, "end": 2, "text": "清空后的文字"}))
                app.poll()
                self.assertEqual([r["text"] for r in app.rows], ["清空后的文字"])
                self.assertEqual(len(app.session.read_text(encoding="utf-8").splitlines()), 2)
                saved = app.session.with_suffix(".txt").read_text(encoding="utf-8-sig")
                self.assertIn("清空前的文字", saved)
                self.assertIn("清空后的文字", saved)
                for widget in (root, app.text, app.qa_text):
                    self.assertTrue(widget.bind("<Control-BackSpace>"))
        finally:
            app.qa.set_enabled(False)
            app.hotkey.close()
            for identifier in root.tk.call("after", "info"):
                root.after_cancel(identifier)
            root.destroy()

    def test_qa_only_uses_confirmed_text_saves_separately_and_ignores_old_answers(self):
        with patch("app.GlobalHotkey"):
            root = tk.Tk()
            app = App(root)
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                app.session = Path(folder) / "session.jsonl"
                app.qa.set_enabled(True)
                app.events.put(("preview", "为什么草稿不能发送？"))
                app.poll()
                self.assertFalse(app.qa.context)
                app.events.put(("segment", {"start": 0, "end": 1, "text": "什么是大模型？"}))
                app.poll()
                self.assertFalse(app.qa.context)
                self.assertFalse(app.qa.active)
                app.open_qa()
                old = app.qa.generation
                app.handle_qa((old, "answer", ("什么是大模型？", "这是测试答案。")))
                self.assertIn("这是测试答案", app.qa_text.get("1.0", "end"))
                saved = json.loads(app.session.with_suffix(".qa.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(saved["answer"], "这是测试答案。")
                self.assertNotIn("测试答案", app.session.read_text(encoding="utf-8"))
                app.clear()
                app.handle_qa((old, "answer", ("旧问题", "迟到的答案")))
                self.assertFalse(app.qa_history)
                self.assertNotIn("迟到", app.qa_text.get("1.0", "end"))
        finally:
            app.qa.set_enabled(False)
            app.hotkey.close()
            for identifier in root.tk.call("after", "info"):
                root.after_cancel(identifier)
            root.destroy()

    def test_click_selection_replaces_question_and_uses_cache(self):
        from unittest.mock import Mock
        with patch("app.GlobalHotkey"):
            root = tk.Tk()
            app = App(root)
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                app.session = Path(folder) / "session.jsonl"
                app.qa_enabled.set(True)
                app.toggle_qa()
                self.assertEqual(len(app.columns.panes()), 2)
                for i, text in enumerate(["这是背景。", "什么是模型？", "如何训练？"]):
                    app.events.put(("segment", {"start": i, "end": i+1, "text": text}))
                app.poll()
                self.assertFalse(app.qa.active)
                app.qa.ask = Mock()
                app.select_question("什么是模型？", [1])
                app.qa.ask.assert_called_with("什么是模型？", ["这是背景。", "什么是模型？"])
                old = app.qa.generation
                app.select_question("如何训练？", [2])
                app.handle_qa((old, "answer", ("旧问题", "旧答案")))
                self.assertEqual(app.qa_answer, "")
                app.handle_qa((app.qa.generation, "answer", ("如何训练？", "训练答案")))
                self.assertIn("训练答案", app.qa_text.get("1.0", "end"))
                app.qa.ask.reset_mock()
                app.select_question("如何训练？", [2])
                app.qa.ask.assert_not_called()
                app.text.tag_add("sel", "1.0", "3.end")
                app.ask_selected()
                self.assertEqual(app.qa.ask.call_args.args[0], "这是背景。\n什么是模型？\n如何训练？")
                self.assertNotIn("00:", app.qa.ask.call_args.args[0])
                app.qa_enabled.set(False)
                app.toggle_qa()
                self.assertEqual(len(app.columns.panes()), 1)
                self.assertEqual(len(app.rows), 3)
                before = app.session.read_bytes()
                app.restore_session(app.session)
                self.assertEqual(app.session.read_bytes(), before)
                self.assertEqual(len(app.rows), 3)
                self.assertTrue(app.text.tag_ranges("row:1"))
                self.assertFalse(app.qa.active)
        finally:
            app.qa.set_enabled(False)
            app.hotkey.close()
            for identifier in root.tk.call("after", "info"):
                root.after_cancel(identifier)
            root.destroy()

    def test_wall_clock_display_saved_metadata_and_plain_clipboard(self):
        root = tk.Tk()
        app = App(root)
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                app.session = Path(folder) / "session.jsonl"
                row = {"start": 2, "end": 4, "text": "复制时只有这一句。",
                       "captured_at": "2026-09-07T14:58:21+08:00"}
                app.events.put(("preview", "这是一句未确认的错词"))
                app.poll()
                self.assertEqual(app.rows, [])
                self.assertFalse(app.session.exists())
                self.assertEqual(app.draft_text, "这是一句未确认的错词")
                with patch.object(root, "clipboard_clear"), patch.object(root, "clipboard_append") as clipboard:
                    app.copy()
                    clipboard.assert_called_once_with("")
                app.events.put(("segment", row))
                app.poll()
                displayed = app.text.get("1.0", "end")
                self.assertIn("14:58:21", displayed)
                self.assertNotIn("00:00:02", displayed)
                self.assertEqual(app.draft_text, "")
                self.assertFalse(app.preview_frame.winfo_manager())
                self.assertNotIn("错词", displayed)
                saved = json.loads(app.session.read_text(encoding="utf-8"))
                self.assertEqual(saved["captured_at"], row["captured_at"])
                self.assertIn("[14:58:21]", app.session.with_suffix(".txt").read_text(encoding="utf-8-sig"))
                with patch.object(root, "clipboard_clear"), patch.object(root, "clipboard_append") as clipboard:
                    app.copy()
                    clipboard.assert_called_once_with("复制时只有这一句。")
                app.set_busy(True)
                self.assertEqual(app.start_button.cget("text"), "停止转写")
                self.assertEqual(str(app.model.cget("state")), "disabled")
                app.events.put(("done", None))
                app.poll()
                self.assertEqual(app.start_button.cget("text"), "开始转写")
        finally:
            app.hotkey.close()
            for identifier in root.tk.call("after", "info"):
                root.after_cancel(identifier)
            root.destroy()


if __name__ == "__main__":
    unittest.main()
