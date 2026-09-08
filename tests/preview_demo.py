"""Manual UI check with simulated text; does not record or register hotkeys."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import App, make_segment
import time


with tempfile.TemporaryDirectory() as folder, patch("app.GlobalHotkey"):
    root = tk.Tk()
    app = App(root)
    root.title("闻录 · 临时字幕界面验证")
    app.hotkey_hint.configure(text="界面验证")
    app.status.set("模拟文字，不录音 · 按回车确认句子")
    app.start_button.configure(state="disabled")
    app.session = Path(folder) / "test.jsonl"
    app.events.put(("preview", "现在正在测试实时字母，讲话过程中会先显示临时结果。"))

    def confirm(_event):
        app.events.put(("segment", make_segment(0, 5,
            "现在正在测试实时字幕，讲话过程中会先显示临时结果，句子结束后自动修正。", time.time())))
        app.events.put(("done", None))

    root.bind("<Return>", confirm)
    root.after(90000, app.close)
    root.mainloop()
