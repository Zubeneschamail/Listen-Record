"""Visual QA fixture: no recording, hotkeys, network calls, or saved user data."""
import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import App

with patch("app.GlobalHotkey"):
    root = tk.Tk()
    app = App(root)
root.title("闻录 · 问答界面验证")
root.geometry("560x380+650+150")
app.hotkey_hint.configure(text="界面验证")
app.start_button.configure(state="disabled")
app.status.set("界面验证 · 不录音、不联网 · 90 秒后自动关闭")
app.qa_enabled.set(True)
app.toggle_qa()
app.qa.runner = lambda prompt, cancel: "太阳光进入大气后，波长较短的蓝光比红光更容易被空气分子散射到各个方向，因此我们通常看到蓝色的天空。"
from tempfile import TemporaryDirectory
with TemporaryDirectory() as directory:
    app.session = Path(directory) / "demo.jsonl"
    for i, text in enumerate(["下面讨论一个光学问题。", "为什么天空通常是蓝色的？", "日落时又为什么呈现红色？"]):
        app.events.put(("segment", {"start": i, "end": i+1, "text": text}))
    root.after(90000, app.close)
    root.mainloop()
