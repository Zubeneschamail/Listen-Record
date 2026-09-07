"""Windows system-audio captions. Audio never leaves the machine."""
from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pyaudiowpatch as pa
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from huggingface_hub.errors import LocalEntryNotFoundError
from opencc import OpenCC
from hotkey import GlobalHotkey
from codex_qa import CodexQA
from segmentation import PauseSegmenter, WordAssembler, decode_chunk, DraftPreview

ROOT = Path(__file__).resolve().parent
SIMPLIFIED = OpenCC("t2s")
logging.basicConfig(filename=ROOT / "app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def timestamp(seconds: float, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    seconds, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02}:{minute:02}:{sec:02}" + (f",{ms:03}" if srt else "")


def export_text(rows, subtitles=False):
    if subtitles:
        return "\n\n".join(f"{i}\n{timestamp(r['start'], True)} --> "
                            f"{timestamp(r['end'], True)}\n{r['text']}"
                            for i, r in enumerate(rows, 1)) + "\n"
    return "\n".join(f"[{row_time(r)}] {r['text']}" for r in rows) + "\n"


def row_time(row):
    if row.get("captured_at"):
        return datetime.fromisoformat(row["captured_at"]).strftime("%H:%M:%S")
    return timestamp(row["start"])  # Older saved rows only have relative timestamps.


def plain_text(rows):
    return "\n".join(row["text"] for row in rows)


def make_segment(start, end, text, captured_epoch):
    return {"start": start, "end": end, "text": text,
            "captured_at": datetime.fromtimestamp(captured_epoch).astimezone().isoformat(timespec="milliseconds")}


def mono_16k(audio, rate):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    divisor = math.gcd(int(rate), 16000)
    return resample_poly(mono, 16000 // divisor, int(rate) // divisor).astype(np.float32)


def load_model(name):
    options = dict(device="cpu", compute_type="int8", cpu_threads=6, num_workers=1,
                   download_root=str(ROOT / "models"))
    try:
        return WhisperModel(name, local_files_only=True, **options)
    except LocalEntryNotFoundError:
        return WhisperModel(name, **options)


def clean_caption(text):
    # Some short/noisy inputs cause Whisper to emit long runs of punctuation.
    text = re.sub(r"([.。…，,!?！？])\1{2,}", r"\1", text.strip())
    return SIMPLIFIED.convert(text) if re.search(r"[\w\u4e00-\u9fff]", text) else ""


class Transcriber:
    def __init__(self, events):
        self.events = events
        self.stop_event = threading.Event()
        self.thread = None
        self.model = None
        self.model_name = None

    def emit(self, kind, value):
        self.events.put((kind, value))

    def start(self, device, model, language):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, args=(device, model, language), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def run(self, device, model_name, language):
        audio_api = stream = None
        chunks = queue.Queue(maxsize=600)  # At most 60 seconds of pending audio.
        overflow = threading.Event()
        origin = 0.0
        try:
            self.emit("status", "正在加载模型，首次使用需下载…")
            if self.model_name != model_name:
                self.model = None
                self.model_name = None
                self.model = load_model(model_name)
                self.model_name = model_name
            if self.stop_event.is_set():
                return
            audio_api = pa.PyAudio()
            info = audio_api.get_device_info_by_index(device)
            if not info.get("isLoopbackDevice"):
                raise RuntimeError("所选设备不是系统回环设备，请刷新后重新选择。")
            rate, channels = int(info["defaultSampleRate"]), int(info["maxInputChannels"])

            def callback(data, count, timing, flags):
                if self.stop_event.is_set():
                    return (None, pa.paComplete)
                if flags:
                    overflow.set()
                    self.stop_event.set()
                    return (None, pa.paAbort)
                samples = np.frombuffer(data, dtype=np.float32).reshape(-1, channels).copy()
                offset = max(0, time.monotonic() - origin - count / rate)
                try:
                    chunks.put_nowait((offset, samples, time.time() - count / rate))
                except queue.Full:
                    overflow.set()
                    self.stop_event.set()
                    return (None, pa.paAbort)
                self.emit("level", min(100, float(np.sqrt(np.mean(samples ** 2))) * 500))
                return (None, pa.paContinue)

            stream = audio_api.open(format=pa.paFloat32, channels=channels, rate=rate,
                                   input=True, input_device_index=device,
                                   frames_per_buffer=rate // 10, stream_callback=callback,
                                   start=False)
            origin = time.monotonic()
            stream.start_stream()
            self.emit("status", "正在转写系统声音")
            segmenter = PauseSegmenter()
            assembler = WordAssembler()
            preview = DraftPreview()
            last_received = time.monotonic()

            def transcribe(chunks_to_decode):
                for chunk in chunks_to_decode:
                    logging.info("Audio split: %.2fs start=%.2f forced=%s overlap=%s",
                                 len(chunk.audio) / 16000, chunk.start, chunk.forced, chunk.overlap)
                    for start, end, text, epoch in decode_chunk(self.model, assembler, chunk, language):
                        text = clean_caption(text)
                        if text:
                            self.emit("segment", make_segment(start, end, text, epoch))
                    self.emit("preview", clean_caption(assembler.pending[2]) if assembler.pending else "")
                    preview.defer()

            while not self.stop_event.is_set() or not chunks.empty():
                try:
                    offset, samples, block_epoch = chunks.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - last_received > 0.7:
                        transcribe(segmenter.finish())
                        self.emit("level", 0)
                    if not self.stop_event.is_set() and not stream.is_active():
                        raise RuntimeError("音频设备已断开或停止响应，请刷新设备后重试。")
                    continue
                last_received = time.monotonic()
                transcribe(segmenter.push(mono_16k(samples, rate), offset, block_epoch))
                self.emit("backlog", chunks.qsize() / 10)
                try:
                    draft = preview.render(self.model, assembler, segmenter, language,
                                           backlog=chunks.qsize() / 10, stopping=self.stop_event.is_set())
                except Exception:
                    logging.exception("Skipping provisional subtitle; final decoding will retry")
                    preview.defer()
                    draft = None
                if draft is not None and not self.stop_event.is_set():
                    self.emit("preview", clean_caption(draft))
            transcribe(segmenter.finish())
            if overflow.is_set():
                raise RuntimeError("音频缓冲溢出或采集不连续，已停止以避免静默丢字。请选更快的模型后重试。")
        except Exception as exc:
            logging.exception("Transcription failed")
            self.emit("error", str(exc))
        finally:
            if stream:
                try:
                    stream.close()
                except Exception:
                    logging.exception("Closing stream")
            if audio_api:
                audio_api.terminate()
            self.emit("preview", "")
            self.emit("done", None)


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.engine = Transcriber(self.events)
        self.qa = CodexQA(self.events)
        self.qa_enabled = tk.BooleanVar(value=False)
        self.qa_history = []
        self.qa_cache = {}
        self.qa_selection = None
        self.qa_question = ""
        self.qa_answer = ""
        self.qa_status = tk.StringVar(value="等待声源中的问题…")
        self.rows = []
        self.session_rows = []
        self.devices = []
        self.busy = False
        self.failed = False
        self.closing = False
        self.hotkey = GlobalHotkey(self.events)
        self.session = None
        self.caption_window = None
        self.draft_text = ""
        self.last_caption = ""
        self.settings_visible = False
        self.pinned = False
        self._drag_origin = None
        root.title("声记 · 系统声音实时转文字")
        root.geometry("560x380")
        root.minsize(420, 280)
        root.overrideredirect(True)
        root.configure(bg="#dde2eb")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#fafbfe")
        style.configure("TLabel", background="#fafbfe", foreground="#778196", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=5)
        style.configure("Slim.Horizontal.TProgressbar", background="#8177ee", troughcolor="#edf0f7",
                        borderwidth=0, thickness=2)
        frame = ttk.Frame(root, padding=(14, 8, 14, 10))
        frame.pack(fill="both", expand=True, padx=1, pady=1)

        def button(parent, text, command, primary=False, width=None):
            return tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                             bg="#7266df" if primary else "#fafbfe",
                             fg="white" if primary else "#667086", activebackground="#e8e5fa",
                             activeforeground="#5548bb", disabledforeground="#a6acba",
                             font=("Microsoft YaHei UI", 9), padx=10, pady=5,
                             cursor="hand2", takefocus=True, **({"width": width} if width else {}))

        header = ttk.Frame(frame)
        header.pack(fill="x", pady=(0, 8))
        title = ttk.Label(header, text="声记", foreground="#273047", font=("Microsoft YaHei UI", 12, "bold"))
        title.pack(side="left", padx=(0, 12))
        hint = ttk.Label(header, text="Ctrl+空格", foreground="#9ba3b3")
        self.hotkey_hint = hint
        hint.pack(side="left")
        drag_space = ttk.Frame(header)
        drag_space.pack(side="left", fill="x", expand=True)
        for widget in (header, title, hint, drag_space):
            widget.bind("<ButtonPress-1>", self.drag_begin)
            widget.bind("<B1-Motion>", self.drag_move)
        button(header, "×", self.close).pack(side="right")
        button(header, "—", self.minimize).pack(side="right")
        button(header, "设置", self.toggle_settings).pack(side="right")
        tk.Checkbutton(header, text="问答", variable=self.qa_enabled, command=self.toggle_qa,
                       bg="#fafbfe", activebackground="#fafbfe", fg="#667086",
                       font=("Microsoft YaHei UI", 9), bd=0, highlightthickness=0).pack(side="right")

        self.settings_panel = ttk.Frame(frame, padding=(0, 2, 0, 10))
        self.device = ttk.Combobox(self.settings_panel, state="readonly", width=36)
        self.device.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.refresh_button = button(self.settings_panel, "刷新", self.refresh)
        self.refresh_button.grid(row=0, column=3, sticky="e", pady=(0, 8))
        self.model = ttk.Combobox(self.settings_panel, state="readonly", width=18,
                                  values=["small · 更准确", "base · 均衡", "tiny · 更快"])
        self.model.current(0)
        self.model.grid(row=1, column=0, sticky="w")
        self.language = ttk.Combobox(self.settings_panel, state="readonly", width=10,
                                     values=["中文", "自动检测", "英语"])
        self.language.current(0)
        self.language.grid(row=1, column=1, padx=8, sticky="w")
        self.settings_panel.columnconfigure(2, weight=1)
        self.hotkey_status = tk.StringVar(value="全局快捷键 Ctrl+空格 · 注册中")
        ttk.Label(self.settings_panel, textvariable=self.hotkey_status, wraplength=480).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(self.settings_panel, text="问答使用已登录的 Codex · 开启后联网发送相关转写并消耗账号额度",
                  wraplength=480).grid(row=3, column=0, columnspan=4, sticky="w", pady=(5, 0))

        self.body = ttk.Frame(frame)
        self.body.pack(fill="both", expand=True)
        controls = ttk.Frame(self.body)
        controls.pack(side="bottom", fill="x", pady=(10, 0))
        self.start_button = button(controls, "开始转写", self.toggle_recording, primary=True)
        self.start_button.pack(side="left")
        self.copy_button = button(controls, "复制全文", self.copy)
        self.copy_button.pack(side="left", padx=(8, 0))
        self.more_button = button(controls, "···", self.open_menu)
        self.more_button.pack(side="right")
        self.footer = tk.StringVar(value="自动保存")
        ttk.Label(controls, textvariable=self.footer, font=("Microsoft YaHei UI", 8)).pack(side="right", padx=4)
        grip = ttk.Label(root, text="◢", cursor="size_nw_se", font=("Segoe UI", 8))
        grip.place(relx=1, rely=1, anchor="se")
        grip.bind("<ButtonPress-1>", self.resize_begin)
        grip.bind("<B1-Motion>", self.resize_move)

        self.status = tk.StringVar(value="准备就绪")
        self.status_label = ttk.Label(self.body, textvariable=self.status, wraplength=510)
        self.status_label.pack(anchor="w", pady=(0, 8))
        self.level = tk.Canvas(self.body, height=2, bg="#edf0f7", bd=0, highlightthickness=0)
        self.level_bar = self.level.create_rectangle(0, 0, 0, 2, fill="#8177ee", outline="")
        self.level.pack(fill="x", pady=(0, 8))
        self.columns = tk.PanedWindow(self.body, orient="horizontal", bg="#e6e8f0",
                                      bd=0, sashwidth=7, sashrelief="flat", showhandle=False)
        self.columns.pack(fill="both", expand=True)
        self.left_panel = ttk.Frame(self.columns)
        self.columns.add(self.left_panel, minsize=150, stretch="always")
        self.preview_frame = tk.Frame(self.left_panel, bg="#f0eefb", padx=10, pady=8)
        self.preview_label = tk.Label(self.preview_frame, text="", bg="#f0eefb", fg="#70638f",
                                      font=("Microsoft YaHei UI", 11), wraplength=500, justify="left", anchor="w")
        self.preview_label.pack(fill="x")
        self.text = tk.Text(self.left_panel, wrap="word", font=("Microsoft YaHei UI", 11),
                            bg="white", fg="#273047", relief="flat", bd=0, highlightthickness=0,
                            padx=12, pady=10, state="disabled", spacing3=10, height=5, width=1,
                            exportselection=False)
        self.text.tag_configure("time", foreground="#a0a8b9", font=("Consolas", 8))
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("qa_selected", background="#eeeafa", foreground="#5548bb")
        self.text.bind("<ButtonPress-1>", self.question_press)
        self.text.bind("<ButtonRelease-1>", self.question_release)
        self.left_panel.bind("<Configure>", lambda e: self.preview_label.configure(wraplength=max(110, e.width-24)))
        self.qa_panel = ttk.Frame(self.columns, padding=(5, 0, 0, 0))
        qa_header = ttk.Frame(self.qa_panel)
        qa_header.pack(fill="x", pady=(0, 5))
        ttk.Label(qa_header, text="Codex", foreground="#5548bb").pack(side="left")
        button(qa_header, "复制", self.copy_answer).pack(side="right")
        self.qa_status_label = ttk.Label(self.qa_panel, textvariable=self.qa_status, wraplength=230)
        self.qa_status_label.pack(side="bottom", fill="x", pady=(5, 0))
        button(self.qa_panel, "回答选中文字", self.ask_selected).pack(side="bottom", anchor="w")
        self.qa_text = tk.Text(self.qa_panel, wrap="word", bg="white", fg="#273047", relief="flat",
                              font=("Microsoft YaHei UI", 10), padx=10, pady=10, spacing3=10,
                              state="disabled", width=1, height=1)
        self.qa_text.tag_configure("question", foreground="#8177aa")
        self.qa_text.pack(fill="both", expand=True)
        self.qa_panel.bind("<Configure>", lambda e: self.qa_status_label.configure(wraplength=max(110, e.width-10)))
        self.qa_status.set("点击左侧文字提问")
        self.menu = tk.Menu(root, tearoff=False, font=("Microsoft YaHei UI", 10))
        self.menu.add_command(label="导出 TXT / SRT", command=self.export)
        self.menu.add_command(label="悬浮字幕", command=self.open_caption)
        self.menu.add_command(label="窗口置顶", command=self.toggle_pin)
        self.menu.add_separator()
        self.menu.add_command(label="清空当前对话", accelerator="Ctrl+Backspace", command=self.clear_conversation)
        self.menu.add_command(label="查看 Codex 回答", command=self.open_qa)
        self.refresh()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Configure>", self.on_resize)
        for widget in (root, self.text, self.qa_text):
            widget.bind("<Control-BackSpace>", self.clear_conversation)
        root.after(100, self.enable_taskbar)
        root.after(100, self.poll)
        self.hotkey.start()

    def drag_begin(self, event):
        self._drag_origin = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def drag_move(self, event):
        if self._drag_origin:
            x, y = event.x_root - self._drag_origin[0], event.y_root - self._drag_origin[1]
            self.root.geometry(f"{x:+d}{y:+d}")

    def resize_begin(self, event):
        self._resize_origin = (event.x_root, event.y_root, self.root.winfo_width(), self.root.winfo_height())

    def resize_move(self, event):
        x, y, width, height = self._resize_origin
        self.root.geometry(f"{max(420, width + event.x_root - x)}x{max(280, height + event.y_root - y)}")

    def on_resize(self, event):
        if event.widget == self.root:
            self.status_label.configure(wraplength=max(360, event.width - 34))

    def enable_taskbar(self, window=None):
        # Keep a taskbar entry even though Tk's native window decorations are hidden.
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        hwnd = user32.GetParent((window or self.root).winfo_id())
        style = user32.GetWindowLongW(hwnd, -20)
        user32.SetWindowLongW(hwnd, -20, (style | 0x00040000) & ~0x00000080)

    def minimize(self):
        # Native minimize preserves the borderless HWND; toggling Tk decorations
        # recreates it and can leave the restored window hidden on Windows.
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow(user32.GetParent(self.root.winfo_id()), 6)

    def toggle_settings(self):
        self.settings_visible = not self.settings_visible
        if self.settings_visible:
            self.settings_panel.pack(fill="x", before=self.body)
        else:
            self.settings_panel.pack_forget()

    def open_menu(self):
        try:
            self.menu.tk_popup(self.more_button.winfo_rootx(), self.more_button.winfo_rooty())
        finally:
            self.menu.grab_release()

    def toggle_pin(self):
        self.pinned = not self.pinned
        self.root.attributes("-topmost", self.pinned)
        self.menu.entryconfigure(2, label="取消置顶" if self.pinned else "窗口置顶")

    def toggle_recording(self):
        if self.closing or (self.busy and self.engine.stop_event.is_set()):
            return
        self.stop() if self.busy else self.start()

    def set_level(self, value):
        self.level.coords(self.level_bar, 0, 0, self.level.winfo_width() * value / 100, 2)

    def refresh(self):
        try:
            with pa.PyAudio() as audio:
                self.devices = list(audio.get_loopback_device_info_generator())
                try:
                    default = audio.get_default_wasapi_loopback()["index"]
                except OSError:
                    default = None
            self.device["values"] = [d["name"] for d in self.devices]
            if self.devices:
                self.device.current(next((i for i, d in enumerate(self.devices) if d["index"] == default), 0))
            else:
                self.device.set("")
                self.status.set("未找到输出设备，请连接或启用扬声器/耳机后刷新。")
        except Exception as exc:
            self.status.set(f"设备枚举失败：{exc}")

    def set_busy(self, busy):
        self.busy = busy
        self.refresh_button.configure(state="disabled" if busy else "normal")
        for widget in (self.device, self.model, self.language):
            widget.configure(state="disabled" if busy else "readonly")
        self.start_button.configure(text="停止转写" if busy else "开始转写", state="normal")

    def start(self):
        if self.device.current() < 0:
            messagebox.showinfo("选择设备", "请先选择系统声音输出设备。")
            return
        # Every start is a fresh recording with its own time origin and file.
        self.clear()
        try:
            folder = ROOT / "recordings"
            folder.mkdir(exist_ok=True)
            self.session = folder / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
            self.session.touch()
        except OSError as exc:
            messagebox.showerror("无法保存记录", str(exc))
            return
        self.failed = False
        self.set_busy(True)
        self.engine.start(int(self.devices[self.device.current()]["index"]),
                          self.model.get().split()[0],
                          {"中文": "zh", "英语": "en", "自动检测": None}[self.language.get()])

    def stop(self):
        self.engine.stop()
        self.start_button.configure(text="收尾中…", state="disabled")
        self.status.set("正在处理剩余音频…")

    def clear(self, preserve_recording=False):
        self.qa.reset()
        self.qa_history.clear()
        self.qa_cache.clear()
        self.qa_selection = None
        self.qa_question = self.qa_answer = ""
        self.qa_status.set("点击左侧文字提问")
        self.render_qa()
        self.rows = []
        if not preserve_recording:
            self.session_rows = []
        self.last_caption = ""
        self.show_preview("")
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def clear_conversation(self, event=None):
        if self.closing or self.root.grab_current():
            return "break"
        self.clear(preserve_recording=True)
        self.footer.set(f"已保存 {len(self.session_rows)} 段")
        if not self.busy:
            self.status.set("当前对话已清空 · 记录已保留")
        return "break"

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(plain_text(self.rows))
        self.copy_button.configure(text="已复制")
        self.root.after(1500, lambda: self.copy_button.configure(text="复制全文"))

    def restore_session(self, path):
        """Restore the stopped view during an update; never resubmit or rewrite rows."""
        path = Path(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if not isinstance(row.get("text"), str):
                raise ValueError("记录格式不正确")
            row_time(row)
        self.clear()
        self.session = path
        self.rows = rows
        self.session_rows = list(rows)
        for i, row in enumerate(rows):
            self.append_row(i, row)
        self.last_caption = rows[-1]["text"] if rows else ""
        self.status.set(f"已停止 · {len(rows)} 段已保存")
        self.footer.set(f"已保存 {len(rows)} 段")

    def append_row(self, row_id, value):
        self.text.configure(state="normal")
        self.text.insert("end", row_time(value) + "  ", ("time", f"row:{row_id}"))
        self.text.insert("end", value["text"], (f"row:{row_id}", f"body:{row_id}"))
        self.text.insert("end", "\n")
        self.text.configure(state="disabled")

    def toggle_qa(self):
        self.qa.set_enabled(self.qa_enabled.get())
        self.text.tag_remove("qa_selected", "1.0", "end")
        self.qa_selection = None
        self.text.configure(cursor="hand2" if self.qa.enabled else "xterm")
        if self.qa.enabled:
            self.columns.add(self.qa_panel, minsize=150, stretch="always")
            self.root.after_idle(self.balance_columns)
            self.qa_status.set("点击左侧文字提问")
        else:
            self.columns.forget(self.qa_panel)
            self.qa_status.set("问答已关闭")

    def balance_columns(self):
        if self.qa.enabled and len(self.columns.panes()) == 2:
            self.columns.sash_place(0, self.columns.winfo_width() // 2, 0)

    def open_qa(self):
        if not self.qa.enabled:
            self.qa_enabled.set(True)
            self.toggle_qa()

    def question_press(self, event):
        self._question_press = (event.x, event.y)

    def question_release(self, event):
        if not self.qa.enabled:
            return
        origin = getattr(self, "_question_press", (event.x, event.y))
        if abs(origin[0] - event.x) + abs(origin[1] - event.y) > 5:
            return  # Drag selects text; the explicit button submits the selection.
        index = self.text.index(f"@{event.x},{event.y}")
        bounds = self.text.bbox(index)
        if not bounds or not bounds[1] <= event.y <= bounds[1] + bounds[3]:
            return  # Blank space below the transcript is not a question.
        for tag in self.text.tag_names(index):
            if tag.startswith("row:"):
                row_index = int(tag.split(":")[1])
                self.select_question(self.rows[row_index]["text"], [row_index])
                break

    def ask_selected(self):
        ranges = self.text.tag_ranges("sel")
        if not ranges:
            self.qa_status.set("先在左侧拖选文字，或直接点击一句")
            return
        start, end = map(str, ranges)
        parts, indices = [], []
        for i, row in enumerate(self.rows):
            tagged = self.text.tag_ranges(f"body:{i}")
            if not tagged:
                continue
            a, b = map(str, tagged)
            if self.text.compare(b, ">", start) and self.text.compare(a, "<", end):
                lo = start if self.text.compare(start, ">", a) else a
                hi = end if self.text.compare(end, "<", b) else b
                parts.append(self.text.get(lo, hi))
                indices.append(i)
        if parts:
            self.select_question("\n".join(parts).strip(), indices)

    def select_question(self, question, indices):
        if not self.qa.enabled or not question or not indices:
            return
        key = (tuple(indices), question)
        if key == self.qa_selection and self.qa.active:
            return
        self.qa.reset()
        self.qa_selection = key
        self.qa_question, self.qa_answer = question, self.qa_cache.get(key, "")
        self.text.tag_remove("qa_selected", "1.0", "end")
        self.text.tag_remove("sel", "1.0", "end")
        for i in indices:
            ranges = self.text.tag_ranges(f"row:{i}")
            if ranges:
                self.text.tag_add("qa_selected", *ranges)
        self.render_qa()
        if self.qa_answer:
            self.qa_status.set("已回答")
            return
        self.qa_status.set("正在回答…")
        # Only the selected passage and its preceding context go to Codex.
        context = [r["text"] for r in self.rows[max(0, indices[0]-8):indices[-1]+1]]
        self.qa.ask(question, context)

    def render_qa(self):
        self.qa_text.configure(state="normal")
        self.qa_text.delete("1.0", "end")
        if self.qa_question:
            self.qa_text.insert("end", self.qa_question + "\n\n", "question")
            self.qa_text.insert("end", self.qa_answer)
        self.qa_text.configure(state="disabled")

    def copy_answer(self):
        if self.qa_answer:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.qa_answer)

    def handle_qa(self, value):
        generation, state, payload = value
        if generation != self.qa.generation or not self.qa.enabled or self.closing:
            return
        if state == "thinking":
            self.qa_status.set("正在回答…")
        elif state == "answer":
            self.qa_question, self.qa_answer = payload
            if self.qa_selection is not None:
                self.qa_cache[self.qa_selection] = self.qa_answer
            self.qa_history.append(payload)
            self.qa_history = self.qa_history[-40:]
            self.render_qa()
            self.qa_status.set("已回答")
            if self.session:
                try:
                    with self.session.with_suffix(".qa.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps({"time": datetime.now().astimezone().isoformat(),
                                               "question": payload[0], "answer": payload[1]}, ensure_ascii=False) + "\n")
                except OSError:
                    self.qa_status.set("回答已显示，但保存失败；可复制回答")
        elif state == "error":
            self.qa_status.set(payload[:220])
        elif state == "done":
            self.qa.active = False

    def export(self):
        if not self.rows:
            messagebox.showinfo("导出", "尚无转写文字。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                  initialfile=datetime.now().strftime("声记-%Y%m%d-%H%M%S.txt"),
                  filetypes=[("文本", "*.txt"), ("字幕", "*.srt")])
        if path:
            try:
                Path(path).write_text(export_text(self.rows, path.lower().endswith(".srt")), encoding="utf-8-sig")
            except OSError as exc:
                messagebox.showerror("导出失败", str(exc))

    def open_caption(self):
        if self.caption_window and self.caption_window.winfo_exists():
            self.caption_window.lift()
            return
        self.caption_window = tk.Toplevel(self.root)
        self.caption_window.title("声记 · 实时字幕")
        self.caption_window.geometry("850x180")
        self.caption_window.configure(bg="#17233a")
        self.caption_window.attributes("-topmost", True)
        self.caption = tk.Label(self.caption_window, text="等待转写…", bg="#17233a", fg="white",
                                font=("Microsoft YaHei UI", 21), wraplength=800, padx=20, pady=20)
        self.caption.pack(fill="both", expand=True)
        self.caption_window.bind("<Configure>", lambda e: self.caption.configure(wraplength=max(200, e.width - 40))
                                 if e.widget == self.caption_window else None)
        self.refresh_caption()

    def refresh_caption(self):
        if self.caption_window and self.caption_window.winfo_exists():
            self.caption.configure(text=("临时 · " + self.draft_text) if self.draft_text else (self.last_caption or "等待转写…"),
                                   fg="#c6bdf4" if self.draft_text else "white")

    def show_preview(self, text):
        self.draft_text = text
        if text:
            # Bound the card height for a small window; the final transcript keeps all text.
            shown = ("…" + text[-110:]) if len(text) > 110 else text
            self.preview_label.configure(text=shown)
            if not self.preview_frame.winfo_manager():
                self.preview_frame.pack(side="bottom", fill="x", pady=(6, 0), before=self.text)
        else:
            self.preview_frame.pack_forget()
            self.preview_label.configure(text="")
        self.refresh_caption()

    def poll(self):
        for _ in range(300):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "qa":
                self.handle_qa(value)
            elif kind == "hotkey":
                if not self.root.grab_current():
                    self.toggle_recording()
            elif kind == "hotkey_status":
                ok, reason = value
                self.hotkey_hint.configure(text="Ctrl+空格" if ok else "快捷键不可用",
                                           foreground="#9ba3b3" if ok else "#bd544f")
                self.hotkey_status.set(("Ctrl+空格 · 全局开始 / 停止转写" if ok else f"Ctrl+空格 {reason}")
                                       + "\nCtrl+Backspace · 窗口内清空当前对话")
            elif kind == "level":
                self.set_level(value)
            elif kind == "backlog":
                self.footer.set(f"积压 {value:.0f}s · {len(self.session_rows)} 段" if value >= 2 else f"已保存 {len(self.session_rows)} 段")
            elif kind == "preview":
                self.show_preview(value)
            elif kind == "segment":
                self.last_caption = value["text"]
                self.show_preview("")
                self.rows.append(value)
                self.session_rows.append(value)
                try:
                    with self.session.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(value, ensure_ascii=False) + "\n")
                    self.session.with_suffix(".txt").write_text(export_text(self.session_rows), encoding="utf-8-sig")
                except OSError as exc:
                    self.failed = True
                    self.engine.stop()
                    self.status.set(f"自动保存失败，请导出当前文字：{exc}")
                self.append_row(len(self.rows) - 1, value)
                if not self.qa.enabled:
                    self.text.see("end")
                self.refresh_caption()
            elif kind == "error":
                self.failed = True
                self.status.set("转写失败：" + value)
                messagebox.showerror("转写失败", value)
            elif kind == "done":
                self.show_preview("")
                self.set_busy(False)
                self.set_level(0)
                if not self.failed:
                    self.status.set(f"已停止 · {len(self.session_rows)} 段已保存")
        self.root.after(100, self.poll)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.qa.set_enabled(False)
        self.hotkey.close()
        if self.busy:
            self.stop()
            self.status.set("正在处理剩余音频，完成后自动关闭…")
            self.root.after(200, self.close_when_done)
        else:
            self.root.destroy()

    def close_when_done(self):
        if self.busy:
            self.root.after(200, self.close_when_done)
        else:
            self.root.destroy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    app = App(tk.Tk())
    if args.restore:
        try:
            app.restore_session(args.restore)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("恢复转写失败", str(exc))
    if args.qa:
        app.open_qa()
    app.root.mainloop()
