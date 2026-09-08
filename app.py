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
from opencc import OpenCC
from hotkey import GlobalHotkey
from codex_qa import CodexQA
from segmentation import PauseSegmenter, WordAssembler, decode_chunk, DraftPreview
from recognition import RecognitionContext, read_preferences, save_preferences
from scrollbars import SlimScrollbar
from chat_view import ChatView
from icon_button import IconButton
from model_runtime import load_model
from floating_caption import FloatingCaption

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


def clean_caption(text):
    # Some short/noisy inputs cause Whisper to emit long runs of punctuation.
    text = re.sub(r"([.。…，,!?！？])\1{2,}", r"\1", text.strip())
    return SIMPLIFIED.convert(text) if re.search(r"[\w\u4e00-\u9fff]", text) else ""


class Transcriber:
    def __init__(self, events):
        self.events = events
        self.stop_event = threading.Event()
        self.context_reset = threading.Event()
        self.thread = None
        self.model = None
        self.model_name = None

    def emit(self, kind, value):
        self.events.put((kind, value))

    def start(self, device, model, language, hotwords=""):
        self.stop_event.clear()
        self.context_reset.clear()
        self.thread = threading.Thread(target=self.run, args=(device, model, language, hotwords), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def reset_context(self):
        self.context_reset.set()

    def run(self, device, model_name, language, hotwords=""):
        audio_api = None
        streams, states = [], {}
        chunks = queue.Queue(maxsize=1200)
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
            devices = device if isinstance(device, list) else [{"index": device}]
            if not devices:
                raise RuntimeError("请先选择音源。")
            for selected in devices:
                info = audio_api.get_device_info_by_index(selected["index"])
                source = selected.get("source", "system" if info.get("isLoopbackDevice") else "microphone")
                if not info.get("maxInputChannels"):
                    raise RuntimeError("所选设备不支持输入，请刷新后重新选择。")
                rate, channels = int(info["defaultSampleRate"]), int(info["maxInputChannels"])
                state = {"rate": rate, "segmenter": PauseSegmenter(), "assembler": WordAssembler(),
                         "preview": DraftPreview(), "guidance": RecognitionContext(hotwords),
                         "received": time.monotonic()}
                states[source] = state

                def callback(data, count, timing, flags, source=source, rate=rate, channels=channels):
                    if self.stop_event.is_set():
                        return (None, pa.paComplete)
                    if flags:
                        overflow.set()
                        self.stop_event.set()
                        return (None, pa.paAbort)
                    samples = np.frombuffer(data, dtype=np.float32).reshape(-1, channels).copy()
                    offset = max(0, time.monotonic() - origin - count / rate)
                    try:
                        chunks.put_nowait((source, offset, samples, time.time() - count / rate))
                    except queue.Full:
                        overflow.set()
                        self.stop_event.set()
                        return (None, pa.paAbort)
                    self.emit("level", min(100, float(np.sqrt(np.mean(samples ** 2))) * 500))
                    return (None, pa.paContinue)

                stream = audio_api.open(format=pa.paFloat32, channels=channels, rate=rate,
                                        input=True, input_device_index=selected["index"],
                                        frames_per_buffer=rate // 10, stream_callback=callback, start=False)
                streams.append(stream)
                state["stream"] = stream
            origin = time.monotonic()
            for stream in streams:
                stream.start_stream()
            backend = getattr(getattr(self.model, "model", None), "device", "cpu")
            source_names = " + ".join("系统声音" if x == "system" else "麦克风" for x in states)
            self.emit("status", f"正在转写 · {'GPU' if backend == 'cuda' else 'CPU'} · {source_names}")

            def transcribe(source, ready):
                state = states[source]
                for chunk in ready:
                    for start, end, text, epoch in decode_chunk(
                            self.model, state["assembler"], chunk, language, state["guidance"],
                            chunks.qsize() / (10 * len(states))):
                        text = clean_caption(text)
                        if text:
                            row = make_segment(start, end, text, epoch)
                            row["source"] = source
                            self.emit("segment", row)
                    pending = state["assembler"].pending
                    self.emit("preview", {"source": source, "text": clean_caption(pending[2]) if pending else ""})
                    state["preview"].defer()

            while not self.stop_event.is_set() or not chunks.empty():
                if self.context_reset.is_set():
                    for state in states.values():
                        state["guidance"].history.clear()
                        state["guidance"].last_end = None
                    self.context_reset.clear()
                try:
                    source, offset, samples, block_epoch = chunks.get(timeout=0.1)
                except queue.Empty:
                    for source, state in states.items():
                        if time.monotonic() - state["received"] > 0.7:
                            transcribe(source, state["segmenter"].finish())
                        if not self.stop_event.is_set() and not state["stream"].is_active():
                            raise RuntimeError("音频设备已断开或停止响应，请刷新设备后重试。")
                    continue
                state = states[source]
                state["received"] = time.monotonic()
                if not self.stop_event.is_set() and any(not s["stream"].is_active() for s in states.values()):
                    raise RuntimeError("音频设备已断开或停止响应，请刷新设备后重试。")
                transcribe(source, state["segmenter"].push(mono_16k(samples, state["rate"]), offset, block_epoch))
                backlog = chunks.qsize() / (10 * len(states))
                self.emit("backlog", backlog)
                try:
                    draft = state["preview"].render(self.model, state["assembler"], state["segmenter"],
                        language, backlog=backlog, stopping=self.stop_event.is_set(), guidance=state["guidance"])
                except Exception:
                    logging.exception("Skipping provisional subtitle; final decoding will retry")
                    state["preview"].defer()
                    draft = None
                if draft is not None and not self.stop_event.is_set():
                    self.emit("preview", {"source": source, "text": clean_caption(draft)})
            for source, state in states.items():
                transcribe(source, state["segmenter"].finish())
            if overflow.is_set():
                raise RuntimeError("音频缓冲溢出或采集不连续，已停止以避免静默丢字。请选更快的模型后重试。")
        except Exception as exc:
            logging.exception("Transcription failed")
            self.emit("error", str(exc))
        finally:
            self.stop_event.set()
            for stream in streams:
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
        self.multi_rows = set()
        self.qa_question = ""
        self.qa_answer = ""
        self.qa_status = tk.StringVar(value="等待声源中的问题…")
        self.rows = []
        self.session_rows = []
        self.devices = []
        self.input_devices = []
        self.drafts = {}
        self.bubble_selection = None
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
        root.title("闻录 · 系统声音实时转文字")
        icon_path = ROOT / "assets" / "wenlu.ico"
        if icon_path.exists():
            root.iconbitmap(default=str(icon_path))
        root.geometry("560x380")
        root.minsize(420, 280)
        root.overrideredirect(True)
        root.configure(bg="#e0e3ea")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f7f8fa")
        style.configure("TLabel", background="#f7f8fa", foreground="#858b98", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=5)
        # Clam's default readonly inputs use a dark, raised system-control face.
        # Give every settings input the same quiet surface as the content cards.
        for control in ("Settings.TCombobox", "Settings.TEntry"):
            style.configure(control, fieldbackground="white", background="white",
                            foreground="#4f586b", bordercolor="#e2e5ed", lightcolor="white",
                            darkcolor="white", borderwidth=1, relief="flat", padding=(8, 5),
                            arrowcolor="#9299aa", arrowsize=12, selectbackground="#E6F2FB",
                            selectforeground="#4f586b")
            style.map(control, fieldbackground=[("disabled", "#f1f3f7"), ("readonly", "white")],
                      background=[("active", "#F0F7FC"), ("readonly", "white")],
                      foreground=[("disabled", "#a2a8b5"), ("readonly", "#4f586b")],
                      bordercolor=[("focus", "#78B7E3"), ("active", "#B5D5ED")],
                      lightcolor=[("focus", "white")], darkcolor=[("focus", "white")])
        root.option_add("*TCombobox*Listbox.font", ("Microsoft YaHei UI", 9))
        root.option_add("*TCombobox*Listbox.background", "white")
        root.option_add("*TCombobox*Listbox.foreground", "#4f586b")
        root.option_add("*TCombobox*Listbox.selectBackground", "#E6F2FB")
        root.option_add("*TCombobox*Listbox.selectForeground", "#007ACC")
        root.option_add("*TCombobox*Listbox.relief", "flat")
        style.configure("Slim.Horizontal.TProgressbar", background="#007ACC", troughcolor="#edf0f7",
                        borderwidth=0, thickness=2)
        frame = ttk.Frame(root, padding=(12, 8, 12, 12))
        frame.pack(fill="both", expand=True, padx=1, pady=1)

        def button(parent, text, command, primary=False, width=None):
            surface = parent.cget("bg") if isinstance(parent, tk.Frame) else "#f7f8fa"
            icons = {"设置": "settings", "复制全文": "copy", "复制": "copy", "···": "more",
                     "刷新": "refresh", "×": "close", "—": "minimize"}
            factory = IconButton if text in icons else tk.Button
            widget = factory(parent, **({"icon": icons[text]} if text in icons else {}),
                             text=text, command=command, relief="flat", bd=0,
                             bg="#007ACC" if primary else surface,
                             fg="white" if primary else "#737b8c", activebackground="#006BB3" if primary else "#E6F2FB",
                             activeforeground="white" if primary else "#007ACC", disabledforeground="#a6acba",
                             font=("Microsoft YaHei UI", 9), padx=10, pady=5,
                             cursor="hand2", takefocus=True, **({"width": width} if width else {}))
            normal_bg = widget.cget("bg")
            widget.bind("<Enter>", lambda e: widget.configure(bg="#006BB3" if primary else "#EDF5FB") if str(widget.cget("state")) != "disabled" else None, add="+")
            widget.bind("<Leave>", lambda e: widget.configure(bg=normal_bg), add="+")
            return widget

        header = ttk.Frame(frame)
        header.pack(fill="x", pady=(0, 8))
        self.brand_image = tk.PhotoImage(file=str(ROOT / "assets" / "logo-24.png"))
        brand = ttk.Label(header, image=self.brand_image)
        brand.pack(side="left", padx=(0, 7))
        title = ttk.Label(header, text="闻录", foreground="#263044", font=("Microsoft YaHei UI", 12, "bold"))
        title.pack(side="left", padx=(0, 12))
        hint = ttk.Label(header, text="", foreground="#bd544f")
        self.hotkey_hint = hint
        drag_space = ttk.Frame(header)
        drag_space.pack(side="left", fill="x", expand=True)
        for widget in (header, brand, title, hint, drag_space):
            widget.bind("<ButtonPress-1>", self.drag_begin)
            widget.bind("<B1-Motion>", self.drag_move)
        button(header, "×", self.close).pack(side="right")
        button(header, "—", self.minimize).pack(side="right")
        self.settings_button = button(header, "设置", self.toggle_settings)
        self.settings_button.pack(side="right", padx=(4, 0))
        self.settings_button.bind("<Leave>", lambda e: self.settings_button.configure(
            bg="#E6F2FB" if self.settings_visible else "#f7f8fa"), add="+")
        tk.Checkbutton(header, text="问答", variable=self.qa_enabled, command=self.toggle_qa,
                       indicatoron=False, selectcolor="#E6F2FB", relief="flat", padx=11, pady=5,
                       bg="#f7f8fa", activebackground="#f7f8fa", fg="#737b8c",
                       font=("Microsoft YaHei UI", 9), bd=0, highlightthickness=0).pack(side="right")

        self.settings_window = tk.Toplevel(root)
        self.settings_window.withdraw()
        self.settings_window.title("闻录 · 设置")
        self.settings_window.transient(root)
        self.settings_window.resizable(False, False)
        self.settings_window.configure(bg="#e0e3ea")
        self.settings_window.protocol("WM_DELETE_WINDOW", self.hide_settings)
        self.settings_window.bind("<Escape>", self.hide_settings)
        self.settings_panel = ttk.Frame(self.settings_window, padding=20)
        self.settings_panel.pack(fill="both", expand=True, padx=1, pady=1)
        self.capture_mode = tk.StringVar(value="系统声音 + 麦克风")
        ttk.Label(self.settings_panel, text="采集").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 8))
        self.mode = ttk.Combobox(self.settings_panel, textvariable=self.capture_mode, state="readonly",
                                values=["系统声音 + 麦克风", "仅系统声音", "仅麦克风"],
                                style="Settings.TCombobox", font=("Microsoft YaHei UI", 9))
        self.mode.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(self.settings_panel, text="麦克风").grid(row=1, column=0, sticky="w", pady=(0, 8))
        self.microphone = ttk.Combobox(self.settings_panel, state="readonly", style="Settings.TCombobox",
                                      font=("Microsoft YaHei UI", 9))
        self.microphone.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(self.settings_panel, text="系统声音").grid(row=2, column=0, sticky="w", padx=(0, 12), pady=(0, 8))
        self.device = ttk.Combobox(self.settings_panel, state="readonly", width=24,
                                   style="Settings.TCombobox", font=("Microsoft YaHei UI", 9))
        self.device.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(0, 8))
        self.refresh_button = button(self.settings_panel, "刷新", self.refresh)
        self.refresh_button.grid(row=2, column=3, sticky="e", pady=(0, 8))
        ttk.Label(self.settings_panel, text="识别").grid(row=3, column=0, sticky="w", padx=(0, 12))
        self.model = ttk.Combobox(self.settings_panel, state="readonly", width=23,
                                  style="Settings.TCombobox", font=("Microsoft YaHei UI", 9),
                                  values=["small · 轻量准确", "large-v3-turbo · 高性能", "base · 均衡", "tiny · 更快"])
        self.model.current(0)
        self.model.grid(row=3, column=1, sticky="ew", padx=(0, 8))
        self.language = ttk.Combobox(self.settings_panel, state="readonly", width=10,
                                     style="Settings.TCombobox", font=("Microsoft YaHei UI", 9),
                                     values=["中文", "自动检测", "英语"])
        self.language.current(0)
        self.language.grid(row=3, column=2, sticky="ew")
        self.settings_panel.columnconfigure(1, weight=2)
        self.settings_panel.columnconfigure(2, weight=1)
        self.hotkey_status = tk.StringVar(value="")
        self.hotkey_error = ttk.Label(self.settings_panel, textvariable=self.hotkey_status,
                                      foreground="#bd544f", wraplength=460)
        ttk.Label(self.settings_panel, text="问答使用已登录的 Codex · 开启后联网发送相关转写并消耗账号额度",
                  wraplength=460).grid(row=6, column=0, columnspan=4, sticky="w", pady=(16, 0))
        self.hotwords = tk.StringVar(value=read_preferences(ROOT / "recognition-settings.json"))
        ttk.Label(self.settings_panel, text="热词 / 纠错").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.hotwords_entry = ttk.Entry(self.settings_panel, textvariable=self.hotwords,
                                        style="Settings.TEntry", font=("Microsoft YaHei UI", 9))
        self.hotwords_entry.grid(row=4, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(self.settings_panel, text="逗号分隔；定向纠错示例：大模形=大模型").grid(row=5, column=1, columnspan=3, sticky="w", pady=(4, 0))
        button(self.settings_panel, "完成", self.hide_settings, primary=True).grid(
            row=7, column=0, columnspan=4, sticky="e", pady=(16, 0))

        self.body = ttk.Frame(frame)
        self.body.pack(fill="both", expand=True)
        controls = ttk.Frame(self.body)
        controls.pack(side="bottom", fill="x", pady=(10, 0))
        self.start_button = button(controls, "开始转写", self.toggle_recording, primary=True)
        self.start_button.pack(side="left")
        self.more_button = button(controls, "···", self.open_menu)
        self.more_button.pack(side="right")
        self.footer = tk.StringVar(value="自动保存")
        self.storage_hint = tk.StringVar(value="自动保存")
        ttk.Label(controls, textvariable=self.storage_hint, font=("Microsoft YaHei UI", 8), foreground="#a0a6b2").pack(side="right", padx=8)
        grip = ttk.Label(root, text="◢", cursor="size_nw_se", font=("Segoe UI", 8))
        grip.place(relx=1, rely=1, anchor="se")
        grip.bind("<ButtonPress-1>", self.resize_begin)
        grip.bind("<B1-Motion>", self.resize_move)

        self.status = tk.StringVar(value="准备就绪")
        self.status_label = ttk.Label(self.body, textvariable=self.status, wraplength=510)
        self.status_label.pack(anchor="w", pady=(0, 8))
        self.level = tk.Canvas(self.body, height=2, bg="#edf0f7", bd=0, highlightthickness=0)
        self.level_bar = self.level.create_rectangle(0, 0, 0, 2, fill="#007ACC", outline="")
        self.level.pack(fill="x", pady=(0, 10))
        self.columns = tk.PanedWindow(self.body, orient="horizontal", bg="#f7f8fa",
                                      bd=0, sashwidth=7, sashrelief="flat", showhandle=False)
        self.columns.pack(fill="both", expand=True)
        self.left_panel = tk.Frame(self.columns, bg="white", highlightbackground="#eceef3", highlightcolor="#eceef3", highlightthickness=1)
        left_header = tk.Frame(self.left_panel, bg="white", padx=10, pady=3)
        left_header.pack(fill="x")
        tk.Label(left_header, text="转写", bg="white", fg="#737b8c", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")
        self.copy_button = button(left_header, "复制全文", self.copy)
        self.copy_button.configure(bg="white", font=("Microsoft YaHei UI", 8))
        self.copy_button.pack(side="right")
        self.columns.add(self.left_panel, minsize=150, stretch="always")
        self.text = tk.Text(self.left_panel, wrap="word", font=("Microsoft YaHei UI", 10),
                            bg="white", fg="#263044", relief="flat", bd=0, highlightthickness=0,
                            padx=14, pady=6, state="disabled", spacing1=2, spacing2=3, spacing3=10, height=5, width=1,
                            exportselection=False)
        self.text.tag_configure("time", foreground="#a3a8b4", font=("Consolas", 8), spacing1=3, spacing3=2)
        self.text.tag_configure("draft", foreground="#668EAB")
        # Indexed document backs excerpt selection and legacy saved transcripts.
        # The visible surface renders source-labelled selectable chat bubbles.
        self.chat = ChatView(self.left_panel, self.bubble_click, self.bubble_select_text)
        self.chat.pack(fill="both", expand=True)
        self.transcript_scrollbar = self.chat.scrollbar
        self.text.tag_configure("qa_selected", background="#E6F2FB", foreground="#007ACC")
        self.text.bind("<ButtonPress-1>", self.question_press)
        self.text.bind("<ButtonRelease-1>", self.question_release)
        self.qa_panel = tk.Frame(self.columns, bg="white", highlightbackground="#eceef3", highlightcolor="#eceef3", highlightthickness=1)
        qa_header = tk.Frame(self.qa_panel, bg="white", padx=10, pady=3)
        qa_header.pack(fill="x")
        tk.Label(qa_header, text="Codex", bg="white", fg="#007ACC", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")
        self.answer_copy_button = button(qa_header, "复制", self.copy_answer)
        self.answer_copy_button.configure(bg="white", font=("Microsoft YaHei UI", 8))
        self.answer_copy_button.pack(side="right")
        self.qa_status_label = tk.Label(self.qa_panel, textvariable=self.qa_status, wraplength=230,
                                        bg="white", fg="#9297a4", font=("Microsoft YaHei UI", 8), anchor="w", justify="left")
        self.qa_status_label.pack(side="bottom", fill="x", padx=10, pady=(0, 5))
        self.ask_selected_button = button(self.qa_panel, "发送所选", self.ask_selected)
        self.ask_selected_button.pack(side="bottom", anchor="e", padx=10, pady=(5, 3))
        self.qa_text = tk.Text(self.qa_panel, wrap="word", bg="white", fg="#263044", relief="flat",
                              font=("Microsoft YaHei UI", 10), padx=14, pady=6, spacing1=2, spacing2=3, spacing3=10,
                              state="disabled", width=1, height=1)
        self.qa_text.tag_configure("question", foreground="#527F9E", spacing3=14)
        self.qa_text.pack(fill="both", expand=True)
        self.answer_scrollbar = SlimScrollbar(self.qa_text)
        self.qa_placeholder = tk.Label(self.qa_text, text="选择文字，开始提问", bg="white", fg="#a0a5b1",
                                       font=("Microsoft YaHei UI", 9), justify="center")
        self.qa_placeholder.place(relx=0.5, rely=0.42, anchor="center")
        self.qa_panel.bind("<Configure>", lambda e: self.qa_status_label.configure(wraplength=max(110, e.width-10)))
        self.qa_status.set("")
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
        if self.settings_visible:
            self.settings_window.lift()
            return
        self.settings_visible = True
        self.settings_button.configure(bg="#E6F2FB", fg="#007ACC")
        window = self.settings_window
        window.update_idletasks()
        width, height = max(520, window.winfo_reqwidth()), window.winfo_reqheight()
        x = max(0, min(self.root.winfo_rootx() + (self.root.winfo_width()-width)//2,
                       window.winfo_screenwidth()-width))
        y = max(0, min(self.root.winfo_rooty() + (self.root.winfo_height()-height)//2,
                       window.winfo_screenheight()-height-40))
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.deiconify()
        window.lift()
        window.grab_set()
        self.device.focus_set()

    def hide_settings(self, event=None):
        self.settings_window.grab_release()
        self.settings_window.withdraw()
        self.settings_visible = False
        self.settings_button.configure(bg="#f7f8fa", fg="#737b8c")
        self.root.focus_set()
        return "break"

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
            old_output = self.devices[self.device.current()]["index"] if self.device.current() >= 0 else None
            old_input = self.input_devices[self.microphone.current()]["index"] if self.microphone.current() >= 0 else None
            with pa.PyAudio() as audio:
                self.devices = list(audio.get_loopback_device_info_generator())
                host = audio.get_host_api_info_by_type(pa.paWASAPI)
                self.input_devices = [audio.get_device_info_by_index(i) for i in range(audio.get_device_count())]
                self.input_devices = [d for d in self.input_devices if d["hostApi"] == host["index"]
                                      and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice")]
                try:
                    default = audio.get_default_wasapi_loopback()["index"]
                except OSError:
                    default = None
                default_input = host.get("defaultInputDevice")
            for widget, devices, previous, fallback in (
                    (self.device, self.devices, old_output, default),
                    (self.microphone, self.input_devices, old_input, default_input)):
                widget["values"] = [d["name"] for d in devices]
                if devices:
                    target = previous if any(d["index"] == previous for d in devices) else fallback
                    widget.current(next((i for i, d in enumerate(devices) if d["index"] == target), 0))
                else:
                    widget.set("")
            if not self.devices and not self.input_devices:
                self.status.set("未找到音频设备，请连接或启用设备后刷新。")
        except Exception as exc:
            self.status.set(f"设备枚举失败：{exc}")

    def set_busy(self, busy):
        self.busy = busy
        if not busy:
            self.storage_hint.set("自动保存")
        self.refresh_button.configure(state="disabled" if busy else "normal")
        self.hotwords_entry.configure(state="disabled" if busy else "normal")
        for widget in (self.device, self.microphone, self.mode, self.model, self.language):
            widget.configure(state="disabled" if busy else "readonly")
        self.start_button.configure(text="停止转写" if busy else "开始转写", state="normal")

    def start(self):
        selected = []
        for source, widget, devices, enabled in (
                ("system", self.device, self.devices, self.capture_mode.get() != "仅麦克风"),
                ("microphone", self.microphone, self.input_devices, self.capture_mode.get() != "仅系统声音")):
            if enabled:
                if widget.current() < 0:
                    messagebox.showinfo("选择音源", "请在设置中选择" + ("系统播放设备。" if source == "system" else "麦克风输入设备。"), parent=self.root)
                    self.toggle_settings()
                    return
                selected.append({"index": int(devices[widget.current()]["index"]), "source": source})
        try:
            save_preferences(ROOT / "recognition-settings.json", self.hotwords.get())
        except OSError as exc:
            messagebox.showerror("无法保存热词", str(exc))
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
        self.engine.start(selected,
                          self.model.get().split()[0],
                          {"中文": "zh", "英语": "en", "自动检测": None}[self.language.get()],
                          self.hotwords.get())

    def stop(self):
        self.engine.stop()
        self.start_button.configure(text="收尾中…", state="disabled")
        self.status.set("正在处理剩余音频…")

    def clear(self, preserve_recording=False):
        self.qa.reset()
        self.qa_history.clear()
        self.qa_cache.clear()
        self.qa_selection = None
        self.multi_rows.clear()
        self.ask_selected_button.configure(text="发送所选")
        self.qa_question = self.qa_answer = ""
        self.qa_status.set("")
        self.render_qa()
        self.rows = []
        self.chat.clear()
        self.bubble_selection = None
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
        self.engine.reset_context()
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
        follow = self.text.yview()[1] >= 0.995
        self.text.configure(state="normal")
        self.chat.finalize(value.get("source", "system"), {**value, "id": row_id, "time": row_time(value)})
        self.text.insert("end", row_time(value) + "\n", ("time", f"row:{row_id}"))
        self.text.insert("end", value["text"], (f"row:{row_id}", f"body:{row_id}"))
        self.text.insert("end", "\n")
        self.text.configure(state="disabled")
        if follow:
            self.text.see("end")

    def toggle_qa(self):
        self.qa.set_enabled(self.qa_enabled.get())
        self.text.tag_remove("qa_selected", "1.0", "end")
        self.chat.highlight(set())
        self.qa_selection = None
        self.multi_rows.clear()
        self.ask_selected_button.configure(text="发送所选")
        self.text.configure(cursor="hand2" if self.qa.enabled else "xterm")
        if self.qa.enabled:
            self.columns.add(self.qa_panel, minsize=150, stretch="always")
            self.root.after_idle(self.balance_columns)
            self.qa_status.set("")
        else:
            self.columns.forget(self.qa_panel)
            self.qa_status.set("问答已关闭")

    def balance_columns(self):
        if self.qa.enabled and len(self.columns.panes()) == 2:
            self.columns.sash_place(0, round(self.columns.winfo_width() * 0.58), 0)

    def open_qa(self):
        if not self.qa.enabled:
            self.qa_enabled.set(True)
            self.toggle_qa()

    def bubble_click(self, row_id, control):
        self.bubble_selection = None
        if control:
            self.toggle_question_selection(row_id)
        elif self.qa.enabled:
            self.select_question(self.rows[row_id]["text"], [row_id])

    def bubble_select_text(self, row_id, text):
        self.bubble_selection = (row_id, text)
        self.multi_rows.clear()
        self.ask_selected_button.configure(text="发送所选")
        for key, bubble in self.chat.bubbles.items():
            bubble.set_selected(False)
            if key != row_id:
                bubble.text.tag_remove("sel", "1.0", "end")

    def question_press(self, event):
        self._question_press = (event.x, event.y)
        self._question_control = bool(event.state & 0x0004)
        if self.qa.enabled and self._question_control:
            return "break"  # Prevent Tk's Ctrl-click selection behaviour.
        if self.qa.enabled and self.multi_rows:
            self.multi_rows.clear()
            self.ask_selected_button.configure(text="发送所选")
            self.text.tag_remove("qa_selected", "1.0", "end")
            self.chat.highlight(set())

    def question_release(self, event):
        if not self.qa.enabled:
            return
        origin = getattr(self, "_question_press", (event.x, event.y))
        control = getattr(self, "_question_control", False)
        if abs(origin[0] - event.x) + abs(origin[1] - event.y) > 5:
            return  # Drag selects text; the explicit button submits the selection.
        index = self.text.index(f"@{event.x},{event.y}")
        bounds = self.text.bbox(index)
        if not bounds or not bounds[1] <= event.y <= bounds[1] + bounds[3]:
            return  # Blank space below the transcript is not a question.
        for tag in self.text.tag_names(index):
            if tag.startswith("row:"):
                row_index = int(tag.split(":")[1])
                if control:
                    self.toggle_question_selection(row_index)
                else:
                    self.select_question(self.rows[row_index]["text"], [row_index])
                break
        if control:
            return "break"

    def toggle_question_selection(self, row_index):
        if not self.qa.enabled or not 0 <= row_index < len(self.rows):
            return
        if row_index in self.multi_rows:
            self.multi_rows.remove(row_index)
        else:
            self.multi_rows.add(row_index)
        self.qa.reset()
        self.qa_selection = None
        self.qa_question = "\n".join(self.rows[i]["text"] for i in sorted(self.multi_rows))
        self.qa_answer = ""
        self.text.tag_remove("qa_selected", "1.0", "end")
        self.chat.highlight(set())
        self.text.tag_remove("sel", "1.0", "end")
        for i in sorted(self.multi_rows):
            self.text.tag_add("qa_selected", *self.text.tag_ranges(f"row:{i}"))
        self.chat.highlight(self.multi_rows)
        count = len(self.multi_rows)
        self.ask_selected_button.configure(text=f"发送 {count} 条" if count else "发送所选")
        self.qa_status.set("")
        self.render_qa()

    def ask_selected(self):
        if self.multi_rows:
            indices = sorted(self.multi_rows)
            question = "\n".join(self.rows[i]["text"] for i in indices)
            self.select_question(question, indices)
            return
        if self.bubble_selection:
            row_id, question = self.bubble_selection
            self.select_question(question, [row_id])
            return
        ranges = self.text.tag_ranges("sel")
        if not ranges:
            self.qa_status.set("Ctrl+左键多选，或在左侧拖选文字")
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

    def clear_question_selection(self):
        self.multi_rows.clear()
        self.bubble_selection = None
        self.ask_selected_button.configure(text="发送所选")
        self.text.tag_remove("qa_selected", "1.0", "end")
        self.text.tag_remove("sel", "1.0", "end")
        self.chat.highlight(set())

    def select_question(self, question, indices):
        if not self.qa.enabled or not question or not indices:
            return
        if len(question) > 2400:
            self.qa_status.set("选中内容超过 2400 字，请减少选择后发送")
            return
        key = (tuple(indices), question)
        if key == self.qa_selection and self.qa.active:
            self.clear_question_selection()
            return
        self.qa.reset()
        self.qa_selection = key
        self.qa_question, self.qa_answer = question, self.qa_cache.get(key, "")
        self.render_qa()
        if self.qa_answer:
            self.qa_status.set("")
            self.clear_question_selection()
            return
        self.qa_status.set("正在回答…")
        # Only the selected passage and its preceding context go to Codex.
        context = ([r["text"] for r in self.rows[max(0, indices[0]-8):indices[0]]]
                   + [self.rows[i]["text"] for i in indices])
        self.qa.ask(question, context)
        self.clear_question_selection()

    def render_qa(self):
        self.answer_copy_button.configure(state="normal" if self.qa_answer else "disabled")
        if self.qa_question:
            self.qa_placeholder.place_forget()
        else:
            self.qa_placeholder.place(relx=0.5, rely=0.42, anchor="center")
        self.qa_text.configure(state="normal")
        self.qa_text.delete("1.0", "end")
        if self.qa_question:
            self.qa_text.insert("end", self.qa_question + "\n", "question")
            self.qa_text.insert("end", self.qa_answer)
        self.qa_text.configure(state="disabled")

    def copy_answer(self):
        if self.qa_answer:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.qa_answer)
            self.answer_copy_button.configure(text="已复制")
            self.root.after(1500, lambda: self.answer_copy_button.configure(text="复制"))

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
            self.qa_status.set("")
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
                  initialfile=datetime.now().strftime("闻录-%Y%m%d-%H%M%S.txt"),
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
        self.caption_window = FloatingCaption(self.root)
        self.refresh_caption()

    def refresh_caption(self):
        if self.caption_window and self.caption_window.winfo_exists():
            self.caption_window.set_text(self.draft_text or self.last_caption or "等待转写…",
                                         draft=bool(self.draft_text))

    def show_preview(self, text, source=None, keep_bubble=False):
        if isinstance(text, dict):
            source, text = text["source"], text["text"]
        if source is None and not text:
            self.drafts.clear()
            for origin in ("system", "microphone"):
                self.chat.draft(origin, "")
        else:
            source = source or "system"
            if text:
                self.drafts[source] = text
            else:
                self.drafts.pop(source, None)
            if not keep_bubble:
                self.chat.draft(source, text)
        self.draft_text = "\n".join(self.drafts.values())
        self.text.configure(state="normal")
        ranges = self.text.tag_ranges("draft")
        if ranges:
            self.text.delete(ranges[0], ranges[-1])
        if self.draft_text:
            self.text.insert("end", self.draft_text + "\n", ("draft",))
        self.text.configure(state="disabled")
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
                self.hotkey_status.set("" if ok else f"快捷键不可用：{reason}")
                if ok:
                    self.hotkey_hint.pack_forget()
                    self.hotkey_error.grid_remove()
                else:
                    self.hotkey_hint.configure(text="快捷键不可用")
                    self.hotkey_hint.pack(side="left")
                    self.hotkey_error.grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))
            elif kind == "level":
                self.set_level(value)
            elif kind == "backlog":
                self.storage_hint.set(f"积压 {value:.0f}s" if value >= 2 else "自动保存")
                self.footer.set(f"积压 {value:.0f}s · {len(self.session_rows)} 段" if value >= 2 else f"已保存 {len(self.session_rows)} 段")
            elif kind == "preview":
                self.show_preview(value)
            elif kind == "segment":
                self.last_caption = value["text"]
                self.show_preview("", value.get("source", "system"), keep_bubble=True)
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
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Wenlu.Desktop")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--captions", action="store_true", help="Open floating captions on startup")
    parser.add_argument("--geometry", help="Window geometry when restoring a running view")
    args = parser.parse_args()
    app = App(tk.Tk())
    if args.geometry:
        app.root.geometry(args.geometry)
    if args.restore:
        try:
            app.restore_session(args.restore)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("恢复转写失败", str(exc))
    if args.qa:
        app.open_qa()
    if args.captions:
        app.open_caption()
    app.root.mainloop()
