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
import typography
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pyaudiowpatch as pa
from scipy.signal import resample_poly
from opencc import OpenCC
from hotkey import GlobalHotkey
from codex_qa import CodexQA
from codex_connection import CodexConnection
from qa_provider import APIProvider, PROVIDERS, DEFAULTS, load_settings
import theme
from segmentation import PauseSegmenter, WordAssembler, decode_chunk, DraftPreview
from paragraphs import ParagraphAssembler
from recognition import RecognitionContext, read_preferences, save_preferences
from scrollbars import SlimScrollbar
from chat_view import ChatView
from ui_components import UIControls, ToggleChip, SplitterHandle
from settings_view import build_settings
from audio_levels import AudioLevelNormalizer
from model_runtime import load_model
from floating_caption import FloatingCaption
from app_paths import DATA, LOGS, RECORDINGS, preferences, save_desktop
from version import VERSION

ROOT = Path(__file__).resolve().parent
SIMPLIFIED = OpenCC("t2s")
logging.basicConfig(filename=LOGS / "app.log", level=logging.INFO,
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
        paragraphs = ParagraphAssembler()
        try:
            self.emit("status", "正在加载本地模型…")
            if self.model_name != model_name:
                self.model = None
                self.model_name = None
                self.model = load_model(model_name, status=lambda text: self.emit('status', text))
                self.model_name = model_name
            if self.stop_event.is_set():
                return
            self.emit('status', '正在连接音频设备…')
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
                         "levels": AudioLevelNormalizer(),
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
                            for paragraph in paragraphs.consume(row):
                                self.emit("segment", paragraph)
                    pending = state["assembler"].pending
                    self.emit("preview", {"source": source, "text": paragraphs.preview(source, clean_caption(pending[2]) if pending else "")})
                    state["preview"].defer()

            while not self.stop_event.is_set() or not chunks.empty():
                if self.context_reset.is_set():
                    paragraphs.pending = None
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
                        if time.monotonic() - state["received"] >= paragraphs.pause and paragraphs.pending and paragraphs.pending['source'] == source:
                            for paragraph in paragraphs.finish():
                                self.emit('segment', paragraph)
                        if not self.stop_event.is_set() and not state["stream"].is_active():
                            raise RuntimeError("音频设备已断开或停止响应，请刷新设备后重试。")
                    continue
                state = states[source]
                state["received"] = time.monotonic()
                if not self.stop_event.is_set() and any(not s["stream"].is_active() for s in states.values()):
                    raise RuntimeError("音频设备已断开或停止响应，请刷新设备后重试。")
                audio = state["levels"].process(mono_16k(samples, state["rate"]))
                transcribe(source, state["segmenter"].push(audio, offset, block_epoch))
                for paragraph in paragraphs.silence(source, offset + len(audio)/16000,
                        getattr(state['segmenter'], 'last_speech_end', None)):
                    self.emit('segment', paragraph)
                if paragraphs.pending and chunks.empty():
                    owner = paragraphs.pending['source']
                    if owner != source and time.monotonic()-states[owner]['received'] >= paragraphs.pause:
                        # Loopback can stop delivering samples while the mic keeps
                        # delivering silence, so the queue-empty handler never runs.
                        transcribe(owner, states[owner]['segmenter'].finish())
                        for paragraph in paragraphs.finish():
                            self.emit('segment', paragraph)
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
                    self.emit("preview", {"source": source, "text": paragraphs.preview(source, clean_caption(draft))})
            for source, state in states.items():
                transcribe(source, state["segmenter"].finish())
            if overflow.is_set():
                raise RuntimeError("音频缓冲溢出或采集不连续，已停止以避免静默丢字。请选更快的模型后重试。")
        except Exception as exc:
            logging.exception("Transcription failed")
            self.emit("error", str(exc))
        finally:
            for paragraph in paragraphs.finish():
                self.emit('segment', paragraph)
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
        typography.setup(root)
        self.events = queue.Queue()
        self.engine = Transcriber(self.events)
        self.qa = CodexQA(self.events)
        self.qa.context_provider = self.session_context_snapshot
        self.codex_connection = CodexConnection(self.events)
        self.qa_settings = load_settings()
        self.qa_provider_name = tk.StringVar(value=PROVIDERS[self.qa_settings['provider']])
        self.codex_status = tk.StringVar(value="尚未检测")
        self.codex_detail = tk.StringVar(value="点击检测会发送测试请求，消耗少量 Codex 额度。")
        self.dark_mode = tk.BooleanVar(value=preferences().get('dark_mode', False))
        try:
            size = max(8, min(16, int(preferences().get('font_size', 9))))
        except (ValueError, TypeError):
            size = 9
        self.font_size = tk.IntVar(value=size)
        root._body_font_size = size
        self.qa_enabled = tk.BooleanVar(value=False)
        self.auto_qa = tk.BooleanVar(value=False)
        self.qa_history = []
        self.qa_display_history = []
        self.question_drafts = {}
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
        style.configure("TLabel", background="#f7f8fa", foreground="#858b98", font=(typography.UI_FAMILY, 9))
        style.configure("TButton", font=(typography.UI_FAMILY, 9), padding=5)
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
        root.option_add("*TCombobox*Listbox.font", (typography.UI_FAMILY, 9))
        root.option_add("*TCombobox*Listbox.background", "white")
        root.option_add("*TCombobox*Listbox.foreground", "#4f586b")
        root.option_add("*TCombobox*Listbox.selectBackground", "#E6F2FB")
        root.option_add("*TCombobox*Listbox.selectForeground", "#007ACC")
        root.option_add("*TCombobox*Listbox.relief", "flat")
        style.configure("Slim.Horizontal.TProgressbar", background="#007ACC", troughcolor="#edf0f7",
                        borderwidth=0, thickness=2)
        frame = self.main_frame = ttk.Frame(root, padding=0)
        frame.pack(fill="both", expand=True, padx=1, pady=1)

        button = UIControls(self).button

        header = self.header = ttk.Frame(frame, padding=(12, 6, 12, 6))
        header.pack(fill="x")
        self.brand_image = tk.PhotoImage(file=str(ROOT / "assets" / "logo-24.png"))
        brand = ttk.Label(header, image=self.brand_image)
        brand.pack(side="left", padx=(0, 7))
        hint = ttk.Label(header, text="", foreground="#bd544f")
        self.hotkey_hint = hint
        drag_space = ttk.Frame(header)
        drag_space.pack(side="left", fill="x", expand=True)
        for widget in (header, brand, hint, drag_space):
            widget.bind("<ButtonPress-1>", self.drag_begin)
            widget.bind("<B1-Motion>", self.drag_move)
        button(header, "×", self.close).pack(side="right")
        button(header, "—", self.minimize).pack(side="right")
        self.settings_button = button(header, "设置", self.toggle_settings)
        self.settings_button.pack(side="right", padx=(4, 0))
        self.settings_button.bind("<Leave>", lambda e: self.settings_button.configure(
            bg=self.theme_color("#E6F2FB" if self.settings_visible else "#f7f8fa")), add="+")
        self.ai_button = ToggleChip(header, 'AI', self.qa_enabled, self.toggle_qa, surface='#f7f8fa')
        self.ai_button.pack(side='right', padx=(4, 0))

        build_settings(self, button)

        self.body = ttk.Frame(frame)
        self.body.pack(fill="both", expand=True)
        controls = self.controls = ttk.Frame(header)
        controls.pack(side="right", padx=(0, 6))
        self.footer = tk.StringVar(value="自动保存")
        self.storage_hint = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.storage_hint, font=(typography.UI_FAMILY, 8), foreground="#a0a6b2").pack(side="right", padx=8)
        self.resize_handles = {}
        edges = {
            'n': ('size_ns', dict(x=8, y=0, relwidth=1, width=-16, height=4)),
            's': ('size_ns', dict(x=8, rely=1, y=-4, relwidth=1, width=-16, height=4)),
            'w': ('size_we', dict(x=0, y=8, width=4, relheight=1, height=-16)),
            'e': ('size_we', dict(relx=1, x=-4, y=8, width=4, relheight=1, height=-16)),
            'nw': ('size_nw_se', dict(x=0, y=0, width=8, height=8)),
            'ne': ('size_ne_sw', dict(relx=1, x=-8, y=0, width=8, height=8)),
            'sw': ('size_ne_sw', dict(x=0, rely=1, y=-8, width=8, height=8)),
            'se': ('size_nw_se', dict(relx=1, x=-8, rely=1, y=-8, width=8, height=8)),
        }
        for edge, (cursor, placement) in edges.items():
            handle = tk.Frame(root, bg="#f7f8fa", bd=0, cursor=cursor)
            handle.place(**placement)
            handle.bind('<ButtonPress-1>', lambda event, edge=edge: self.resize_begin(event, edge))
            handle.bind('<B1-Motion>', self.resize_move)
            handle.bind('<ButtonRelease-1>', self.flush_resize)
            self.resize_handles[edge] = handle

        self.status = tk.StringVar(value="准备就绪")
        self.status_label = ttk.Label(self.body, textvariable=self.status, wraplength=510)
        self.level = tk.Canvas(self.body, height=2, bg="#edf0f7", bd=0, highlightthickness=0)
        self.level_bar = self.level.create_rectangle(0, 0, 0, 2, fill="#007ACC", outline="")
        self.columns = tk.PanedWindow(self.body, orient="horizontal", bg="#E3E9F0",
                                      bd=0, sashwidth=1, sashrelief="flat", showhandle=False,
                                      opaqueresize=False, proxybackground="#007ACC",
                                      proxyborderwidth=0, proxyrelief="flat")
        self.columns.pack(fill="both", expand=True)
        self.left_panel = tk.Frame(self.columns, bg="white", bd=0, highlightthickness=0)
        self.copy_button = button(self.left_panel, "复制全文", self.copy)
        self.columns.add(self.left_panel, minsize=150, stretch="always")
        self.text = tk.Text(self.left_panel, wrap="word", font=(typography.UI_FAMILY, 10),
                            bg="white", fg="#263044", relief="flat", bd=0, highlightthickness=0,
                            padx=14, pady=6, state="disabled", spacing1=2, spacing2=3, spacing3=10, height=5, width=1,
                            exportselection=False)
        self.text.tag_configure("time", foreground="#a3a8b4", font=("Consolas", 8), spacing1=3, spacing3=2)
        self.text.tag_configure("draft", foreground="#668EAB")
        # Indexed document backs excerpt selection and legacy saved transcripts.
        # The visible surface renders source-labelled selectable chat bubbles.
        self.chat = ChatView(self.left_panel, self.bubble_click, self.bubble_select_text)
        self.chat.pack(fill="both", expand=True)
        self.copy_button.place(relx=1, x=-14, y=3, anchor="ne")
        self.copy_button.lift()
        self.start_button = button(self.left_panel, "开始转写", self.toggle_recording)
        self.start_button.place(relx=1, rely=1, x=-14, y=-6, anchor="se")
        self.start_button.lift()
        self.transcript_scrollbar = self.chat.scrollbar
        self.text.tag_configure("qa_selected", background="#E6F2FB", foreground="#007ACC")
        self.text.bind("<ButtonPress-1>", self.question_press)
        self.text.bind("<ButtonRelease-1>", self.question_release)
        self.qa_panel = tk.Frame(self.columns, bg="white", bd=0, highlightthickness=0)
        self.splitter_handle = SplitterHandle(self.columns)
        for panel in (self.left_panel, self.qa_panel):
            panel.bind('<Configure>', self.splitter_handle.position, add='+')
        qa_header = self.qa_header = tk.Frame(self.qa_panel, bg="white", padx=12, pady=5)
        qa_header.pack(fill="x")
        tk.Label(qa_header, textvariable=self.qa_provider_name, bg="white", fg="#737b8c", font=(typography.UI_FAMILY, 9, "bold")).pack(side="left")
        self.answer_copy_button = button(qa_header, "复制", self.copy_answer)
        self.answer_copy_button.configure(bg="white", font=(typography.UI_FAMILY, 8))
        self.answer_copy_button.pack(side="right")
        self.codex_status_label = tk.Label(qa_header, textvariable=self.codex_status,
                                          bg="white", fg="#9297a4", font=(typography.UI_FAMILY, 8))
        self.codex_status_label.pack(side="left", padx=(8, 0))
        self.auto_button = ToggleChip(qa_header, 'Auto', self.auto_qa, self.toggle_auto_qa)
        self.auto_button.pack(side="right", padx=(4, 0), before=self.codex_status_label)
        # Keep a drag surface when the normal title bar is hidden. Text widgets
        # retain their selection behavior; copy/send buttons remain clickable.
        for widget in (qa_header, self.chat.canvas, self.chat.inner,
                       *(child for child in qa_header.winfo_children() if isinstance(child, tk.Label))):
            widget.bind('<ButtonPress-1>', self.focus_drag_begin, add='+')
            widget.bind('<B1-Motion>', self.focus_drag_move, add='+')
        qa_footer = tk.Frame(self.qa_panel, bg="white")
        qa_footer.pack(side="bottom", fill="x", padx=10, pady=(2, 3))
        self.ask_selected_button = button(qa_footer, "发送所选", self.ask_selected)
        self.ask_selected_button.pack(side="right")
        from reference_settings import build_add_button
        build_add_button(self, qa_footer)
        from reference_settings import ReferencePathLabel
        self.qa_status_label = self.reference_path_label = ReferencePathLabel(
            qa_footer, self.qa_status, self.reference_controls['remove_path'])
        self.reference_path_label.set_paths(self.reference_controls['paths']())
        self.qa_status_label.pack(side="left", fill="x", expand=True)
        self.qa_text = tk.Text(self.qa_panel, wrap="word", bg="white", fg="#263044", relief="flat",
                              font=(typography.UI_FAMILY, self.font_size.get()), padx=16, pady=10, spacing1=2, spacing2=5, spacing3=12,
                              state="disabled", width=1, height=1)
        self.qa_text.tag_configure("question", foreground="#007ACC", spacing1=8, spacing3=10)
        self.qa_text.tag_bind("question", "<Button-1>", self.edit_question)
        self.qa_text.pack(fill="both", expand=True)
        self.answer_scrollbar = SlimScrollbar(self.qa_text)
        self.qa_placeholder = tk.Label(self.qa_text, text="选择文字，开始提问", bg="white", fg="#a0a5b1",
                                       font=(typography.UI_FAMILY, 9), justify="center")
        self.qa_placeholder.place(relx=0.5, rely=0.42, anchor="center")
        self.qa_status.set("")
        self.refresh()
        self.load_desktop_settings()
        self.tray = None
        self.apply_theme()
        self.configure_qa_provider()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Configure>", self.on_resize)
        self.focus_mode = False
        root.bind('<F11>', self.toggle_focus_mode)
        root.bind('<F12>', self.toggle_auto_shortcut)
        root.bind('<Escape>', self.exit_focus_mode)
        for widget in (root, self.text, self.qa_text):
            widget.bind("<Control-BackSpace>", self.clear_conversation)
        root.after(100, self.enable_taskbar)
        root.bind("<Map>", self.on_main_window_mapped, add="+")
        root.after(200, self.start_tray)
        root.after(100, self.poll)
        self.hotkey.start()

    def theme_color(self, value):
        return theme.color(value, self.dark_mode.get())

    def toggle_focus_mode(self, event=None):
        if self.root.grab_current() is not None:
            return
        if self.focus_mode:
            return self.exit_focus_mode()
        self.clear_resize_preview()
        self.root.update_idletasks()
        self._normal_view = dict(geometry=self.root.geometry(),
            topmost=self.root.attributes('-topmost'), qa=self.qa_enabled.get(),
            sash=self.columns.sash_coord(0)[0] if len(self.columns.panes()) == 2 else None)
        self.focus_mode = True
        self.open_qa()
        self.header.pack_forget()
        self.status_label.pack_forget()
        for handle in self.resize_handles.values():
            handle.lift()
        self.main_frame.configure(padding=0)
        self.main_frame.pack_configure(padx=0, pady=0)
        return 'break'

    def exit_focus_mode(self, event=None):
        if not self.focus_mode or self.root.grab_current() is not None:
            return
        self.focus_mode = False
        view = self._normal_view
        self.main_frame.configure(padding=0)
        self.main_frame.pack_configure(padx=1, pady=1)
        self.header.pack(fill='x', before=self.body)
        for handle in self.resize_handles.values():
            handle.lift()
        if not view['qa']:
            self.qa_enabled.set(False)
            self.toggle_qa()
        self.root.attributes('-topmost', view['topmost'])
        # Keep any position/size chosen while focused when restoring chrome.
        self.root.update_idletasks()
        if view['sash'] is not None:
            self.columns.sash_place(0, view['sash'], 0)
        return 'break'

    def apply_theme(self):
        theme.apply(self.root, self.dark_mode.get())

    def change_theme(self):
        self.apply_theme()
        self.save_desktop_settings()

    def change_font_size(self, event=None):
        size = self.font_size.get()
        self.root._body_font_size = size
        self.qa_text.configure(font=(typography.UI_FAMILY, size))
        self.text.configure(font=(typography.UI_FAMILY, size))
        for bubble in self.chat.bubbles.values():
            bubble.font.configure(size=size)
            bubble.text.configure(font=bubble.font)
            bubble.measured_text = None
            bubble.layout_key = None
        self.chat.resize()
        self.save_desktop_settings()

    def load_desktop_settings(self):
        values = preferences()
        for key, widget in (("model", self.model), ("language", self.language),
                            ("mode", self.mode), ("output", self.device), ("input", self.microphone)):
            if values.get(key) in widget['values']:
                widget.set(values[key])

    def save_desktop_settings(self):
        try:
            save_preferences(DATA / "recognition-settings.json", self.hotwords.get())
            save_desktop(dict(model=self.model.get(), language=self.language.get(), mode=self.mode.get(),
                              output=self.device.get(), input=self.microphone.get(), dark_mode=self.dark_mode.get(),
                              font_size=self.font_size.get()))
        except OSError:
            logging.exception('Saving desktop settings')

    def open_folder(self, path):
        import os
        os.startfile(str(path))

    def start_tray(self):
        try:
            if self.tray is None:
                import pystray
                from PIL import Image
                self.tray = pystray.Icon('Wenlu', Image.open(ROOT / 'assets/logo-32.png'), '闻录',
                    pystray.Menu(pystray.MenuItem('打开闻录', lambda: self.events.put(('tray_show', None)), default=True),
                                 pystray.MenuItem('退出', lambda: self.events.put(('tray_exit', None)))))
                self.tray.run_detached()
            return True
        except Exception as exc:
            self.tray = None
            logging.exception('Starting system tray')
            return False

    def hide_to_tray(self):
        if self.start_tray():
            self.root.withdraw()
        else:
            messagebox.showerror('托盘不可用', '无法创建托盘图标，请重试。', parent=self.root)

    def import_legacy_data(self):
        if self.busy:
            messagebox.showinfo('迁移数据', '请先停止转写。', parent=self.root)
            return
        folder = filedialog.askdirectory(title='选择旧版闻录文件夹', parent=self.root)
        if folder:
            try:
                from app_paths import migrate_legacy
                migrate_legacy(folder)
                self.hotwords.set(read_preferences(DATA / 'recognition-settings.json'))
                messagebox.showinfo('迁移完成', '缺少的记录已复制，已有记录和原文件均保留。', parent=self.root)
            except OSError as exc:
                messagebox.showerror('迁移失败', str(exc), parent=self.root)

    def open_model_manager(self):
        from desktop_dialogs import model_manager
        model_manager(self)

    def open_about(self):
        from desktop_dialogs import about
        about(self)

    def drag_begin(self, event):
        self._drag_origin = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def focus_drag_begin(self, event):
        self._drag_origin = None
        if self.focus_mode:
            self.drag_begin(event)

    def focus_drag_move(self, event):
        if self.focus_mode:
            self.drag_move(event)

    def drag_move(self, event):
        if self._drag_origin:
            x, y = event.x_root - self._drag_origin[0], event.y_root - self._drag_origin[1]
            # A leading '-' means distance from the opposite screen edge in
            # Tk geometry. '+-20' is the absolute coordinate -20 instead.
            self.root.geometry(f"+{x}+{y}")

    def resize_begin(self, event, edge='se'):
        self._resize_origin = (event.x_root, event.y_root, self.root.winfo_width(), self.root.winfo_height())
        self._resize_edge = edge
        self._resize_position = (self.root.winfo_rootx(), self.root.winfo_rooty())
        self._pending_geometry = None
        self.clear_resize_preview()
        preview = self._resize_preview = tk.Toplevel(self.root)
        preview.withdraw()
        preview.overrideredirect(True)
        preview.attributes('-disabled', True)
        preview.attributes('-topmost', True)
        preview.configure(bg='#ff00ff')
        preview.attributes('-transparentcolor', '#ff00ff')
        tk.Frame(preview, bg='#ff00ff', highlightbackground='#007ACC',
                 highlightthickness=2, bd=0).pack(fill='both', expand=True)
        preview.geometry(f'{self.root.winfo_width()}x{self.root.winfo_height()}'
                         f'+{self.root.winfo_rootx()}+{self.root.winfo_rooty()}')
        preview.deiconify()

    def clear_resize_preview(self):
        preview = getattr(self, '_resize_preview', None)
        if preview is not None:
            preview.destroy()
            self._resize_preview = None

    def resize_move(self, event):
        x, y, width, height = self._resize_origin
        edge = getattr(self, '_resize_edge', 'se')
        left, top = getattr(self, '_resize_position', (self.root.winfo_rootx(), self.root.winfo_rooty()))
        dx, dy = event.x_root - x, event.y_root - y
        new_width = max(420, width + (-dx if 'w' in edge else dx if 'e' in edge else 0))
        new_height = max(280, height + (-dy if 'n' in edge else dy if 's' in edge else 0))
        if 'w' in edge:
            left += width - new_width
        if 'n' in edge:
            top += height - new_height
        position = f'+{left}+{top}'
        self._pending_geometry = f'{new_width}x{new_height}'
        if 'w' in edge or 'n' in edge:
            self._pending_geometry += position
        preview = getattr(self, '_resize_preview', None)
        if preview is not None:
            preview.geometry(f'{new_width}x{new_height}' + position)

    def flush_resize(self, event=None):
        if event is not None and getattr(self, '_resize_preview', None) is not None:
            self.resize_move(event)
        self.clear_resize_preview()
        geometry = getattr(self, '_pending_geometry', None)
        if geometry:
            self._pending_geometry = None
            self.root.geometry(geometry)

    def on_resize(self, event):
        if event.widget == self.root and event.width != getattr(self, '_status_width', None):
            self._status_width = event.width
            self.status_label.configure(wraplength=max(360, event.width - 34))

    def on_main_window_mapped(self, event):
        if event.widget is self.root:
            self.root.after_idle(self.enable_taskbar)

    def enable_taskbar(self, window=None):
        from window_effects import apply_shadow
        apply_shadow(window or self.root)
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
        desired = (style | 0x00040000) & ~0x00000080
        if style == desired:
            return
        # Explorer only re-evaluates a visible window's taskbar eligibility when
        # it is shown again. Changing the extended style alone is insufficient.
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        visible = user32.IsWindowVisible(hwnd)
        if visible:
            user32.ShowWindow(hwnd, 0)
        user32.SetWindowLongW(hwnd, -20, desired)
        if visible:
            user32.ShowWindow(hwnd, 5)

    def minimize(self):
        self.hide_to_tray()

    def toggle_settings(self):
        if self.settings_visible:
            self.settings_window.lift()
            return
        self.settings_visible = True
        self.settings_button.configure(bg=self.theme_color("#E6F2FB"), fg=self.theme_color("#737b8c"))
        window = self.settings_window
        window.update_idletasks()
        width, height = 480, 440
        x = max(0, min(self.root.winfo_rootx() + (self.root.winfo_width()-width)//2,
                       window.winfo_screenwidth()-width))
        y = max(0, min(self.root.winfo_rooty() + (self.root.winfo_height()-height)//2,
                       window.winfo_screenheight()-height-40))
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.deiconify()
        from window_effects import apply_shadow
        self.root.after_idle(lambda: apply_shadow(window))
        self.root.after_idle(self.apply_theme)
        window.lift()
        window.grab_set()
        self.settings_tabs.focus_set()

    def hide_settings(self, event=None):
        self.save_desktop_settings()
        self.settings_window.grab_release()
        self.settings_window.withdraw()
        self.settings_visible = False
        self.settings_button.configure(bg=self.theme_color("#f7f8fa"), fg=self.theme_color("#737b8c"))
        self.root.focus_set()
        return "break"

    def toggle_pin(self):
        self.pinned = not self.pinned
        self.root.attributes("-topmost", self.pinned)
        self.root.after_idle(self.enable_taskbar)
        self.pin_button.configure(text="取消置顶" if self.pinned else "置顶窗口")

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
            self.storage_hint.set("")
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
            save_preferences(DATA / "recognition-settings.json", self.hotwords.get())
        except OSError as exc:
            messagebox.showerror("无法保存热词", str(exc))
            return
        # Recording files have separate time origins; the visible conversation
        # and AI state persist across stop/start until explicitly cleared.
        try:
            folder = RECORDINGS
            folder.mkdir(exist_ok=True)
            self.session = folder / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
            self.session.touch()
        except OSError as exc:
            messagebox.showerror("无法保存记录", str(exc))
            return
        self.session_rows = []
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
        self.cancel_question_edit()
        self.qa.reset()
        self.qa_history.clear()
        self.qa_display_history.clear()
        self.question_drafts.clear()
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
        if not self.qa_enabled.get():
            self.auto_qa.set(False)
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
            self.codex_connection.check()
        else:
            self.columns.forget(self.qa_panel)
            self.qa_status.set("问答已关闭")
        self.splitter_handle.position()

    def toggle_auto_shortcut(self, event=None):
        if self.closing or self.root.grab_current():
            return 'break'
        self.open_qa()
        self.auto_qa.set(not self.auto_qa.get())
        self.toggle_auto_qa()
        return 'break'

    def session_context_snapshot(self):
        # Snapshot on the UI thread; compression runs on the request worker.
        if not self.auto_qa.get():
            return None
        return ([(row.get('source', 'system'), row['text']) for row in self.rows],
                [tuple(turn) for turn in self.qa_history])

    def toggle_auto_qa(self):
        # Reset pending work so turning Auto off cancels its in-flight response.
        self.qa.reset()
        self.qa_selection = None
        self.clear_question_selection()
        if self.auto_qa.get():
            # History provides context only; never submit old questions on enable.
            self.qa.context.extend(row['text'][-1200:] for row in self.rows[-8:])
            self.qa_status.set("自动识别后续问题…")
        else:
            self.qa_status.set("")

    def open_qa_settings(self):
        from qa_settings_dialog import show
        show(self)

    def configure_qa_provider(self):
        self.qa.reset()
        self.codex_connection.close()
        provider = self.qa_settings['provider']
        self.qa_provider_name.set(PROVIDERS[provider])
        self.qa.stream_runner = None
        if provider == 'codex':
            self.qa.runner = self.qa._run
            self.codex_connection = CodexConnection(self.events)
        else:
            profile = dict(DEFAULTS[provider], **self.qa_settings['profiles'].get(provider, {}))
            from reference_files import load_settings as load_references
            backend = APIProvider(provider, profile, workspace_loader=load_references)
            self.qa.stream_runner = backend.run
            self.codex_connection = CodexConnection(self.events, login=backend.preflight,
                probe=lambda cancel: backend.run('只回复 OK。', cancel, timeout=30, use_references=False),
                name=PROVIDERS[provider], failure=backend.failure)
        self.qa_cache.clear()
        self.qa_selection = None
        self.qa_question = self.qa_answer = ''
        self.render_qa()
        self.render_codex_connection()
        if self.qa.enabled:
            self.codex_connection.check()

    def check_codex_connection(self):
        if self.qa.active:
            self.codex_detail.set("正在回答，请完成后再检测。")
            return
        self.codex_connection.check(probe=True)

    def render_codex_connection(self):
        state = self.codex_connection.state
        labels = {"unknown": "尚未检测", "checking": "检测中…", "missing": "未安装",
                  "unauthenticated": "未登录", "authenticated": "已登录 · 待验证",
                  "verified": "响应正常", "unavailable": "连接异常"}
        if self.qa_settings['provider'] != 'codex':
            labels.update(authenticated="待验证", unauthenticated="未配置密钥")
        self.codex_status.set("● " + labels[state])
        self.codex_detail.set(self.codex_connection.detail)
        status_color = {"verified": "#15803D", "checking": "#2563EB",
                        "authenticated": "#B45309", "missing": "#B91C1C",
                        "unauthenticated": "#B45309", "unavailable": "#B91C1C"}.get(state, "#9297a4")
        self.codex_status_label._light_colors['foreground'] = status_color
        self.codex_status_label.configure(fg=self.theme_color(status_color))
        self.codex_check_button.configure(state="disabled" if state == "checking" else "normal")

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
        editor = getattr(self, 'question_editor', None)
        if editor is not None:
            revised = editor.get('1.0', 'end-1c').strip()
            if not revised or len(revised) > 2400:
                self.qa_status.set('请输入问题，最多 2400 字')
                editor.bell()
                return
            if not self.qa.enabled or self.codex_connection.state not in ('authenticated', 'verified'):
                self.qa_status.set('请先启用 AI 并检查模型连接')
                return
            indices = self.edit_question_indices
            turn_index = self.edit_question_turn
            self.cancel_question_edit()
            self.replace_question(turn_index, revised, indices)
            return
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
        if not self.qa.enabled or not question:
            return
        if self.codex_connection.state not in ("authenticated", "verified"):
            self.qa_status.set(self.codex_connection.detail)
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
        if self.auto_qa.get() and self.qa_question:
            self.qa_display_history.append((self.qa_question, self.qa_answer))
        self.qa_question, self.qa_answer = question, self.qa_cache.get(key, "")
        self.render_qa()
        if self.qa_answer:
            self.qa_status.set("")
            self.clear_question_selection()
            return
        self.qa_status.set("正在回答…")
        # Only the selected passage and its preceding context go to Codex.
        context = ([r["text"] for r in self.rows[max(0, indices[0]-8):indices[0]]]
                   + [self.rows[i]["text"] for i in indices]) if indices else [r["text"] for r in self.rows[-12:]]
        self.qa.ask(question, context)
        self.clear_question_selection()

    def edit_question(self, event):
        position = self.qa_text.index(f"@{event.x},{event.y}")
        turn = next((tag for tag in self.qa_text.tag_names(position)
                     if tag.startswith('question_turn:')), None)
        if turn is None:
            return
        turns = self.qa_display_history + ([(self.qa_question, self.qa_answer)] if self.qa_question else [])
        index = int(turn.split(':')[1])
        if index >= len(turns):
            return
        question = turns[index][0]
        if getattr(self, 'question_editor', None) is not None:
            if index == self.edit_question_turn:
                return 'break'
            # Resolve the clicked turn before reflowing the previous editor.
            self.cancel_question_edit(preserve=True)
        self.edit_question_turn = index
        self.edit_question_original = question
        self.edit_question_indices = list(self.qa_selection[0]) if index == len(self.qa_display_history) and self.qa_selection else []
        start, end = self.qa_text.tag_ranges(turn)
        editor = self.question_editor = tk.Text(self.qa_text, wrap='word', undo=True,
            font=(typography.UI_FAMILY, self.font_size.get()), bg=self.qa_text.cget('bg'),
            fg=self.theme_color('#007ACC'), insertbackground=self.theme_color('#263044'),
            relief='flat', bd=0, highlightthickness=0, padx=0, pady=4, height=3)
        editor.insert('1.0', self.question_drafts.get((index, question), question))
        self.qa_text.configure(state='normal')
        self.qa_text.delete(start, end)
        self.qa_text.window_create(start, window=editor, stretch=True)
        self.qa_text.insert(f'{start}+1c', '\n')
        self.qa_text.configure(state='disabled')
        self.ask_selected_button.configure(text='发送修改后的问题')
        def resize(event=None):
            if not editor.winfo_exists():
                return
            pixels = max(80, self.qa_text.winfo_width()-48)
            average = max(1, self.font_size.get() * .75)
            editor.configure(width=max(8, int(pixels/average)))
            editor.update_idletasks()
            count = editor.count('1.0', 'end-1c', 'displaylines')
            editor.configure(height=min(10, max(2, (count[0] if count else 0)+1)))
        editor.bind('<KeyRelease>', resize)
        editor.bind('<Control-Return>', lambda e: (self.ask_selected(), 'break')[1])
        editor.bind('<Escape>', lambda e: (self.cancel_question_edit(), 'break')[1])
        resize()
        editor.focus_set()
        editor.mark_set('insert', 'end-1c')
        return 'break'

    def replace_question(self, index, question, indices):
        turns = self.qa_display_history + [(self.qa_question, self.qa_answer)]
        if not 0 <= index < len(turns):
            return
        old = turns[index]
        for i in range(len(self.qa_history)-1, -1, -1):
            if tuple(self.qa_history[i]) == old:
                self.qa_history[i] = (question, '')
                break
        self.qa_cache.clear()
        self.qa_selection = None
        if index < len(self.qa_display_history):
            self.qa_display_history[index] = (question, '')
        else:
            self.qa_question, self.qa_answer = question, ''
        context = ([r['text'] for r in self.rows[max(0, indices[0]-8):indices[0]]]
                   + [self.rows[i]['text'] for i in indices]) if indices else [r['text'] for r in self.rows[-12:]]
        self.qa.ask(question, context)
        self.qa_replacement = (self.qa.generation, index, old)
        self.clear_question_selection()
        self.qa_status.set('正在回答…')
        self.render_qa()

    def cancel_question_edit(self, preserve=False):
        editor = getattr(self, 'question_editor', None)
        if editor is not None:
            key = (self.edit_question_turn, self.edit_question_original)
            if preserve:
                self.question_drafts[key] = editor.get('1.0', 'end-1c')
                if len(self.question_drafts) > 64:
                    self.question_drafts.pop(next(iter(self.question_drafts)))
            else:
                self.question_drafts.pop(key, None)
            self.question_editor = None
            editor.destroy()
            self.ask_selected_button.configure(text='发送所选')
            self.render_qa()

    def render_qa(self, streaming=False):
        editor = getattr(self, 'question_editor', None)
        editor_focused = editor is not None and self.root.focus_get() is editor
        has_answer = bool(self.qa_answer or self.qa_display_history)
        self.answer_copy_button.configure(state="normal" if has_answer else "disabled")
        if has_answer:
            self.answer_copy_button.pack(side="right")
        else:
            self.answer_copy_button.pack_forget()
        if self.qa_question or self.qa_display_history:
            self.qa_placeholder.place_forget()
        else:
            self.qa_placeholder.place(relx=0.5, rely=0.42, anchor="center")
        self.qa_text.configure(state="normal")
        previous = self.qa_text.get("1.0", "end-1c")
        turns = self.qa_display_history + ([(self.qa_question, self.qa_answer)] if self.qa_question else [])
        current = "\n\n".join(question + "\n" + answer for question, answer in turns)
        if editor is not None and not 0 <= self.edit_question_turn < len(turns):
            # A provider/conversation reset can remove the turn being edited.
            editor.destroy()
            self.question_editor = editor = None
            editor_focused = False
            self.ask_selected_button.configure(text='发送所选')
        follow = editor is None and self.qa_text.yview()[1] >= 1.0 - 1e-6
        # Preserve the visible text line, not a percentage of a growing document.
        # A fixed fraction moves the viewport down as new answers are appended.
        top_line = self.qa_text.index('@0,0')
        if editor is None and streaming and previous and current.startswith(previous):
            self.qa_text.insert("end", current[len(previous):])
        else:
            if editor is not None:
                # Detach before deleting text: otherwise Tk destroys the embedded
                # editor, including its draft, cursor, selection and undo history.
                self.qa_text.window_configure(str(editor), window='')
            self.qa_text.delete("1.0", "end")
            for index, (question, answer) in enumerate(turns):
                if index:
                    self.qa_text.insert("end", "\n\n")
                if editor is not None and index == self.edit_question_turn:
                    self.qa_text.window_create('end', window=editor, stretch=True)
                    self.qa_text.insert('end', '\n')
                else:
                    self.qa_text.insert("end", question + "\n", ("question", f"question_turn:{index}"))
                self.qa_text.insert("end", answer)
        if follow:
            self.qa_text.see('end')
        else:
            self.qa_text.yview(top_line)
        self.qa_text.configure(state="disabled")
        if editor_focused:
            editor.focus_set()

    def copy_answer(self):
        if self.qa_answer or self.qa_display_history:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.qa_text.get("1.0", "end-1c")
                                       if self.qa_display_history else self.qa_answer)
            self.answer_copy_button.configure(text="已复制")
            self.root.after(1500, lambda: self.answer_copy_button.configure(text="复制"))

    def handle_qa(self, value):
        generation, state, payload = value
        if generation != self.qa.generation or not self.qa.enabled or self.closing:
            return
        replacement = getattr(self, 'qa_replacement', None)
        if replacement and replacement[0] == generation and state in ('thinking', 'partial', 'answer'):
            if state == 'thinking':
                self.qa_status.set('正在回答…')
                return
            _, index, old = replacement
            if index < len(self.qa_display_history):
                self.qa_display_history[index] = tuple(payload)
            else:
                self.qa_question, self.qa_answer = payload
            self.render_qa()
            if state == 'answer':
                self.codex_connection.record()
                for i in range(len(self.qa_history)-1, -1, -1):
                    if tuple(self.qa_history[i]) == old or tuple(self.qa_history[i]) == (payload[0], ''):
                        self.qa_history[i] = payload
                        break
                self.qa_status.set('')
                if self.session:
                    path = self.session.with_suffix('.qa.jsonl')
                    try:
                        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
                        record = dict(time=datetime.now().astimezone().isoformat(), question=payload[0], answer=payload[1])
                        for i in range(len(records)-1, -1, -1):
                            if (records[i].get('question'), records[i].get('answer')) == old:
                                records[i] = record
                                break
                        else:
                            records.append(record)
                        temporary = path.with_suffix('.tmp')
                        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records), encoding='utf-8')
                        temporary.replace(path)
                    except (OSError, ValueError):
                        self.qa_status.set('回答已显示，但保存失败；可复制回答')
            return
        if state == "thinking":
            if self.auto_qa.get():
                self.qa_selection = None
                if self.qa_question and (self.qa_question != payload or self.qa_answer):
                    self.qa_display_history.append((self.qa_question, self.qa_answer))
                self.qa_question, self.qa_answer = payload, ""
                self.render_qa()
            self.qa_status.set("正在回答…")
        elif state == "partial":
            self.qa_question, self.qa_answer = payload
            self.render_qa(streaming=True)
        elif state == "answer":
            self.codex_connection.record()
            self.qa_question, self.qa_answer = payload
            if self.qa_selection is not None:
                self.qa_cache[self.qa_selection] = self.qa_answer
            self.qa_history.append(payload)
            self.render_qa(streaming=True)
            self.qa_status.set("")
            if self.session:
                try:
                    with self.session.with_suffix(".qa.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps({"time": datetime.now().astimezone().isoformat(),
                                               "question": payload[0], "answer": payload[1]}, ensure_ascii=False) + "\n")
                except OSError:
                    self.qa_status.set("回答已显示，但保存失败；可复制回答")
        elif state == "error":
            self.codex_connection.record(payload)
            if self.auto_qa.get():
                self.auto_qa.set(False)
                self.qa.reset()
            self.qa_status.set(payload[:220])
        elif state == "done":
            self.qa_replacement = None
            self.qa.active = False
            self.render_codex_connection()

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
            if kind == "desktop_callback":
                value()
            elif kind == "tray_show":
                self.root.deiconify()
                self.root.lift()
            elif kind == "tray_exit":
                self.close()
            elif kind == "status":
                self.status.set(value)
            elif kind == "qa":
                self.handle_qa(value)
            elif kind == "codex_connection":
                self.render_codex_connection()
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
                    self.hotkey_error.grid(row=1, column=0, sticky="w", pady=(8, 0))
            elif kind == "level":
                self.set_level(value)
            elif kind == "backlog":
                self.storage_hint.set(f"积压 {value:.0f}s" if value >= 2 else "")
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
                if self.auto_qa.get():
                    self.qa.feed(value["text"])
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
        if (self.auto_qa.get() and not self.closing
                and self.codex_connection.state in ("authenticated", "verified")):
            self.qa.tick()
        self.root.after(100, self.poll)

    def close(self):
        if self.closing:
            return
        self.clear_resize_preview()
        self.save_desktop_settings()
        if getattr(self, 'tray', None):
            self.tray.stop()
        self.closing = True
        self.qa.set_enabled(False)
        self.codex_connection.close()
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
    from app_paths import migrate_legacy
    from single_instance import SingleInstance
    instance = SingleInstance()
    if instance.duplicate:
        instance.close()
        raise SystemExit(0)
    migrate_legacy()
    app = App(tk.Tk())
    instance.poll(app.root)
    from desktop_dialogs import background_update_check
    app.root.after(8000, background_update_check, app)
    from model_download import cached_model
    if not preferences() and cached_model('small') is None:
        app.root.after(500, app.open_model_manager)
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
    try:
        app.root.mainloop()
    finally:
        instance.close()
