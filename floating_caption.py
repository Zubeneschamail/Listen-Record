"""Transparent, draggable, always-on-top captions for Windows."""
import tkinter as tk
from tkinter import font as tkfont
from difflib import SequenceMatcher


class CaptionPages:
    """Two stable lines per page; never replay consumed pages on final correction."""
    def __init__(self, measure, width):
        self.measure, self.width = measure, width
        self.text, self.offset, self.draft = "", 0, False

    def update(self, text, draft):
        if text != self.text:
            if not self.draft:
                self.offset = 0
            elif self.offset:
                # Map the consumed boundary through insertions/deletions in a revised draft.
                for tag, a, b, c, d in SequenceMatcher(None, self.text, text, autojunk=False).get_opcodes():
                    if a <= self.offset <= b:
                        self.offset = c + self.offset - a if tag == 'equal' else d
                        break
            self.text = text
        self.draft = draft
        while True:
            lines, starts = [], []
            start = self.offset
            line = ""
            for index in range(self.offset, len(text)):
                char = text[index]
                if char == '\n' or (line and self.measure(line + char) > self.width):
                    lines.append(line)
                    starts.append(start)
                    start = index + 1 if char == '\n' else index
                    line = '' if char == '\n' else char
                else:
                    line += char
            if line:
                lines.append(line)
                starts.append(start)
            if len(lines) <= 2:
                return lines
            self.offset = starts[2]


class FloatingCaption(tk.Toplevel):
    TRANSPARENT = '#010203'

    def __init__(self, parent):
        super().__init__(parent)
        self.title('闻录 · 实时字幕')
        self.overrideredirect(True)
        self.configure(bg=self.TRANSPARENT)
        self.attributes('-transparentcolor', self.TRANSPARENT)
        self.attributes('-topmost', True)
        self.width = min(850, self.winfo_screenwidth() - 40)
        x = max(0, (self.winfo_screenwidth() - self.width) // 2)
        y = max(0, self.winfo_screenheight() - 220)
        self.geometry(f'{self.width}x100+{x}+{y}')
        self.canvas = tk.Canvas(self, bg=self.TRANSPARENT, highlightthickness=0, bd=0,
                                cursor='fleur')
        self.canvas.pack(fill='both', expand=True)
        self.close_button = tk.Button(self, text='×', command=self.destroy,
            font=('Microsoft YaHei UI', 16), fg='#007ACC', bg='#EAF4FC',
            activebackground='#DCECF8', activeforeground='#007ACC',
            relief='flat', bd=0, highlightthickness=0, cursor='hand2', takefocus=False)
        self.canvas.bind('<ButtonPress-1>', self._start_drag)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.bind('<Escape>', lambda event: self.destroy())
        self._hover_job = None
        self._drag_origin = None
        self.caption_font = tkfont.Font(family='Microsoft YaHei UI', size=21)
        line_height = self.caption_font.metrics('linespace') + 8
        self.geometry(f'{self.width}x{line_height * 2 + 36}')
        self.pages = CaptionPages(self.caption_font.measure, self.width - 64)
        self.line_items = []
        for line in range(2):
            items = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                items.append(self.canvas.create_text(32 + dx, 18 + line * line_height + dy,
                    text='', font=self.caption_font, anchor='nw', fill='#17233A', tags='caption'))
            self.line_items.append(items)
        self.set_text('等待转写…')
        self._poll_hover()

    def set_text(self, text, draft=False):
        lines = self.pages.update(text, draft)
        for index, items in enumerate(self.line_items):
            value = lines[index] if index < len(lines) else ''
            for item in items:
                if self.canvas.itemcget(item, 'text') != value:
                    self.canvas.itemconfigure(item, text=value)
            self.canvas.itemconfigure(items[-1], fill='#A7D9FA' if draft else '#FFFFFF')

    def _start_drag(self, event):
        self._drag_origin = (event.x_root, event.y_root, self.winfo_x(), self.winfo_y())

    def _drag(self, event):
        if self._drag_origin:
            px, py, x, y = self._drag_origin
            self.geometry(f'{event.x_root - px + x:+d}{event.y_root - py + y:+d}')

    def _poll_hover(self):
        # Transparent pixels pass mouse events through, so use pointer bounds.
        x, y = self.winfo_pointerxy()
        inside = (self.winfo_x() <= x < self.winfo_x() + self.winfo_width() and
                  self.winfo_y() <= y < self.winfo_y() + self.winfo_height())
        if inside:
            self.close_button.place(relx=1, x=-2, y=0, anchor='ne', width=28, height=28)
        else:
            self.close_button.place_forget()
        self._hover_job = self.after(100, self._poll_hover)

    def destroy(self):
        if self._hover_job is not None:
            self.after_cancel(self._hover_job)
            self._hover_job = None
        super().destroy()
