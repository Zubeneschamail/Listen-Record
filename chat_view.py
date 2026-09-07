"""Selectable, source-labelled chat bubbles for the transcript."""
import tkinter as tk
import tkinter.font as tkfont
from scrollbars import SlimScrollbar

SURFACE = "#ffffff"


class Bubble(tk.Frame):
    def __init__(self, parent, row, click, selected_text):
        super().__init__(parent, bg=SURFACE)
        self.row, self.click, self.selected_text = row, click, selected_text
        self.mine = row.get("source") == "microphone"
        self.normal = "#007ACC" if self.mine else "#EDF4FA"
        self.label = tk.Label(self, text=row.get("time", ""),
                              bg=SURFACE, fg="#a3adba", font=("Microsoft YaHei UI", 8))
        self.canvas = tk.Canvas(self, bg=SURFACE, bd=0, highlightthickness=0)
        self.text = tk.Text(self.canvas, wrap="char", font=("Microsoft YaHei UI", 10),
                            bg=self.normal, fg="white" if self.mine else "#263044", bd=0, highlightthickness=0,
                            padx=0, pady=0, spacing2=4, cursor="xterm", exportselection=False,
                            selectborderwidth=0, selectbackground="#006BB3" if self.mine else "#DCECF8")
        self.text.insert("1.0", row["text"])
        self.text.configure(state="disabled")
        self.font = tkfont.Font(root=self, font=self.text.cget("font"))
        self.layout_key = None
        self.text_id = self.canvas.create_window(14, 10, anchor="nw", window=self.text)
        self.selected = False
        self.text.bind("<ButtonPress-1>", self.press)
        self.text.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.draft = row.get("draft", False)
        if self.draft:
            self.text.configure(fg="#BBDCF3" if self.mine else "#7896AD")

    def press(self, event):
        self.origin = (event.x, event.y)
        if event.state & 4:
            return "break"

    def release(self, event):
        if self.draft:
            return
        if event.state & 4:
            self.click(self.row["id"], True)
            return "break"
        ranges = self.text.tag_ranges("sel")
        if ranges:
            self.selected_text(self.row["id"], self.text.get(*ranges))
            return
        x, y = getattr(self, "origin", (event.x, event.y))
        if abs(x-event.x)+abs(y-event.y) <= 5:
            self.click(self.row["id"], False)

    def layout(self, width):
        key = (width, self.row["text"], self.selected, self.draft)
        if key == self.layout_key:
            return
        self.layout_key = key
        font = self.font
        limit = max(60, int((width-30)*.86)-28)
        content = max(28, min(limit, max((font.measure(line) for line in self.row["text"].splitlines()), default=28)))
        line_count = 0
        for line in self.row["text"].split("\n"):
            used, count = 0, 1
            for char in line:
                advance = font.measure(char)
                if used+advance > content and used:
                    count += 1
                    used = 0
                used += advance
            line_count += count
        height = line_count*(font.metrics("linespace")+4)+2
        bubble_width, bubble_height = content+28, height+20
        x = width-bubble_width-12 if self.mine else 12
        self.canvas.place(x=x, y=8, width=bubble_width, height=bubble_height)
        self.label.place(x=12, y=bubble_height+11, width=width-24, height=16)
        self.label.configure(anchor="e" if self.mine else "w")
        self.canvas.itemconfigure(self.text_id, width=content, height=height)
        self.canvas.delete("shape")
        fill = ("#006BB3" if self.mine else "#DCECF8") if self.selected else self.normal
        w, h, r = bubble_width, bubble_height, 9

        def rounded(inset, radius, color):
            # Straight sides and true quarter-circle corners, without spline bulges.
            x0, y0, x1, y1 = inset, inset, w-inset, h-inset
            self.canvas.create_rectangle(x0+radius, y0, x1-radius, y1,
                                         fill=color, outline="", tags="shape")
            self.canvas.create_rectangle(x0, y0+radius, x1, y1-radius,
                                         fill=color, outline="", tags="shape")
            for ax, ay, start in ((x0,y0,90), (x1-2*radius,y0,0),
                                   (x0,y1-2*radius,180), (x1-2*radius,y1-2*radius,270)):
                self.canvas.create_arc(ax, ay, ax+2*radius, ay+2*radius,
                                       start=start, extent=90, fill=color, outline="", tags="shape")

        rounded(0, r, fill)
        self.canvas.tag_lower("shape")
        foreground = ("#BBDCF3" if self.mine else "#7896AD") if self.draft else ("white" if self.mine else "#263044")
        self.text.configure(bg=fill, fg=foreground, selectforeground=foreground)
        self.configure(height=bubble_height+32)

    def set_selected(self, selected):
        if self.selected == selected:
            return
        self.selected = selected
        self.layout(self.winfo_width())


class ChatView(tk.Frame):
    def __init__(self, parent, click, selected_text):
        super().__init__(parent, bg=SURFACE)
        self.click, self.selected_text = click, selected_text
        self.bubbles = {}
        self.canvas = tk.Canvas(self, bg=SURFACE, bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=SURFACE)
        self.inner_id = self.canvas.create_window(0, 0, anchor="nw", window=self.inner)
        self.scrollbar = SlimScrollbar(self.canvas)
        self.scrollbar.configure(bg=SURFACE)
        self.canvas.bind("<Configure>", self.resize)
        self.inner.bind("<Configure>", self.update_scroll_region)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.pending_follow = None

    def update_scroll_region(self, event=None):
        if not self.bubbles:
            # An empty packed frame retains its previous requested height.
            self.inner.configure(height=1)
            self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), 1))
            self.canvas.yview_moveto(0)
            self.scrollbar.set(0, 1)
        else:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize(self, event=None):
        width = max(120, self.canvas.winfo_width()-14)
        self.canvas.itemconfigure(self.inner_id, width=width)
        for bubble in self.bubbles.values():
            bubble.layout(width)

    def wheel(self, event):
        self.canvas.yview_scroll((-1 if event.delta > 0 else 1)*max(1, abs(event.delta)//40), "units")
        return "break"

    def add(self, key, row):
        follow = self.canvas.yview()[1] >= .995
        old = self.bubbles.get(key)
        if old:
            # Keep a provisional bubble at the same position while revising it.
            old.row = row
            old.draft = row.get("draft", False)
            old.label.configure(text=row.get("time", ""))
            old.text.configure(state="normal")
            old.text.delete("1.0", "end")
            old.text.insert("end", row["text"])
            old.text.configure(state="disabled")
        else:
            old = Bubble(self.inner, row, self.click, self.selected_text)
            self.bubbles[key] = old
            old.pack(fill="x")
            for widget in (old, old.label, old.canvas, old.text):
                widget.bind("<MouseWheel>", self.wheel)
        self.resize()
        if follow:
            if self.pending_follow:
                self.winfo_toplevel().after_cancel(self.pending_follow)
            self.pending_follow = self.winfo_toplevel().after_idle(self.follow)
        return old

    def follow(self):
        self.pending_follow = None
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1)

    def finalize(self, source, row):
        draft_key = "draft:"+source
        bubble = self.bubbles.pop(draft_key, None)
        if bubble:
            self.bubbles[row["id"]] = bubble
        self.add(row["id"], row)

    def draft(self, source, text):
        key = "draft:"+source
        if text:
            self.add(key, {"id": key, "source": source, "text": text, "draft": True})
        elif key in self.bubbles:
            self.bubbles.pop(key).destroy()
            self.update_scroll_region()

    def clear(self):
        if self.pending_follow:
            self.winfo_toplevel().after_cancel(self.pending_follow)
            self.pending_follow = None
        for bubble in self.bubbles.values():
            bubble.destroy()
        self.bubbles.clear()
        self.update_scroll_region()

    def highlight(self, indices):
        for key, bubble in self.bubbles.items():
            bubble.set_selected(key in indices)
            bubble.text.tag_remove("sel", "1.0", "end")
