"""Small overlay scrollbar: rounded thumb, no arrows and no layout reflow."""
import tkinter as tk


class SlimScrollbar(tk.Canvas):
    _custom_theme = True
    _scrollbar_overlay = True

    def __init__(self, text, inset=0, overlay_parent=None):
        super().__init__(overlay_parent if overlay_parent is not None else text,
                         width=14, bg=text.cget('background'), bd=0,
                         highlightthickness=0, cursor="arrow")
        self.target = text
        self.inset = inset
        self.dark = getattr(self.winfo_toplevel(), '_dark_theme', False)
        self.first, self.last = 0.0, 1.0
        self.hover = self.dragging = False
        self.thumb = self.create_line(7, 8, 7, 44, width=3, capstyle=tk.ROUND, fill="#CDD9E5")
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Enter>", lambda e: self.set_hover(True))
        self.bind("<Leave>", lambda e: self.set_hover(False))
        self.bind("<ButtonPress-1>", self.press)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.release)
        self.bind("<MouseWheel>", self.wheel)
        text.configure(yscrollcommand=self.set)

    def set(self, first, last):
        self.first, self.last = float(first), float(last)
        if self.last - self.first >= 0.999:
            self.place_forget()
        else:
            self.place(in_=self.target, relx=1, x=-self.inset, y=0, anchor="ne", width=14, relheight=1, bordermode="ignore")
            tk.Misc.lift(self)
            self.draw()

    def metrics(self):
        track = max(1, self.winfo_height() - 16)
        visible = max(0, min(1, self.last-self.first))
        size = min(track, max(36, track*visible))
        travel = track-size
        top = 8 + travel * self.first / max(0.0001, 1-visible)
        return top, size, travel, visible

    def draw(self):
        top, size, _, _ = self.metrics()
        active = self.hover or self.dragging
        radius = 2.5 if active else 1.5
        self.coords(self.thumb, 7, top+radius, 7, max(top+radius, top+size-radius))
        resting = "#46586C" if self.dark else "#CDD9E5"
        hovered = "#759BB8" if self.dark else "#8FB8D6"
        self.itemconfigure(self.thumb, width=radius*2,
                           fill="#007ACC" if self.dragging else hovered if self.hover else resting)

    def apply_theme(self, dark):
        self.dark = dark
        self.configure(bg=self.target.cget('background'))
        self.draw()

    def set_hover(self, hover):
        self.hover = hover
        self.draw()

    def press(self, event):
        top, size, travel, visible = self.metrics()
        if not top <= event.y <= top+size and travel > 0:
            value = (event.y-8-size/2)/travel*(1-visible)
            self.target.yview_moveto(max(0, min(1-visible, value)))
            self.first, self.last = self.target.yview()
        self.dragging = True
        self.drag_y, self.drag_first = event.y, self.first
        self.draw()
        return "break"

    def drag(self, event):
        if not self.dragging:
            return "break"
        _, _, travel, visible = self.metrics()
        if travel > 0:
            value = self.drag_first + (event.y-self.drag_y)/travel*(1-visible)
            self.target.yview_moveto(max(0, min(1-visible, value)))
        return "break"

    def release(self, event):
        self.dragging = False
        self.draw()
        return "break"

    def wheel(self, event):
        if event.delta:
            self.target.yview_scroll((-1 if event.delta > 0 else 1)*max(1, abs(event.delta)//40), "units")
        return "break"
