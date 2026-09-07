"""Small overlay scrollbar: rounded thumb, no arrows and no layout reflow."""
import tkinter as tk


class SlimScrollbar(tk.Canvas):
    def __init__(self, text):
        super().__init__(text, width=12, bg="white", bd=0, highlightthickness=0, cursor="arrow")
        self.target = text
        self.first, self.last = 0.0, 1.0
        self.hover = self.dragging = False
        self.thumb = self.create_line(6, 6, 6, 30, width=6, capstyle=tk.ROUND, fill="#cbd4de")
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
            self.place(relx=1, x=-2, y=0, anchor="ne", width=12, relheight=1, bordermode="ignore")
            tk.Misc.lift(self)
            self.draw()

    def metrics(self):
        track = max(1, self.winfo_height() - 12)
        visible = max(0, min(1, self.last-self.first))
        size = min(track, max(24, track*visible))
        travel = track-size
        top = 6 + travel * self.first / max(0.0001, 1-visible)
        return top, size, travel, visible

    def draw(self):
        top, size, _, _ = self.metrics()
        self.coords(self.thumb, 6, top+3, 6, max(top+3, top+size-3))
        self.itemconfigure(self.thumb, fill="#007ACC" if self.dragging else "#8bb7d6" if self.hover else "#cbd4de")

    def set_hover(self, hover):
        self.hover = hover
        self.draw()

    def press(self, event):
        top, size, travel, visible = self.metrics()
        if not top <= event.y <= top+size and travel > 0:
            value = (event.y-6-size/2)/travel*(1-visible)
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
