"""Small overlay scrollbar: rounded thumb, no arrows and no layout reflow."""
import tkinter as tk


class SlimScrollbar(tk.Canvas):
    def __init__(self, text):
        super().__init__(text, width=12, bg="white", bd=0, highlightthickness=0, cursor="arrow")
        self.target = text
        self.first, self.last = 0.0, 1.0
        self.hover = self.dragging = False
        self.track = self.create_line(6, 10, 6, 30, width=2, capstyle=tk.ROUND,
                                      fill="#f1f5f9", state="hidden")
        self.thumb = self.create_line(6, 10, 6, 38, width=4, capstyle=tk.ROUND, fill="#dce3eb")
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
        track = max(1, self.winfo_height() - 20)
        visible = max(0, min(1, self.last-self.first))
        size = min(track, max(28, track*visible))
        travel = track-size
        top = 10 + travel * self.first / max(0.0001, 1-visible)
        return top, size, travel, visible

    def draw(self):
        top, size, _, _ = self.metrics()
        active = self.hover or self.dragging
        radius = 3 if active else 2
        self.coords(self.track, 6, 10, 6, max(10, self.winfo_height()-10))
        self.itemconfigure(self.track, state="normal" if active else "hidden")
        self.coords(self.thumb, 6, top+radius, 6, max(top+radius, top+size-radius))
        self.itemconfigure(self.thumb, width=radius*2,
                           fill="#007ACC" if self.dragging else "#9dc3df" if self.hover else "#dce3eb")

    def set_hover(self, hover):
        self.hover = hover
        self.draw()

    def press(self, event):
        top, size, travel, visible = self.metrics()
        if not top <= event.y <= top+size and travel > 0:
            value = (event.y-10-size/2)/travel*(1-visible)
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
