"""Small themed context ring for the composer footer."""
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import theme
import typography
from context_usage import percentage, usage_label


class ContextMeter(tk.Canvas):
    _custom_theme = True

    def __init__(self, parent):
        super().__init__(parent, width=28, height=28, bd=0, highlightthickness=0, bg='white')
        self.report = None
        self.dark = False
        self.hovered = False
        self.bind('<Enter>', lambda event: self.set_hovered(True))
        self.bind('<Leave>', lambda event: self.set_hovered(False))
        self.bind('<Unmap>', lambda event: self.set_hovered(False))
        self.draw()

    def set_hovered(self, hovered):
        self.hovered = hovered
        self.configure(width=80 if hovered else 28)
        self.draw()

    def set_usage(self, report=None):
        self.report = report
        self.draw()

    def apply_theme(self, dark):
        self.dark = dark
        self.configure(bg=theme.color('white', dark))
        self.draw()

    def draw(self):
        self.delete('all')
        value = percentage(self.report)
        tone = '#b91c1c' if value is not None and value >= 90 else '#b45309' if value is not None and value >= 75 else '#737b8c'
        tone = theme.color(tone, self.dark)
        # Tk's native oval/arc primitives have no antialiasing. Supersample the
        # transparent ring, as we do for the other small toolbar icons.
        scale = 8
        ring = Image.new('RGBA', (18 * scale, 18 * scale))
        painter = ImageDraw.Draw(ring)
        bounds = (scale, scale, 17 * scale, 17 * scale)
        painter.ellipse(bounds, outline=theme.color('#e3e9f0', self.dark), width=2 * scale)
        if value is not None and value > 0:
            if value >= 100:
                painter.ellipse(bounds, outline=tone, width=2 * scale)
            else:
                painter.arc(bounds, start=-90, end=-90 + value * 3.6, fill=tone, width=2 * scale)
        self.ring_image = ImageTk.PhotoImage(ring.resize((18, 18), Image.Resampling.LANCZOS), master=self)
        # Right packing keeps the ring and send button fixed while the number
        # expands into the flexible reference-path area on the left.
        ring_x = 66 if self.hovered else 14
        self.create_image(ring_x, 14, image=self.ring_image)
        self.create_text(ring_x-15, 14, anchor='e', text=usage_label(self.report), fill=tone,
                         font=(typography.UI_FAMILY, 8), tags='percentage',
                         state='normal' if self.hovered else 'hidden')
