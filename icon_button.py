"""Image buttons with retained text labels and delayed hover/focus descriptions."""
from pathlib import Path
import math
import tkinter as tk
import typography
from PIL import Image, ImageDraw, ImageTk

ICONS = Path(__file__).resolve().parent / 'assets' / 'icons'


class IconButton(tk.Button):
    def __init__(self, parent, icon, **kwargs):
        self.icon_name = icon
        self.images = {name: tk.PhotoImage(master=parent, file=str(ICONS / f'{name}.png'))
                       for name in (icon, 'check')}
        original = Image.open(ICONS / f'{icon}.png').convert('RGBA')
        self.neutral_images = {}
        if icon in ('settings', 'copy', 'close', 'minimize', 'more', 'add', 'collapse'):
            for tone in ('#737b8c', '#AEBBCD'):
                variants = {}
                for name in (icon, 'check'):
                    source = Image.open(ICONS / f'{name}.png').convert('RGBA')
                    tinted = Image.new('RGBA', source.size, tone)
                    tinted.putalpha(source.getchannel('A'))
                    variants[name] = ImageTk.PhotoImage(tinted, master=parent)
                self.neutral_images[tone.lower()] = variants
            self.images.update(self.neutral_images['#737b8c'])
        if icon in ('microphone', 'waveform'):
            self.active_source = Image.new('RGBA', original.size, '#007ACC')
            self.active_source.putalpha(original.getchannel('A'))
            self.images['active'] = ImageTk.PhotoImage(self.active_source, master=parent)
        muted = Image.new('RGBA', original.size, '#728299')
        muted.putalpha(original.getchannel('A'))
        self.disabled_image = ImageTk.PhotoImage(muted, master=parent)
        self.tooltip_window = None
        self.tooltip_timer = None
        self.animation_timer = None
        self.animation_frames = []
        self.animation_frame = 0
        super().__init__(parent, **kwargs)
        super().configure(image=self.images[icon], compound='none', width=30, height=28, padx=0, pady=0,
                          disabledforeground='')
        self.bind('<Enter>', self.schedule_tooltip, add='+')
        self.bind('<FocusIn>', self.schedule_tooltip, add='+')
        self.bind('<Leave>', self.hide_tooltip, add='+')
        self.bind('<FocusOut>', self.hide_tooltip, add='+')
        self.bind('<ButtonPress-1>', self.hide_tooltip, add='+')
        self.bind('<Destroy>', self.hide_tooltip, add='+')
        self.bind('<Destroy>', self.stop_animation, add='+')

    def configure(self, cnf=None, **kwargs):
        # Tk stipples the entire image rectangle when disabledforeground is set,
        # including transparent pixels. Render our own muted icon instead.
        if 'disabledforeground' in kwargs:
            kwargs['disabledforeground'] = ''
        result = super().configure(cnf, **kwargs)
        tone = str(self.cget('foreground')).lower()
        if self.neutral_images and ('foreground' in kwargs or 'fg' in kwargs):
            self.images.update(self.neutral_images.get(tone, self.neutral_images['#737b8c']))
        if any(key in kwargs for key in ('text', 'state', 'fg', 'foreground')):
            name = ('active' if self.icon_name in ('microphone', 'waveform') and self.cget('text') == '停止转写'
                    else 'check' if self.cget('text') == '已复制' else self.icon_name)
            finishing = self.icon_name == 'waveform' and self.cget('text') == '收尾中…'
            image = (self.disabled_image if str(self.cget('state')) == 'disabled' and not finishing else
                     self.images[name])
            super().configure(image=image)
            if name == 'active' and str(self.cget('state')) != 'disabled':
                if self.animation_timer is None:
                    self.animate_recording()
            else:
                self.stop_animation()
            self.hide_tooltip()
        return result

    config = configure

    def animate_recording(self):
        if not self.animation_frames:
            for i in range(48):
                # Render once at 4x resolution for smooth rounded blue bars.
                # Phase offsets simulate a travelling voice waveform, not movement
                # of the button itself. No audio capture or asset files are needed.
                phase = 2 * math.pi * i / 48
                frame = Image.new('RGBA', (96, 96))
                draw = ImageDraw.Draw(frame)
                for bar, x in enumerate((3, 7.5, 12, 16.5, 21)):
                    pulse = (1 + math.sin(phase + bar * 1.1)) / 2
                    height = 5 + (9 + 4 * (1 - abs(bar - 2) / 2)) * pulse
                    shade = int(50 * (1 - pulse))
                    fill = (shade, 122 + shade, 204 + int(shade * .6), 255)
                    draw.rounded_rectangle(((x - 1) * 4, (12 - height / 2) * 4,
                                            (x + 1) * 4, (12 + height / 2) * 4),
                                           radius=4, fill=fill)
                frame = frame.resize((24, 24), Image.Resampling.LANCZOS)
                self.animation_frames.append(ImageTk.PhotoImage(frame, master=self))
        # Keep the same 24px canvas and centered bars throughout the cycle.
        super().configure(image=self.animation_frames[self.animation_frame])
        self.animation_frame = (self.animation_frame + 1) % len(self.animation_frames)
        self.animation_timer = self.winfo_toplevel().after(40, self.animate_recording)

    def stop_animation(self, event=None):
        if event is not None and event.widget is not self:
            return
        if self.animation_timer is not None:
            self.winfo_toplevel().after_cancel(self.animation_timer)
            self.animation_timer = None
        self.animation_frame = 0

    def schedule_tooltip(self, event=None):
        self.hide_tooltip()
        self.tooltip_timer = self.winfo_toplevel().after(500, self.show_tooltip)

    def show_tooltip(self):
        self.tooltip_timer = None
        if not self.winfo_viewable():
            return
        tip = self.tooltip_window = tk.Toplevel(self)
        tip.overrideredirect(True)
        tip.attributes('-topmost', True)
        description = {'···': '更多操作', '×': '关闭', '—': '最小化'}.get(self.cget('text'), self.cget('text'))
        tk.Label(tip, text=description, bg='#263e52', fg='white', font=(typography.UI_FAMILY,9),
                 padx=9, pady=5).pack()
        tip.update_idletasks()
        x = max(0, min(self.winfo_rootx(), tip.winfo_screenwidth()-tip.winfo_reqwidth()))
        y = self.winfo_rooty()+self.winfo_height()+6
        if y+tip.winfo_reqheight() > tip.winfo_screenheight():
            y = self.winfo_rooty()-tip.winfo_reqheight()-6
        tip.geometry(f'+{x}+{max(0,y)}')

    def hide_tooltip(self, event=None):
        if self.tooltip_timer:
            self.winfo_toplevel().after_cancel(self.tooltip_timer)
            self.tooltip_timer = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class IconToggle(IconButton):
    """A themed on/off icon using the same hover description as other buttons."""
    _custom_theme = True

    def __init__(self, parent, icon, text, variable, command, surface='white'):
        self.variable, self.on_change = variable, command
        self.surface = surface
        self.dark, self.hover = False, False
        super().__init__(parent, icon, text=text, command=self.toggle,
                         relief='flat', bd=0, cursor='hand2', takefocus=True)
        source = Image.open(ICONS / f'{icon}.png').convert('RGBA')
        self.toggle_images = {}
        for tone in ('#737b8c', '#AEBBCD', '#007ACC', '#60A5FA'):
            tinted = Image.new('RGBA', source.size, tone)
            tinted.putalpha(source.getchannel('A'))
            self.toggle_images[tone] = ImageTk.PhotoImage(tinted, master=self)
        self.trace = variable.trace_add('write', lambda *_: self.draw_toggle())
        self.bind('<Enter>', lambda e: self.set_hover(True), add='+')
        self.bind('<Leave>', lambda e: self.set_hover(False), add='+')
        self.bind('<Return>', lambda e: self.invoke(), add='+')
        self.bind('<Destroy>', self.cleanup_toggle, add='+')
        self.apply_theme(False)

    def toggle(self):
        self.variable.set(not self.variable.get())
        self.on_change()

    def set_hover(self, value):
        self.hover = value
        self.draw_toggle()

    def apply_theme(self, dark):
        self.dark = dark
        self.draw_toggle()

    def draw_toggle(self):
        from theme import color
        active = self.variable.get()
        tone = ('#60A5FA' if self.dark else '#007ACC') if active else ('#AEBBCD' if self.dark else '#737b8c')
        tk.Button.configure(self, image=self.toggle_images[tone], bg=color('#EDF5FB' if self.hover else self.surface, self.dark),
                            activebackground=color('#E6F2FB', self.dark), highlightcolor=color('#007ACC', self.dark))

    def cleanup_toggle(self, event):
        if event.widget is self:
            self.variable.trace_remove('write', self.trace)
