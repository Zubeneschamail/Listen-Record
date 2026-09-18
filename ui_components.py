"""Reusable themed controls shared by the main window and settings."""
import tkinter as tk
from tkinter import ttk
import typography
from icon_button import IconButton
from theme import color
from scrollbars import SlimScrollbar


class ToggleChip(tk.Canvas):
    """Compact rounded toggle shared by AI and Auto, including keyboard input."""
    _custom_theme = True

    def __init__(self, parent, text, variable, command, surface='white'):
        super().__init__(parent, width=54, height=26, bg=surface, bd=0,
                         highlightthickness=0, takefocus=True, cursor='hand2')
        self.text, self.variable, self.command = text, variable, command
        self.surface, self.dark, self.hover, self.focused = surface, False, False, False
        self.trace = variable.trace_add('write', lambda *_: self.draw())
        self.bind('<Enter>', lambda e: self.set_hover(True))
        self.bind('<Leave>', lambda e: self.set_hover(False))
        self.bind('<ButtonRelease-1>', self.click)
        self.bind('<space>', self.click)
        self.bind('<Return>', self.click)
        self.bind('<FocusIn>', lambda e: self.set_focus(True))
        self.bind('<FocusOut>', lambda e: self.set_focus(False))
        self.bind('<Destroy>', self.cleanup)
        self.draw()

    def set_hover(self, value):
        self.hover = value
        self.draw()

    def set_focus(self, value):
        self.focused = value
        self.draw()

    def click(self, event=None):
        if event is None or not hasattr(event, 'num') or event.num != 1 or (0 <= event.x < 54 and 0 <= event.y < 26):
            self.invoke()
        return 'break'

    def invoke(self):
        self.variable.set(not self.variable.get())
        self.command()

    def apply_theme(self, dark):
        self.dark = dark
        self.configure(bg=color(self.surface, dark))
        self.draw()

    def draw(self):
        self.delete('all')
        active = self.variable.get()
        fill = color('#E6F2FB' if active else '#f1f3f7', self.dark)
        if self.hover:
            fill = color('#dcecf8' if active else '#edf0f7', self.dark)
        ink = ('#60A5FA' if self.dark else '#007ACC') if active else color('#737b8c', self.dark)
        self.create_polygon(9, 1, 45, 1, 53, 1, 53, 9, 53, 17, 53, 25,
                            45, 25, 9, 25, 1, 25, 1, 17, 1, 9, 1, 1,
                            smooth=True, splinesteps=24, fill=fill,
                            outline=ink if self.focused else fill, width=1)
        self.create_text(27, 13, text=self.text, fill=ink,
                         font=(typography.UI_FAMILY, 9))

    def cleanup(self, event):
        if event.widget is self:
            self.variable.trace_remove('write', self.trace)


class PopupMenu(tk.Toplevel):
    """Small themed action surface with outside-click and keyboard dismissal."""
    _custom_theme = True

    def __init__(self, parent):
        super().__init__(parent)
        self.withdraw()
        self.overrideredirect(True)
        self.transient(parent)
        self.configure(bd=0, highlightthickness=1)
        self.rows = []
        self.commands = []
        self.dark = False
        self.bind('<Escape>', lambda e: self.hide())
        self.bind('<ButtonPress-1>', self.outside_click)
        self.bind('<FocusOut>', lambda e: self.after_idle(self.check_focus))

    def add_command(self, label, command):
        index = len(self.rows)
        row = tk.Button(self, text=label, anchor='w', command=lambda: self.invoke(index),
                        font=(typography.UI_FAMILY, 9), relief='flat', bd=0,
                        highlightthickness=0, padx=14, pady=8, cursor='hand2', takefocus=True)
        row.pack(fill='x', padx=5, pady=(5 if not index else 0, 5))
        row.bind('<Enter>', lambda e: self.highlight(index, True))
        row.bind('<Leave>', lambda e: self.highlight(index, False))
        row.bind('<FocusIn>', lambda e: self.highlight(index, True))
        row.bind('<FocusOut>', lambda e: self.highlight(index, False))
        row.bind('<Down>', lambda e: self.move(index, 1))
        row.bind('<Up>', lambda e: self.move(index, -1))
        row.bind('<Return>', lambda e: self.invoke(index))
        self.rows.append(row)
        self.commands.append(command)
        self.apply_theme(self.dark)

    def move(self, index, direction):
        self.rows[(index + direction) % len(self.rows)].focus_set()
        return 'break'

    def highlight(self, index, active):
        self.rows[index].configure(bg=color('#E6F2FB' if active else 'white', self.dark))

    def apply_theme(self, dark):
        self.dark = dark
        self.configure(bg=color('white', dark), highlightbackground=color('#e2e5ed', dark))
        for row in self.rows:
            row.configure(bg=color('white', dark), fg=color('#4f586b', dark),
                          activebackground=color('#E6F2FB', dark),
                          activeforeground=color('#4f586b', dark))

    def show(self, anchor):
        if self.winfo_viewable():
            self.hide()
            return
        self.update_idletasks()
        width, height = 144, self.winfo_reqheight()
        x = max(0, min(anchor.winfo_rootx(), self.winfo_screenwidth() - width))
        y = max(0, anchor.winfo_rooty() - height - 6)
        self.geometry(f'{width}x{height}+{x}+{y}')
        self.deiconify()
        self.lift()
        self.focus_set()
        self.grab_set()
        from window_effects import apply_shadow
        apply_shadow(self)

    def outside_click(self, event):
        if not (self.winfo_rootx() <= event.x_root < self.winfo_rootx() + self.winfo_width()
                and self.winfo_rooty() <= event.y_root < self.winfo_rooty() + self.winfo_height()):
            self.hide()

    def check_focus(self):
        if self.winfo_exists() and self.winfo_viewable():
            focused = self.focus_get()
            if focused is None or focused.winfo_toplevel() is not self:
                self.hide()

    def hide(self):
        if self.grab_current() is self:
            self.grab_release()
        self.withdraw()

    def invoke(self, index):
        self.hide()
        self.commands[index]()
        return 'break'


class SplitterHandle(tk.Canvas):
    """Wide pointer target over a one-pixel sash; commit layout on release."""
    _custom_theme = True

    def __init__(self, panes):
        super().__init__(panes, width=13, bd=0, highlightthickness=0,
                         bg='white', cursor='sb_h_double_arrow')
        self.panes, self.drag = panes, None
        self.line = self.create_line(6, 0, 6, 10000, fill='#E3E9F0')
        self.bind('<ButtonPress-1>', self.begin)
        self.bind('<B1-Motion>', self.move)
        self.bind('<ButtonRelease-1>', self.end)
        self.bind('<Escape>', self.cancel)
        panes.bind('<Configure>', self.position, add='+')

    def apply_theme(self, dark):
        self.configure(bg=color('white', dark))
        self.itemconfigure(self.line, fill=color('#E3E9F0', dark))

    def position(self, event=None):
        if len(self.panes.panes()) != 2:
            self.place_forget()
            return
        self.place(x=self.panes.sash_coord(0)[0]-6, y=0, width=13, relheight=1, bordermode='ignore')
        tk.Misc.lift(self)
        # Scrollbars share this parent so their entire hit area can stay above
        # the wider sash overlay, including after a resize or pane movement.
        for widget in self.panes.winfo_children():
            if getattr(widget, '_scrollbar_overlay', False) and widget.winfo_manager():
                tk.Misc.lift(widget)

    def begin(self, event):
        self.drag = (self.panes.sash_coord(0)[0], event.x_root)
        self.target = self.drag[0]
        self.focus_set()
        self.grab_set()
        return 'break'

    def move(self, event):
        if self.drag:
            panes = self.panes.panes()
            minimum = int(self.panes.panecget(panes[0], 'minsize'))
            maximum = self.panes.winfo_width() - int(self.panes.panecget(panes[1], 'minsize')) - 1
            self.target = max(minimum, min(maximum, self.drag[0] + event.x_root - self.drag[1]))
            self.panes.proxy_place(self.target, 0)
        return 'break'

    def end(self, event):
        if self.drag:
            self.move(event)
            self.panes.sash_place(0, self.target, 0)
            self.cancel()
            self.position()
        return 'break'

    def cancel(self, event=None):
        self.drag = None
        self.panes.proxy_forget()
        if self.grab_current() is self:
            self.grab_release()
        return 'break'


class ScrollPage(ttk.Frame):
    """A themed, independently scrolling settings page."""
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg='#f7f8fa', bd=0, highlightthickness=0,
                                width=480, height=1, yscrollincrement=18)
        self.canvas.pack(fill='both', expand=True)
        self.content = ttk.Frame(self.canvas, padding=(4, 8))
        self.item = self.canvas.create_window(0, 0, window=self.content, anchor='nw')
        self.scrollbar = SlimScrollbar(self.canvas)
        self.canvas.bind('<Configure>', self.resize)
        self.content.bind('<Configure>', self.update_region)

    def resize(self, event):
        self.canvas.itemconfigure(self.item, width=max(1, event.width - 14))
        self.update_region()

    def update_region(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def wheel(self, event):
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(-int(event.delta / 120) * 3, 'units')
        return 'break'


class SettingsTabs(ttk.Frame):
    """Flat navigation with a fixed viewport for scrollable pages."""
    def __init__(self, parent):
        super().__init__(parent)
        self.navigation = ttk.Frame(self)
        self.navigation.pack(fill='x', pady=(0, 10))
        self.pages = {}
        self.current = None

    def add(self, page, text, **unused):
        tab = tk.Button(self.navigation, text=text, command=lambda: self.select(page),
            font=(typography.UI_FAMILY, 9), relief='flat', bd=0, highlightthickness=0,
            padx=10, pady=7, cursor='hand2', bg='#f7f8fa', fg='#737b8c',
            activebackground='#E6F2FB', activeforeground='#007ACC')
        tab.pack(side='left', padx=(0, 6))
        self.pages[str(page)] = (page, tab)
        if self.current is None:
            self.select(page)

    def select(self, page=None):
        if page is None:
            return self.current
        key = str(page)
        if self.current is not None:
            self.pages[self.current][0].pack_forget()
        self.current = key
        self.pages[key][0].pack(fill='both', expand=True)
        dark = getattr(self.winfo_toplevel(), '_dark_theme', False)
        for candidate, (_, tab) in self.pages.items():
            bg, fg = ('#E6F2FB', '#007ACC') if candidate == key else ('#f7f8fa', '#737b8c')
            originals = getattr(tab, '_light_colors', {})
            originals.update(background=bg, foreground=fg)
            tab._light_colors = originals
            tab.configure(bg=color(bg, dark), fg=color(fg, dark))


class UIControls:
    def __init__(self, app):
        self.app = app

    def row(self, parent, label):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text=label, width=10).pack(side='left', padx=(0, 12))
        field = ttk.Frame(row)
        field.pack(side='left', fill='x', expand=True)
        return field

    def combo(self, parent, **options):
        options.setdefault('width', 24)
        return ttk.Combobox(parent, state='readonly', style='Settings.TCombobox',
                            font=(typography.UI_FAMILY, 9), **options)

    def note(self, parent, text=None, variable=None):
        return ttk.Label(parent, text=text, textvariable=variable, wraplength=400,
                         foreground='#858b98', font=(typography.UI_FAMILY, 8))

    def action(self, parent, title, description, command, caption='打开'):
        row = ttk.Frame(parent, padding=(0, 8))
        row.pack(fill='x')
        button = self.button(row, caption, command)
        button.pack(side='right', padx=(12, 0))
        ttk.Label(row, text=title, foreground='#4f586b').pack(anchor='w')
        self.note(row, description).pack(anchor='w', pady=(3, 0))
        return button

    def button(self, parent, text, command, primary=False, width=None):
        surface = parent.cget("bg") if isinstance(parent, tk.Frame) else "#f7f8fa"
        icons = {"设置": "settings", "复制全文": "copy", "复制": "copy", "···": "more",
                 "刷新": "refresh", "×": "close", "—": "minimize", "发送所选": "send",
                 "开始转写": "waveform", "停止转写": "waveform", "添加资料": "add"}
        factory = IconButton if text in icons else tk.Button
        widget = factory(parent, **({"icon": icons[text]} if text in icons else {}),
                         text=text, command=command, relief="flat", bd=0,
                         bg="#007ACC" if primary else surface,
                         fg="white" if primary else "#737b8c", activebackground="#006BB3" if primary else "#E6F2FB",
                         activeforeground="white" if primary else "#007ACC", disabledforeground="#a6acba",
                         font=(typography.UI_FAMILY, 9), padx=10, pady=5,
                         cursor="hand2", takefocus=True, **({"width": width} if width else {}))
        normal_bg = widget.cget("bg")
        widget.bind("<Enter>", lambda e: widget.configure(bg=self.app.theme_color("#006BB3" if primary else "#EDF5FB")) if str(widget.cget("state")) != "disabled" else None, add="+")
        widget.bind("<Leave>", lambda e: widget.configure(bg=self.app.theme_color(normal_bg)), add="+")
        return widget
