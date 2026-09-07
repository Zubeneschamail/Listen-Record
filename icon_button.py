"""Image buttons with retained text labels and delayed hover/focus descriptions."""
from pathlib import Path
import tkinter as tk

ICONS = Path(__file__).resolve().parent / 'assets' / 'icons'


class IconButton(tk.Button):
    def __init__(self, parent, icon, **kwargs):
        self.icon_name = icon
        self.images = {name: tk.PhotoImage(master=parent, file=str(ICONS / f'{name}.png'))
                       for name in (icon, 'check')}
        self.tooltip_window = None
        self.tooltip_timer = None
        super().__init__(parent, **kwargs)
        super().configure(image=self.images[icon], compound='none', width=30, height=28, padx=0, pady=0)
        self.bind('<Enter>', self.schedule_tooltip, add='+')
        self.bind('<FocusIn>', self.schedule_tooltip, add='+')
        self.bind('<Leave>', self.hide_tooltip, add='+')
        self.bind('<FocusOut>', self.hide_tooltip, add='+')
        self.bind('<ButtonPress-1>', self.hide_tooltip, add='+')
        self.bind('<Destroy>', self.hide_tooltip, add='+')

    def configure(self, cnf=None, **kwargs):
        result = super().configure(cnf, **kwargs)
        if 'text' in kwargs:
            super().configure(image=self.images['check' if kwargs['text'] == '已复制' else self.icon_name])
            self.hide_tooltip()
        return result

    config = configure

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
        tk.Label(tip, text=description, bg='#263e52', fg='white', font=('Microsoft YaHei UI',9),
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
