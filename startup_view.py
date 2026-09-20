"""Full-window startup surface matching Wenlu's existing light/dark palette."""
import tkinter as tk
import typography
from startup_checks import LABELS


class StartupOverlay(tk.Frame):
    def __init__(self, app):
        self.app = app
        bg = app.theme_color('#f7f8fa')
        super().__init__(app.root, bg=bg, bd=0)
        self.place(x=0, y=0, relwidth=1, relheight=1)
        self.lift()
        self.grab_set()
        self.focus_set()
        tk.Button(self, text='×', command=app.close, bg=bg, fg=app.theme_color('#858b98'),
                  relief='flat', bd=0, font=(typography.UI_FAMILY, 16), cursor='hand2').place(relx=1, x=-12, y=8, anchor='ne')
        card = tk.Frame(self, bg=bg)
        card.place(relx=.5, rely=.5, anchor='center', relwidth=.84)
        self.spinner = tk.Canvas(card, width=40, height=40, bg=bg, bd=0, highlightthickness=0)
        self.spinner.pack(pady=(0, 8))
        self.spinner.create_oval(5, 5, 35, 35, outline=app.theme_color('#E3E9F0'), width=3)
        self.arc = self.spinner.create_arc(5, 5, 35, 35, start=90, extent=90, style='arc', outline='#007ACC', width=3)
        tk.Label(card, text='正在准备闻录', bg=bg, fg=app.theme_color('#263044'),
                 font=(typography.UI_FAMILY, 13, 'bold')).pack(pady=(0, 14))
        self.rows = {}
        for key, label in LABELS.items():
            row = tk.Frame(card, bg=bg)
            row.pack(fill='x', pady=3)
            tk.Label(row, text=label, bg=bg, fg=app.theme_color('#4f586b'),
                     font=(typography.UI_FAMILY, 10)).pack(side='left')
            value = tk.Label(row, text='等待检测', bg=bg, fg=app.theme_color('#9299aa'),
                             font=(typography.UI_FAMILY, 9))
            value.pack(side='right')
            self.rows[key] = value
        self.detail = tk.Label(card, text='正在启动检查…', bg=bg, fg=app.theme_color('#858b98'),
                               font=(typography.UI_FAMILY, 9), wraplength=350, justify='center')
        self.detail.pack(fill='x', pady=(12, 0))
        card.bind('<Configure>', lambda e: self.detail.configure(wraplength=max(160, e.width-8)))
        self.angle, self.timer = 0, None
        self.animate()

    def animate(self):
        self.angle = (self.angle - 20) % 360
        self.spinner.itemconfigure(self.arc, start=self.angle)
        self.timer = self.after(65, self.animate)

    def render(self, results):
        text = {'waiting': '等待检测', 'checking': '检测中…', 'success': '正常', 'error': '需处理', 'blocked': '待模型就绪'}
        for key, (state, _) in results.items():
            self.rows[key].configure(text=text[state], fg=self.app.theme_color(
                '#007ACC' if state == 'checking' else '#15803d' if state == 'success' else '#b45309' if state in ('error','blocked') else '#9299aa'))
        pending = [detail for state, detail in results.values() if state == 'checking']
        self.detail.configure(text=(pending[0][:110] if pending else '检查完成'))

    def destroy(self):
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        if self.grab_current() is self:
            self.grab_release()
        super().destroy()
