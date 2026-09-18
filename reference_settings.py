"""Reference material controls embedded in the shared settings viewport."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

import typography
from reference_files import load_settings, save_settings, validate_source
from ui_components import UIControls, PopupMenu
from scrollbars import SlimScrollbar


class ReferencePathItem(tk.Label):
    """Compact attachment paths with a full-path hover preview."""
    def __init__(self, parent, status, remove):
        super().__init__(parent, width=1, bg='white', fg='#9297a4',
                         font=(typography.UI_FAMILY, 8), anchor='w')
        self.paths = []
        self.status = status
        self.tip = None
        self.timer = None
        self.close_button = tk.Button(self, text='×', command=lambda: remove(self.paths[0]),
            font=(typography.UI_FAMILY, 10), bd=0, relief='flat', padx=0, pady=0,
            bg='white', fg='#737b8c', activebackground='#E6F2FB', takefocus=False,
            cursor='hand2')
        self.close_button.bind('<Enter>', self.hide_tip)
        self.close_button.bind('<Leave>', self.leave)
        self.status.trace_add('write', self.refresh)
        self.bind('<Configure>', self.refresh)
        self.bind('<Enter>', self.schedule_tip)
        self.bind('<Leave>', self.leave)
        self.bind('<ButtonPress-1>', self.hide_tip)
        self.bind('<Destroy>', self.hide_tip)

    def set_paths(self, paths):
        self.hide_tip()
        self.paths = list(paths)
        self.refresh()

    def refresh(self, *_):
        text = self.status.get() or ('；'.join(self.paths))
        font = tkfont.Font(font=self.cget('font'))
        width = max(0, min(200, self.winfo_width() - 24))
        if font.measure(text) > width:
            # Preserve both the drive/directory prefix and the final filename.
            left, right = len(text) // 2, len(text) - len(text) // 2
            while left + right and font.measure(text[:left] + '...' + (text[-right:] if right else '')) > width:
                if left >= right:
                    left -= 1
                else:
                    right -= 1
            text = text[:left] + '...' + (text[-right:] if right else '')
        self.configure(text=text)
        if self.close_button.winfo_manager():
            self.position_close()

    def position_close(self):
        font = tkfont.Font(font=self.cget('font'))
        x = min(font.measure(self.cget('text')) + 4, max(0, self.winfo_width() - 20))
        self.close_button.place(x=x, rely=.5, y=-2, anchor='w', width=18, height=18)

    def schedule_tip(self, event=None):
        self.hide_tip()
        if self.paths:
            self.position_close()
        if self.paths or self.status.get():
            self.timer = self.after(400, self.show_tip)

    def leave(self, event=None):
        self.hide_tip()
        self.after_idle(self.hide_close_if_outside)

    def hide_close_if_outside(self):
        if not self.winfo_exists():
            return
        x, y = self.winfo_pointerxy()
        if not (self.winfo_rootx() <= x < self.winfo_rootx() + self.winfo_width()
                and self.winfo_rooty() <= y < self.winfo_rooty() + self.winfo_height()):
            self.close_button.place_forget()

    def show_tip(self):
        self.timer = None
        self.tip = tk.Toplevel(self)
        self.tip.overrideredirect(True)
        self.tip.attributes('-topmost', True)
        text = '\n'.join(self.paths)
        if self.status.get():
            text = self.status.get() + ('\n' + text if text else '')
        tk.Label(self.tip, text=text, justify='left', wraplength=520, bg='#263e52',
                 fg='white', font=(typography.UI_FAMILY, 9), padx=9, pady=6).pack()
        self.tip.update_idletasks()
        x = max(0, min(self.winfo_rootx(), self.winfo_screenwidth() - self.tip.winfo_reqwidth()))
        y = max(0, self.winfo_rooty() - self.tip.winfo_reqheight() - 6)
        self.tip.geometry(f'+{x}+{y}')

    def hide_tip(self, event=None):
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ReferencePathLabel(tk.Frame):
    """Each source has its own hover target and removal action."""
    _custom_theme = True

    def __init__(self, parent, status, remove):
        super().__init__(parent, bg='white', height=22, width=1)
        self.pack_propagate(False)
        self.status, self.remove = status, remove
        self.paths, self.items = [], []
        self.dark = False
        self.status_label = tk.Label(self, anchor='w', bg='white', fg='#9297a4',
                                     font=(typography.UI_FAMILY, 8))
        self.status.trace_add('write', self.refresh)
        self.bind('<Configure>', self.refresh)

    def set_paths(self, paths):
        for item in self.items:
            item.destroy()
        self.paths, self.items = list(paths), []
        for path in self.paths:
            item = ReferencePathItem(self, tk.StringVar(master=self, value=''), self.remove)
            item.set_paths([path])
            self.items.append(item)
        self.apply_theme(self.dark)
        self.refresh()

    def refresh(self, *_):
        self.status_label.pack_forget()
        for item in self.items:
            item.place_forget()
        if self.status.get():
            self.status_label.configure(text=self.status.get())
            self.status_label.pack(fill='both', expand=True)
        else:
            available = max(1, self.winfo_width())
            share = max(1, available // max(1, len(self.items)) - 4)
            x = 0
            for item in self.items:
                font = tkfont.Font(font=item.cget('font'))
                width = min(224, font.measure(item.paths[0]) + 24, share)
                item.place(x=x, y=0, width=width, relheight=1)
                x += width + 4

    def apply_theme(self, dark):
        from theme import color
        self.dark = dark
        surface = color('white', dark)
        self.configure(bg=surface)
        self.status_label.configure(bg=surface, fg=color('#9297a4', dark))
        for item in self.items:
            item.configure(bg=surface, fg=color('#9297a4', dark))
            item.close_button.configure(bg=surface, fg=color('#737b8c', dark),
                                        activebackground=color('#E6F2FB', dark),
                                        activeforeground=color('#4f586b', dark))


def build(app, parent):
    ui = UIControls(app)
    settings = load_settings()
    enabled = tk.BooleanVar(value=settings['enabled'])
    heading = ttk.Frame(parent)
    heading.pack(fill='x', pady=(0, 8))
    count = tk.StringVar()
    notice = tk.StringVar()
    selected_path = tk.StringVar(value='添加与你讨论的问题相关的文档或项目目录。')
    ttk.Checkbutton(heading, text='回答时查阅参考资料', variable=enabled,
                    command=lambda: commit(list(settings['paths']))).pack(side='left')
    ui.note(heading, variable=count).pack(side='right')
    listing = tk.Listbox(parent, height=9, bg='white', fg='#4f586b',
                        selectbackground='#E6F2FB', selectforeground='#263044',
                        font=(typography.UI_FAMILY, 9), relief='flat', bd=0,
                        highlightthickness=1, highlightbackground='#e2e5ed',
                        activestyle='none', exportselection=False, selectmode='extended')
    listing.pack(fill='x', pady=(0, 8))
    SlimScrollbar(listing)
    ui.note(parent, variable=selected_path).pack(fill='x', pady=(0, 10))
    actions = ttk.Frame(parent)
    actions.pack(fill='x')

    def dialog_parent():
        return app.settings_window if app.settings_window.winfo_viewable() else app.root

    def refresh():
        listing.delete(0, 'end')
        for index, value in enumerate(settings['paths']):
            path = Path(value)
            prefix = '目录' if path.is_dir() else '文件'
            missing = ' · 不可用' if not path.exists() else ''
            listing.insert('end', f'  r{index+1} · {prefix}  {path.name}{missing}')
        count.set(f"{len(settings['paths'])} 项资料")
        if hasattr(app, 'reference_path_label'):
            app.reference_path_label.set_paths(settings['paths'])
        provider_changed()

    def provider_changed(*_):
        notice.set('DeepSeek 会按需搜索和读取，并在回答中标注文件来源。'
                   if app.qa_settings['provider'] == 'deepseek' else
                   '当前仅 DeepSeek 支持查阅资料，请在「AI 与外观」切换模型服务。')

    def commit(paths):
        updated = {'enabled': enabled.get(), 'paths': paths}
        try:
            save_settings(updated)
        except OSError:
            enabled.set(settings['enabled'])
            messagebox.showerror('保存失败', '无法保存参考资料设置，请检查磁盘权限或空间。', parent=dialog_parent())
            return
        settings.update(updated)
        app.qa.reset()
        app.qa_cache.clear()
        app.qa_selection = None
        app.qa_status.set('')
        refresh()

    def add(values):
        if not values:
            return
        paths = list(settings['paths'])
        errors = []
        for value in values:
            try:
                value = validate_source(value)
                if value.casefold() not in {p.casefold() for p in paths}:
                    if len(paths) >= 32:
                        raise ValueError('最多添加 32 项资料；可用文件夹统一管理。')
                    paths.append(value)
            except (OSError, ValueError) as exc:
                errors.append(f'{Path(value).name}：{str(exc) if isinstance(exc, ValueError) else "路径不可访问"}')
        if paths != settings['paths']:
            enabled.set(True)
            commit(paths)
        if errors:
            messagebox.showwarning('部分资料未添加', '\n'.join(errors[:5]), parent=dialog_parent())

    def add_files():
        add(filedialog.askopenfilenames(parent=dialog_parent(), title='添加参考文件',
            filetypes=[('文档与文本', '*.txt *.md *.pdf *.docx *.csv *.json'), ('所有文件', '*.*')]))

    def add_folder():
        folder = filedialog.askdirectory(parent=dialog_parent(), title='添加参考文件夹')
        if folder:
            add([folder])

    def remove():
        selected = set(listing.curselection())
        if selected:
            commit([p for i, p in enumerate(settings['paths']) if i not in selected])
            selected_path.set('已移除引用，原文件保留。')

    def remove_path(path):
        if path in settings['paths']:
            commit([value for value in settings['paths'] if value != path])
            selected_path.set('已移除引用，原文件保留。')

    def selection(event=None):
        indices = listing.curselection()
        selected_path.set(settings['paths'][indices[0]] if len(indices) == 1 else
                          f'已选 {len(indices)} 项资料' if indices else '选择资料可查看完整路径。')

    ui.button(actions, '添加文件', add_files, primary=True).pack(side='left')
    ui.button(actions, '添加文件夹', add_folder).pack(side='left', padx=8)
    ui.button(actions, '移除', remove).pack(side='right')
    listing.bind('<<ListboxSelect>>', selection)
    ui.note(parent, variable=notice).pack(fill='x', pady=(16, 6))
    ui.note(parent, '支持文本、代码、PDF、DOCX。只读访问；相关片段会发送给 DeepSeek。\n'
            '自动跳过依赖、构建目录和常见密钥文件。扫描版 PDF 请先做 OCR；\n'
            'PDF / DOCX 最大 20 MB，PDF 最多 100 页，文本最大 2 MB。').pack(fill='x')
    app.qa_provider_name.trace_add('write', provider_changed)
    # Named controls support UI verification without touching personal files.
    app.reference_controls = dict(listing=listing, enabled=enabled, add=add, remove=remove,
                                  add_files=add_files, add_folder=add_folder,
                                  paths=lambda: list(settings['paths']), remove_path=remove_path)
    refresh()


def show_add_menu(app):
    """Quick attachment picker, using the same actions as settings."""
    app.reference_add_menu.show(app.reference_add_button)


def build_add_button(app, parent):
    button = app.reference_add_button = UIControls(app).button(
        parent, '添加资料', lambda: show_add_menu(app))
    button.pack(side='left')
    menu = app.reference_add_menu = PopupMenu(app.root)
    menu.add_command(label='文件', command=app.reference_controls['add_files'])
    menu.add_command(label='文件夹', command=app.reference_controls['add_folder'])
