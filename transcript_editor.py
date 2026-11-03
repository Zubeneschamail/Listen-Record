"""A local correction editor; saving never sends a new model request."""
import tkinter as tk
from tkinter import ttk
import typography


def place_editor(window, parent, width, height):
    from window_effects import work_area
    try:
        left, top, right, bottom = work_area(parent)
    except (AttributeError, OSError):
        left, top, right, bottom = 0, 0, parent.winfo_screenwidth(), parent.winfo_screenheight()
    scale = window.winfo_fpixels('1i')/96
    side, caption = round(24*scale), round(48*scale)
    x = parent.winfo_rootx()+(parent.winfo_width()-width)//2
    y = parent.winfo_rooty()+(parent.winfo_height()-height)//2
    x = max(left, min(x, right-width-side))
    y = max(top, min(y, bottom-height-caption))
    window.geometry(f'{width}x{height}+{x}+{y}')


class TranscriptEditor(tk.Toplevel):
    def __init__(self, app, row_id, original):
        self.app = app
        super().__init__(app.root)
        self.app, self.row_id = app, row_id
        self.expected = app.rows[row_id]['text']
        self.original = original
        app.transcript_editor = self
        self.title('修正转录')
        self.transient(app.root)
        self.geometry('480x350')
        self.minsize(380, 320)
        self.configure(padx=16, pady=14)
        ttk.Label(self, text='修正后的文字').pack(anchor='w')
        self.editor = tk.Text(self, wrap='word', height=7, undo=True,
                              font=(typography.UI_FAMILY, app.font_size.get()), bd=1, relief='solid')
        self.editor.insert('1.0', self.expected)
        self.editor.pack(fill='both', expand=True, pady=(8, 8))
        self.detail = tk.StringVar(value='保存后用于复制、导出和后续问答。原转录仍保留。')
        note = ttk.Label(self, textvariable=self.detail, wraplength=440)
        note.pack(fill='x')
        note.bind('<Configure>', lambda event: note.configure(wraplength=max(160, event.width)))
        self.remember = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text='保存本段后，另存常用错词规则', variable=self.remember).pack(
            anchor='w', pady=(8, 0))
        actions = ttk.Frame(self)
        actions.pack(fill='x', pady=(12, 0))
        ttk.Button(actions, text='恢复原转录', command=lambda: self.reset(original)).pack(side='left')
        ttk.Button(actions, text='保存', command=self.save).pack(side='right')
        ttk.Button(actions, text='取消', command=self.destroy).pack(side='right', padx=(0, 8))
        self.bind('<Escape>', lambda event: self.destroy())
        self.editor.bind('<Control-s>', lambda event: (self.save(), 'break')[1])
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.update_idletasks()
        place_editor(self, app.root, 480, 350)
        app.apply_theme()
        self.grab_set()
        self.editor.focus_set()

    def reset(self, text):
        self.editor.delete('1.0', 'end')
        self.editor.insert('1.0', text)
        self.editor.focus_set()

    def save(self):
        from ime_support import is_composing
        if is_composing(self.editor):
            self.detail.set('请先确认正在输入的文字，再保存。')
            return
        try:
            text = self.editor.get('1.0', 'end-1c')
            self.app.correct_transcript(self.row_id, text, self.expected)
        except (OSError, ValueError) as exc:
            self.detail.set('未保存：'+str(exc))
            return
        remember = self.remember.get()
        self.destroy()
        if remember:
            self.app.open_correction_rule_editor(self.original, text.strip())

    def destroy(self):
        if self.app.transcript_editor is self:
            self.app.transcript_editor = None
        if str(self.tk.call('grab', 'current')) == str(self):
            self.grab_release()
        super().destroy()


class CorrectionRuleEditor(tk.Toplevel):
    def __init__(self, app, original, corrected, preferences):
        super().__init__(app.root)
        self.app, self.preferences = app, preferences
        app.transcript_editor = self
        self.title('记住常用错词')
        self.transient(app.root)
        self.geometry('520x470')
        self.minsize(430, 430)
        self.configure(padx=16, pady=14)
        ttk.Label(self, text='本段修正已保存。请填写以后要替换的完整错词。', wraplength=470).pack(anchor='w')
        reference = ttk.Frame(self)
        reference.pack(fill='both', expand=True, pady=(10, 12))
        self.reference = tk.Text(reference, wrap='word', height=6, bd=1, relief='solid',
                                  font=(typography.UI_FAMILY, app.font_size.get()))
        scroll = ttk.Scrollbar(reference, command=self.reference.yview)
        scroll.pack(side='right', fill='y')
        self.reference.configure(yscrollcommand=scroll.set)
        self.reference.pack(side='left', fill='both', expand=True)
        self.reference.insert('1.0', '原转录：\n'+original+'\n\n本段修正：\n'+corrected)
        self.reference.configure(state='disabled')
        self.wrong, self.right = tk.StringVar(), tk.StringVar()
        fields = ttk.Frame(self)
        fields.pack(fill='x')
        fields.columnconfigure(1, weight=1)
        ttk.Label(fields, text='错词').grid(row=0, column=0, sticky='w', padx=(0, 12))
        self.wrong_entry = ttk.Entry(fields, textvariable=self.wrong)
        self.wrong_entry.grid(row=0, column=1, sticky='ew', pady=(0, 8))
        ttk.Label(fields, text='正确词').grid(row=1, column=0, sticky='w', padx=(0, 12))
        self.right_entry = ttk.Entry(fields, textvariable=self.right)
        self.right_entry.grid(row=1, column=1, sticky='ew')
        self.replace_existing = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text='替换此错词的已有规则（如有）', variable=self.replace_existing).pack(
            anchor='w', pady=(10, 8))
        self.detail = tk.StringVar(value='例如：密等性 → 幂等性。保存后用于后续识别，已有转录需单独修正。')
        note = ttk.Label(self, textvariable=self.detail, wraplength=470)
        note.pack(fill='x')
        note.bind('<Configure>', lambda event: note.configure(wraplength=max(160, event.width)))
        self.traces = [(variable, variable.trace_add('write', self.changed))
                       for variable in (self.wrong, self.right)]
        actions = ttk.Frame(self)
        actions.pack(fill='x', pady=(12, 0))
        ttk.Button(actions, text='保存规则', command=self.save).pack(side='right')
        ttk.Button(actions, text='取消', command=self.destroy).pack(side='right', padx=(0, 8))
        self.bind('<Escape>', lambda event: self.destroy())
        self.bind('<Control-s>', lambda event: (self.save(), 'break')[1])
        self.bind('<Return>', lambda event: (self.save(), 'break')[1])
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.update_idletasks()
        place_editor(self, app.root, 520, 470)
        app.apply_theme()
        self.grab_set()
        self.wrong_entry.focus_set()

    def changed(self, *args):
        self.replace_existing.set(False)
        wrong, right = self.wrong.get().strip(), self.right.get().strip()
        if len(wrong) == 1 and '\u3400' <= wrong <= '\u9fff':
            self.detail.set('单字规则会替换每次出现的这个字，建议填写完整错词，例如“密等性”。')
        elif wrong and right:
            self.detail.set(f'将保存：{wrong} → {right}。保存后用于后续识别。')
        else:
            self.detail.set('例如：密等性 → 幂等性。保存后用于后续识别，已有转录需单独修正。')

    def save(self):
        from ime_support import is_composing
        if is_composing(self.wrong_entry) or is_composing(self.right_entry):
            self.detail.set('请先确认正在输入的词语，再保存。')
            return
        try:
            self.app.remember_correction_rule(self.wrong.get(), self.right.get(),
                self.preferences.fingerprint, self.replace_existing.get())
        except (OSError, ValueError) as exc:
            self.detail.set('规则未保存：'+str(exc))
            return
        self.destroy()

    def destroy(self):
        for variable, token in getattr(self, 'traces', []):
            variable.trace_remove('write', token)
        self.traces = []
        if self.app.transcript_editor is self:
            self.app.transcript_editor = None
        if str(self.tk.call('grab', 'current')) == str(self):
            self.grab_release()
        super().destroy()
