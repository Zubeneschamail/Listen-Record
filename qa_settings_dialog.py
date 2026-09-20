"""Compact model settings; editing does not change the active service until saved."""
import copy
import tkinter as tk
from tkinter import ttk, messagebox
from desktop_dialogs import dialog
from qa_provider import DEFAULTS, PROVIDERS, endpoint, save_settings
from ui_components import guard_combo_wheel


def show(app, initial_provider=None, on_close=None, initial_error=None):
    existing = getattr(app, 'qa_settings_window', None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    window = dialog(app, '闻录 · 模型服务')
    app.qa_settings_window = window
    window.transient(app.settings_window)
    drafts = copy.deepcopy(app.qa_settings['profiles'])
    active = [initial_provider or app.qa_settings['provider']]
    provider = tk.StringVar(value=PROVIDERS[active[0]])
    base = tk.StringVar()
    model = tk.StringVar()
    key = tk.StringVar()
    help_text = tk.StringVar()
    ttk.Label(window, text='问答服务').grid(row=0, column=0, sticky='w', padx=(0, 16))
    selector = ttk.Combobox(window, textvariable=provider, values=list(PROVIDERS.values()), state='readonly', width=38)
    guard_combo_wheel(selector)
    selector.grid(row=0, column=1, sticky='ew')
    fields = ttk.Frame(window)
    fields.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(12, 0))
    fields.columnconfigure(1, weight=1)
    ttk.Label(fields, text='API 地址').grid(row=0, column=0, sticky='w', padx=(0, 16))
    address = ttk.Entry(fields, textvariable=base, width=42, style='Settings.TEntry')
    address.grid(row=0, column=1, sticky='ew')
    ttk.Label(fields, text='模型名称').grid(row=1, column=0, sticky='w', pady=10)
    models = ttk.Combobox(fields, textvariable=model, width=40)
    guard_combo_wheel(models)
    models.grid(row=1, column=1, sticky='ew', pady=10)
    ttk.Label(fields, text='API Key').grid(row=2, column=0, sticky='w')
    ttk.Entry(fields, textvariable=key, show='•', width=42, style='Settings.TEntry').grid(row=2, column=1, sticky='ew')
    ttk.Label(window, textvariable=help_text, wraplength=420, justify='left').grid(
        row=2, column=0, columnspan=2, sticky='w', pady=14)

    def stash():
        drafts[active[0]] = {'base_url': base.get().strip(), 'model': model.get().strip(), 'api_key': key.get().strip()}

    def populate():
        name = active[0]
        values = dict(DEFAULTS[name], **drafts.get(name, {}))
        base.set(DEFAULTS[name]['base_url'] if name == 'deepseek' else values['base_url'])
        model.set(values['model'])
        key.set(values.get('api_key', ''))
        models.configure(values=('deepseek-flash', 'deepseek-v4-pro') if name == 'deepseek' else ())
        address.configure(state='readonly' if name == 'deepseek' else 'normal')
        help_text.set(('Pro 图片答疑先由 Flash 读图，共计两次模型调用。\n' if name == 'deepseek' else
                       '支持 Chat Completions 接口，地址按服务商要求填写（如 /v1）。\n') +
                      'API Key 加密保存在本机，调用按服务商计费。')
        app.root.after_idle(app.apply_theme)

    def changed(event=None):
        stash()
        active[0] = next(name for name, label in PROVIDERS.items() if label == provider.get())
        populate()

    def close():
        key.set('')
        drafts.clear()
        window.grab_release()
        window.destroy()
        if app.settings_window.winfo_viewable():
            app.settings_window.grab_set()
        if on_close:
            app.root.after_idle(on_close)

    def save():
        stash()
        name = active[0]
        try:
            endpoint(drafts[name]['base_url'])
            if not drafts[name]['model'] or not drafts[name]['api_key']:
                raise ValueError('请填写模型名称和 API Key。')
            settings = {'provider': name, 'profiles': drafts}
            save_settings(settings)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror('无法保存', str(exc), parent=window)
            return
        app.qa_settings = copy.deepcopy(settings)
        app.configure_qa_provider()
        close()

    # Changing the destination must not silently reuse a credential for another host.
    def address_changed(*_):
        if active[0] == 'compatible':
            key.set('')
    base.trace_add('write', address_changed)
    selector.bind('<<ComboboxSelected>>', changed)
    actions = ttk.Frame(window)
    actions.grid(row=3, column=0, columnspan=2, sticky='e')
    ttk.Button(actions, text='取消', command=close).pack(side='left', padx=(0, 8))
    ttk.Button(actions, text='保存', command=save).pack(side='left')
    window.protocol('WM_DELETE_WINDOW', close)
    populate()
    if initial_error:
        help_text.set('启动检测未通过：' + initial_error[:300] + '\n请检查服务配置并保存，再重新检测。')
    window.update_idletasks()
    parent = app.settings_window
    x = max(0, parent.winfo_rootx() + (parent.winfo_width() - window.winfo_reqwidth()) // 2)
    y = max(0, parent.winfo_rooty() + (parent.winfo_height() - window.winfo_reqheight()) // 2)
    window.geometry(f'+{x}+{y}')
    app.root.after_idle(app.apply_theme)
    window.grab_set()
    return window
