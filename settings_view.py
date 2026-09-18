"""Settings composition; business actions remain owned by App."""
import tkinter as tk
from tkinter import ttk
from ui_components import UIControls, SettingsTabs, ScrollPage
from app_paths import DATA, RECORDINGS
from recognition import read_preferences


def build_settings(app, button):
    ui = UIControls(app)
    window = app.settings_window = tk.Toplevel(app.root)
    window.withdraw()
    window.title('闻录 · 设置')
    window.transient(app.root)
    window.overrideredirect(True)
    window.resizable(False, False)
    window.configure(bg='#f7f8fa')
    window.protocol('WM_DELETE_WINDOW', app.hide_settings)
    window.bind('<Escape>', app.hide_settings)
    panel = app.settings_panel = ttk.Frame(window, padding=12)
    panel.pack(fill='both', expand=True)
    titlebar = ttk.Frame(panel)
    titlebar.pack(fill='x', pady=(0, 8))
    title = ttk.Label(titlebar, text='设置', foreground='#4f586b')
    title.pack(side='left')
    button(titlebar, '×', app.hide_settings).pack(side='right')
    origin = [None]
    def begin(event):
        origin[0] = (event.x_root-window.winfo_rootx(), event.y_root-window.winfo_rooty())
    def drag(event):
        if origin[0] is not None:
            window.geometry(f'+{event.x_root-origin[0][0]}+{event.y_root-origin[0][1]}')
    for widget in (titlebar, title):
        widget.bind('<ButtonPress-1>', begin)
        widget.bind('<B1-Motion>', drag)
    tabs = app.settings_tabs = SettingsTabs(panel)
    tabs.pack(fill='both', expand=True)
    pages = app.settings_pages = {}
    for key, title in (('audio', '音频与转录'), ('ai', 'AI 与外观'), ('references', '参考资料'), ('tools', '工具与数据')):
        page = ScrollPage(tabs)
        tabs.add(page, text=title, padding=0)
        pages[key] = page
    audio, ai, tools = (pages[k].content for k in ('audio', 'ai', 'tools'))
    from reference_settings import build as build_references
    build_references(app, pages['references'].content)
    def scroll(event):
        if not isinstance(event.widget, ttk.Combobox):
            return tabs.pages[tabs.select()][0].wheel(event)
    window.bind('<MouseWheel>', scroll)

    app.capture_mode = tk.StringVar(value='系统声音 + 麦克风')
    field = ui.row(audio, '采集方式')
    app.mode = ui.combo(field, textvariable=app.capture_mode,
        values=['系统声音 + 麦克风', '仅系统声音', '仅麦克风'])
    app.mode.pack(fill='x')
    field = ui.row(audio, '麦克风')
    app.microphone = ui.combo(field)
    app.microphone.pack(fill='x')
    field = ui.row(audio, '系统声音')
    app.device = ui.combo(field)
    app.device.pack(fill='x')
    field = ui.row(audio, '转录模型')
    app.model = ui.combo(field, values=['small · 轻量准确', 'large-v3-turbo · 高性能', 'base · 均衡', 'tiny · 更快'])
    app.model.current(0)
    app.model_label = tk.StringVar(value=app.model.get())
    app.model.configure(textvariable=app.model_label)
    app.model_manage_button = button(field, '', lambda: run(app.open_model_manager))
    app.model_manage_button.configure(textvariable=app.model_label, anchor='w', bg='white', fg='#4f586b')
    app.model_manage_button.pack(fill='x')
    field = ui.row(audio, '识别语言')
    app.language = ui.combo(field, values=['中文', '自动检测', '英语'])
    app.language.current(0)
    app.language.pack(fill='x')
    field = ui.row(audio, '热词 / 纠错')
    app.hotwords = tk.StringVar(value=read_preferences(DATA / 'recognition-settings.json'))
    app.hotwords_entry = ttk.Entry(field, textvariable=app.hotwords, style='Settings.TEntry')
    app.hotwords_entry.pack(fill='x')
    ui.note(audio, '用逗号分隔；纠错示例：大模形=大模型').pack(anchor='w', padx=(82, 0), pady=4)

    field = ui.row(ai, '模型服务')
    button(field, '配置服务', app.open_qa_settings).pack(side='right')
    ttk.Label(field, textvariable=app.qa_provider_name, foreground='#007ACC').pack(side='left', pady=6)
    field = ui.row(ai, '连接状态')
    app.codex_check_button = button(field, '检查连接', app.check_codex_connection)
    app.codex_check_button.pack(side='right')
    ttk.Label(field, textvariable=app.codex_status).pack(side='left', pady=6)
    ui.note(ai, variable=app.codex_detail).pack(anchor='w', pady=(2, 12))
    ui.note(ai, 'AI 会发送相关转录文字并消耗所选服务额度。').pack(anchor='w', pady=(0, 18))
    ttk.Separator(ai).pack(fill='x', pady=(0, 12))
    field = ui.row(ai, '正文字号')
    app.font_size_picker = ui.combo(field, textvariable=app.font_size, values=list(range(8, 17)), width=6)
    app.font_size_picker.pack(side='left')
    app.font_size_picker.bind('<<ComboboxSelected>>', app.change_font_size)
    field = ui.row(ai, '界面主题')
    ttk.Checkbutton(field, text='深色模式', variable=app.dark_mode, command=app.change_theme).pack(side='left')
    app.pin_button = ui.action(ai, '窗口置顶', '让闻录保持在其他窗口上方', app.toggle_pin, '置顶窗口')

    def run(action):
        app.hide_settings()
        action()

    app.settings_actions = {}
    for title, description, action, caption in (
        ('导出记录', '保存当前文字为 TXT 或 SRT 字幕', app.export, '导出'),
        ('悬浮字幕', '以独立悬浮窗口显示实时字幕', app.open_caption, '打开'),
        ('AI 回答', '展开主窗口的 AI 问答区域', app.open_qa, '查看'),
        ('历史记录', '打开本机保存的转录文件夹', lambda: app.open_folder(RECORDINGS), '打开'),
        ('迁移旧版数据', '导入旧版本的模型与转录记录', app.import_legacy_data, '迁移'),
        ('最小化到托盘', '保持后台运行，隐藏主窗口', app.hide_to_tray, '最小化'),
        ('关于与更新', '查看版本并检查更新', app.open_about, '查看')):
        app.settings_actions[title] = ui.action(tools, title, description, lambda action=action: run(action), caption)
    footer = ttk.Frame(panel)
    footer.pack(side='bottom', fill='x', pady=(8, 0))
    footer.columnconfigure(0, weight=1)
    ui.note(footer, '设置自动保存').grid(row=0, column=0, sticky='w')
    button(footer, '完成', app.hide_settings, primary=True).grid(row=0, column=1, sticky='e')
    app.hotkey_status = tk.StringVar(value='')
    app.hotkey_error = ttk.Label(footer, textvariable=app.hotkey_status, foreground='#bd544f', wraplength=440)
