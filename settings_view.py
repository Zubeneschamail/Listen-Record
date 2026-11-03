"""Settings composition; business actions remain owned by App."""
import tkinter as tk
from tkinter import ttk
from ui_components import UIControls, SettingsTabs, ScrollPage, StatusIndicator
from app_paths import DATA, RECORDINGS
from recognition import read_preferences, INTERVIEW_HOTWORDS, add_interview_hotwords


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
    window.bind('<ButtonPress-1>', app.raise_settings, add='+')
    window.bind('<FocusIn>', app.raise_settings, add='+')
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
    for key, title in (('audio', '转录'), ('ai', '答疑'), ('references', '资料'),
                       ('appearance', '外观'), ('tools', '数据')):
        page = ScrollPage(tabs)
        tabs.add(page, text=title, padding=0)
        pages[key] = page
    audio, ai, appearance, tools = (pages[k].content for k in ('audio', 'ai', 'appearance', 'tools'))
    from reference_settings import build as build_references
    build_references(app, pages['references'].content)
    def scroll(event):
        if isinstance(event.widget, (tk.Listbox, tk.Text)):
            return 'break'
        return tabs.pages[tabs.select()][0].wheel(event)
    window.bind('<MouseWheel>', scroll)

    def run(action):
        app.hide_settings()
        action()

    ui.section(audio, '声音来源')
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
    field = ui.row(audio, '回声消除')
    app.echo_checkbox = ttk.Checkbutton(field, text='外放回声消除', variable=app.echo_cancellation)
    app.echo_checkbox.pack(side='left')
    ui.note(audio, '仅系统声音 + 麦克风时生效；保存后，下次转录启用。使用耳机时建议关闭。').pack(
        anchor='w', pady=(4, 10))
    ui.section(audio, '识别设置')
    field = ui.row(audio, '转录模型')
    from qwen_engine import MODEL_LABELS, QWEN_MODEL, QWEN_HELP, default_model_label
    app.model = ui.combo(field, values=list(MODEL_LABELS))
    app.model.current(MODEL_LABELS.index(default_model_label()))
    app.model_label = tk.StringVar(value=app.model.get())
    app.model.configure(textvariable=app.model_label)
    app.model_manage_button = button(field, '', app.open_model_manager)
    app.model_manage_button.configure(textvariable=app.model_label, anchor='w', bg='white', fg='#4f586b')
    app.model_manage_button.pack(fill='x')
    field = ui.row(audio, '识别语言')
    app.language = ui.combo(field, values=['中文', '自动检测', '英语'])
    app.language.current(0)
    app.language.pack(fill='x')
    field = ui.row(audio, '热词 / 纠错')
    app.hotwords = tk.StringVar(value=read_preferences(DATA / 'recognition-settings.json'))
    presets = ttk.Menubutton(field, text='面试术语')
    app.hotwords_presets = tk.Menu(presets, tearoff=False)
    for topic in INTERVIEW_HOTWORDS:
        app.hotwords_presets.add_command(label=topic, command=lambda topic=topic:
            app.hotwords.set(add_interview_hotwords(app.hotwords.get(), topic)))
    presets.configure(menu=app.hotwords_presets)
    presets.pack(side='right', padx=(6, 0))
    app.hotwords_entry = ttk.Entry(field, textvariable=app.hotwords, style='Settings.TEntry')
    app.hotwords_entry.pack(side='left', fill='x', expand=True)
    app.recognition_help = tk.StringVar()
    def update_recognition_help(*args):
        app.recognition_help.set(QWEN_HELP if app.model.get().split()[0] == QWEN_MODEL else
            '普通词用于识别提示；密等性=幂等性只纠正输出。多个词用逗号分隔。')
    app.model_label.trace_add('write', update_recognition_help)
    update_recognition_help()
    ui.note(audio, variable=app.recognition_help).pack(anchor='w', padx=(92, 0), pady=4)

    ui.section(audio, '手动分段')
    ui.note(audio, variable=app.manual_cut_status).pack(anchor='w', pady=(2, 4))
    from manual_cut import MOUSE_OPTIONS
    field = ui.row(audio, '鼠标快捷键')
    app.cut_mouse_picker = ui.combo(field, textvariable=app.manual_cut_mouse, values=list(MOUSE_OPTIONS))
    app.cut_mouse_picker.pack(fill='x')
    ui.note(audio, variable=app.manual_mouse_status).pack(anchor='w', pady=(2, 4))
    ui.note(audio, '按下后定稿已采集内容，继续录音并另起一段；双音源一起分段。保存后生效。').pack(anchor='w', pady=(0, 8))

    ui.section(audio, '运行状态')
    field = ui.row(audio, 'GPU 加速')
    app.gpu_settings_row = field.master
    app.gpu_check_button = button(field, '检测', app.check_gpu_acceleration)
    app.gpu_check_button.pack(side='right')
    app.gpu_status_indicator = StatusIndicator(field, app.gpu_check_status, app.gpu_detection_state)
    app.gpu_status_indicator.pack(side='left', pady=6)
    ui.note(audio, variable=app.gpu_runtime_status).pack(anchor='w', pady=(2, 4))
    ui.note(audio, variable=app.gpu_check_detail).pack(anchor='w', pady=(0, 10))

    ui.section(ai, '答疑模型')
    field = ui.row(ai, '模型服务')
    button(field, '配置', app.open_qa_settings).pack(side='right')
    ttk.Label(field, textvariable=app.qa_provider_name, foreground='#007ACC').pack(side='left', pady=6)
    field = ui.row(ai, '连接状态')
    app.connection_check_button = button(field, '检查连接', app.check_qa_connection)
    app.connection_check_button.pack(side='right')
    app.connection_status_indicator = StatusIndicator(field, app.connection_status, app.qa_detection_state)
    app.connection_status_indicator.pack(side='left', pady=6)
    ui.note(ai, variable=app.connection_detail).pack(anchor='w', pady=(2, 12))
    field = ui.row(ai, '账户余额')
    app.balance_check_button = button(field, '查询', app.check_qa_balance)
    app.balance_check_button.pack(side='right')
    ttk.Label(field, textvariable=app.balance_status, foreground='#007ACC').pack(side='left', pady=6)
    balance_area = ttk.Frame(ai)
    balance_area.pack(fill='x', pady=(0, 12))
    ui.note(balance_area, variable=app.balance_detail).pack(anchor='w', pady=(2, 0))
    ui.section(ai, '自动答疑')
    field = ui.row(ai, '自动答疑')
    app.settings_auto_qa = tk.BooleanVar(value=app.auto_qa.get())
    ttk.Checkbutton(field, text='自动读取转录问题和剪贴板', variable=app.settings_auto_qa).pack(side='left')
    from qa_worker import QUESTION_SOURCES
    field = ui.row(ai, '面试官音源')
    app.question_source = ui.combo(field, values=list(QUESTION_SOURCES))
    app.question_source.current(0)
    app.question_source.pack(fill='x')
    ui.note(ai, '自动时优先系统声音；仅采集麦克风时使用麦克风。另一方的声音仅作上下文。').pack(anchor='w', pady=(2, 6))
    ui.note(ai, variable=app.clipboard_status).pack(anchor='w', pady=(2, 6))
    ui.note(ai, '内容会发送给所选模型服务并按服务商计费；重启后自动答疑关闭。').pack(anchor='w', pady=(0, 12))
    ui.section(ai, '屏幕提问')
    ttk.Checkbutton(ai, text='启用 Ctrl+K 截屏提问', variable=app.screen_capture_enabled).pack(anchor='w', pady=5)
    ui.note(ai, variable=app.screen_capture_status).pack(anchor='w', pady=(2, 4))
    ui.note(ai, '保存后全局接管 Ctrl+K；按下时截取前台窗口所在屏幕，发送给当前模型。无需开启自动答疑。').pack(anchor='w', pady=(0, 12))

    ui.section(appearance, '显示')
    field = ui.row(appearance, '正文字号')
    app.font_size_picker = ui.combo(field, textvariable=app.font_size, values=list(range(8, 17)), width=6)
    app.font_size_picker.pack(side='left')
    app.font_size_picker.bind('<<ComboboxSelected>>', app.change_font_size)
    field = ui.row(appearance, '界面主题')
    ttk.Checkbutton(field, text='深色模式', variable=app.dark_mode, command=app.change_theme).pack(side='left')
    ui.section(appearance, '窗口')
    app.pin_button = ui.action(appearance, '窗口置顶', None, app.toggle_pin, '置顶窗口')
    ui.action(appearance, '悬浮字幕', '独立窗口显示实时转录', lambda: run(app.open_caption), '打开')
    ui.section(appearance, '屏幕共享')
    ttk.Checkbutton(appearance, text='共享时隐藏闻录窗口', variable=app.capture_hidden).pack(anchor='w', pady=5)
    ui.note(appearance, variable=app.capture_privacy.status).pack(anchor='w', pady=(2, 4))
    ui.note(appearance, '自己仍可见和操作；任务栏入口也会隐藏，可从系统托盘打开。保存后生效，请在共享预览中确认。').pack(anchor='w')

    app.settings_actions = {}
    ui.section(tools, '练习记录')
    for title, description, action, caption in (
        ('导出记录', '导出当前转录为 TXT / SRT', app.export, '导出'),
        ('保存问答副本', '保存已完成问答，可重新打开练习', app.save_qa_copy, '保存'),
        ('打开练习', '查看转录、问答及已保存的修正', app.open_recording, '打开'),
        ('历史记录', '打开本机转录文件夹', lambda: app.open_folder(RECORDINGS), '打开')):
        app.settings_actions[title] = ui.action(tools, title, description, lambda action=action: run(action), caption)
    ui.section(tools, '应用')
    app.settings_actions['关于与更新'] = ui.action(tools, '关于与更新', None, lambda: run(app.open_about), '查看')
    footer = ttk.Frame(panel)
    footer.pack(side='bottom', fill='x', pady=(8, 0))
    footer.columnconfigure(0, weight=1)
    app.settings_save_button = button(footer, '保存', app.save_settings_dialog, primary=True)
    app.settings_save_button.grid(row=0, column=1, sticky='e')
    app.hotkey_status = tk.StringVar(value='')
    app.hotkey_error = ttk.Label(footer, textvariable=app.hotkey_status, foreground='#bd544f', wraplength=440)
