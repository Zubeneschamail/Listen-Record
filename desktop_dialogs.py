"""Small, non-blocking desktop service dialogs."""
import threading
import tkinter as tk
import typography
from tkinter import ttk, messagebox
from app_paths import DATA, LOGS, MODELS
from version import VERSION


def dispatch(app, action):
    app.events.put(('desktop_callback', action))


def dialog(app, title):
    window = tk.Toplevel(app.root)
    window.title(title)
    window.transient(app.root)
    window.configure(padx=20, pady=16, bg='#f7f8fa')
    window.resizable(False, False)
    app.root.after_idle(app.apply_theme)
    return window


def about(app):
    from updates import check_update, download_update, launch_installer_when_closed
    window = dialog(app, '关于闻录')
    ttk.Label(window, text=f'闻录 {VERSION}', font=(typography.UI_FAMILY, 14)).pack(anchor='w')
    ttk.Label(window, textvariable=app.codex_status).pack(anchor='w', pady=(8, 0))
    status = tk.StringVar(value='本地转写 · 数据保存在此电脑\n可在设置中切换问答模型并检查连接。')
    ttk.Label(window, textvariable=status, wraplength=420).pack(anchor='w', pady=12)
    ttk.Button(window, text='打开日志目录', command=lambda: app.open_folder(LOGS)).pack(anchor='w')
    def post(text):
        dispatch(app, lambda: status.set(text) if window.winfo_exists() else None)
    def completed(manifest, path):
        if not window.winfo_exists():
            return
        button.configure(state='normal')
        if messagebox.askyesno('更新已就绪', '安装更新将停止转写并保存记录，是否现在安装？', parent=window):
            launch_installer_when_closed(path, manifest['sha256'])
            app.close()
    def work():
        try:
            manifest = check_update()
            if manifest is None:
                post('已是最新版本。')
                return
            post('发现版本 ' + manifest['version'] + '，正在下载…')
            path = download_update(manifest, lambda n: post(f'正在下载更新：{n}%'))
            dispatch(app, lambda: completed(manifest, path))
        except Exception as exc:
            post('更新检查失败，可稍后重试：' + str(exc))
        finally:
            dispatch(app, lambda: button.configure(state='normal') if window.winfo_exists() else None)
    def start():
        button.configure(state='disabled')
        status.set('正在检查更新…')
        threading.Thread(target=work, daemon=True).start()
    button = ttk.Button(window, text='检查更新', command=start)
    button.pack(anchor='e', pady=(16, 0))


def model_manager(app):
    import multiprocessing as mp
    import time
    from model_download import cached_model, download_worker, BUNDLED_MODELS, progress_text
    existing = getattr(app, '_model_window', None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return
    window = dialog(app, '模型管理')
    if app.settings_window.winfo_viewable():
        window.transient(app.settings_window)
        window.grab_set()
    app._model_window = window
    window.configure(bg='#f7f8fa')
    ttk.Label(window, text='转录模型', font=(typography.UI_FAMILY, 12, 'bold')).pack(anchor='w')
    ttk.Label(window, text='统一下载和切换；下载完成后可离线使用。').pack(anchor='w', pady=(6, 0))
    names = ['small', 'large-v3-turbo', 'base', 'tiny']
    def model_labels():
        result = []
        current = app.model.get().split()[0]
        for name in names:
            path = cached_model(name)
            state = '内置' if path == BUNDLED_MODELS / name else '已下载' if path else '未下载'
            result.append(f'{name} · {state}' + (' · 使用中' if name == current else ''))
        return result
    choice = ttk.Combobox(window, values=model_labels(),
                          state='readonly', width=34, style='Settings.TCombobox')
    choice.current(names.index(app.model.get().split()[0]))
    choice.pack(fill='x', pady=12)
    status = tk.StringVar()
    ttk.Label(window, textvariable=status, wraplength=340).pack(anchor='w')
    progress = ttk.Progressbar(window, mode='indeterminate', style='Slim.Horizontal.TProgressbar')
    job = None
    receiver = None
    connected = False
    started = 0
    poll_id = None

    def selected():
        return choice.get().split()[0]

    def stop_job():
        nonlocal job, receiver, poll_id
        if poll_id is not None:
            app.root.after_cancel(poll_id)
            poll_id = None
        if job is not None:
            if job.is_alive():
                job.terminate()
            job.join(timeout=1)
            job = None
        if receiver is not None:
            receiver.close()
            receiver = None
        if not window.winfo_exists():
            return
        progress.stop()
        progress.pack_forget()
        choice.configure(state='readonly')
        button.configure(text='下载 / 重试', state='normal')

    def refresh(event=None):
        name = selected()
        choice.configure(values=model_labels())
        choice.current(names.index(name))
        ready = cached_model(name) is not None
        status.set('已找到本机模型，可直接离线使用。' if ready else '尚未下载此模型。首次下载需要连接网络。')
        button.configure(text='使用此模型' if ready else '下载模型')

    def poll():
        nonlocal connected, poll_id
        poll_id = None
        if not window.winfo_exists() or job is None:
            return
        terminal = False
        try:
            while receiver.poll():
                kind, message = receiver.recv()
                connected = connected or kind != 'connecting'
                if kind == 'bytes':
                    completed, total = message
                    progress.stop()
                    progress.configure(mode='determinate', maximum=max(total, 1), value=completed)
                    status.set(progress_text(completed, total))
                else:
                    status.set(message)
                    if kind == 'progress' and '校验' in message:
                        progress.stop()
                        progress.configure(mode='indeterminate')
                        progress.start(15)
                if kind in ('done', 'error'):
                    terminal = True
                    stop_job()
                    if kind == 'done':
                        refresh()
                    break
        except EOFError:
            pass
        if terminal:
            return
        if not job.is_alive():
            stop_job()
            status.set('下载进程意外退出，请重试。')
            return
        if not connected and time.monotonic() - started > 30:
            stop_job()
            status.set('连接超时，请检查网络后重试。')
            return
        poll_id = app.root.after(150, poll)

    def start():
        nonlocal job, receiver, connected, started, poll_id
        if job is not None:
            stop_job()
            status.set('已取消下载，重试时将复用已下载的文件。')
            return
        if cached_model(selected()) is not None and button.cget('text') != '下载 / 重试':
            if app.busy:
                status.set('请先停止转写，再切换模型。')
                return
            for label in app.model['values']:
                if label.split()[0] == selected():
                    app.model.set(label)
                    app.save_desktop_settings()
                    break
            close()
            return
        context = mp.get_context('spawn')
        receiver, sender = context.Pipe(duplex=False)
        job = context.Process(target=download_worker, args=(selected(), sender), daemon=True)
        try:
            job.start()
        except Exception as exc:
            job = None
            receiver.close()
            receiver = None
            status.set('无法启动下载：' + str(exc))
            return
        finally:
            sender.close()
        connected = False
        started = time.monotonic()
        choice.configure(state='disabled')
        button.configure(text='取消下载')
        status.set('正在检查模型与网络连接…')
        progress.pack(fill='x', pady=(12, 0), before=button)
        progress.configure(mode='indeterminate', value=0)
        progress.start(15)
        poll_id = app.root.after(150, poll)

    def close():
        stop_job()
        window.grab_release()
        window.destroy()
        if app.settings_window.winfo_viewable():
            app.settings_window.grab_set()

    button = tk.Button(window, text='下载模型', command=start, bg='#007ACC', fg='white',
                       activebackground='#006BB3', activeforeground='white', relief='flat',
                       bd=0, padx=18, pady=7, font=(typography.UI_FAMILY, 9))
    button.pack(anchor='e', pady=(14, 0))
    choice.bind('<<ComboboxSelected>>', refresh)
    window.protocol('WM_DELETE_WINDOW', close)
    window.bind('<Destroy>', lambda event: stop_job() if event.widget is window else None)
    refresh()
    window.update_idletasks()
    parent = app.settings_window if app.settings_window.winfo_viewable() else app.root
    x = max(0, parent.winfo_rootx() + (parent.winfo_width() - window.winfo_reqwidth()) // 2)
    y = max(0, parent.winfo_rooty() + (parent.winfo_height() - window.winfo_reqheight()) // 2)
    window.geometry(f'+{x}+{y}')
    app.root.after_idle(app.apply_theme)


def background_update_check(app):
    """Notify only; installation is always initiated by the user."""
    def work():
        try:
            from updates import check_update
            manifest = check_update()
            if manifest:
                def notify():
                    app.menu.entryconfigure('关于 / 检查更新', label='发现新版本 ' + manifest['version'], command=app.open_about)
                dispatch(app, notify)
        except Exception:
            import logging
            logging.info('Background update check unavailable')
    threading.Thread(target=work, daemon=True).start()
