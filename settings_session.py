"""Draft lifetime and explicit commit for the main settings dialog."""
from app_paths import DATA
from capture_privacy import CapturePrivacyError
from recognition import preferences_revision, PreferenceConflictError


class SettingsSession:
    def __init__(self, app, *, recognition_revision, recognition_readable):
        self.app = app
        self.hotwords = app.hotwords.get()
        self.recognition_revision = recognition_revision
        self.recognition_readable = recognition_readable
        self.values = [(widget, widget.get()) for widget in
                       (app.model, app.language, app.mode, app.device, app.microphone,
                        app.hotwords, app.dark_mode, app.font_size, app.capture_hidden, app.echo_cancellation,
                        app.screen_capture_enabled, app.question_source)]
        if hasattr(app, 'manual_cut_mouse'):
            self.values.append((app.manual_cut_mouse, app.manual_cut_mouse.get()))
        self.pinned = app.pinned
        self.references = app.reference_controls['snapshot']()
        app.settings_auto_qa.set(app.auto_qa.get())

    def restore(self, preview=True):
        app = self.app
        changed = [widget for widget, value in self.values if widget.get() != value]
        for widget, value in self.values:
            widget.set(value)
        if preview and app.dark_mode in changed:
            app.apply_theme()
        if preview and app.font_size in changed:
            app.change_font_size()
        if preview and app.pinned != self.pinned:
            app.toggle_pin()
        app.reference_controls['restore'](self.references)

    def save(self):
        app = self.app
        words_changed = app.hotwords.get() != self.hotwords
        if words_changed and (not self.recognition_readable or
                preferences_revision(DATA / 'recognition-settings.json') != self.recognition_revision):
            raise PreferenceConflictError('词库已在其他地方更新或无法核对，原文件未改动。请先保留草稿，再重新打开设置核对后保存。')
        # Keep a failed multi-file save from leaving only some settings applied.
        files = [DATA / name for name in
                 ('desktop-settings.json', 'reference-files.json')]
        if words_changed:
            files.append(DATA / 'recognition-settings.json')
        originals = {path: path.read_bytes() if path.exists() else None for path in files}
        previous_capture_mode = app.capture_privacy.enabled
        try:
            if app.capture_hidden.get() != previous_capture_mode:
                app.capture_privacy.set_enabled(app.capture_hidden.get())
            app.save_desktop_settings(force=True, raise_errors=True, save_recognition=words_changed)
            app.reference_controls['persist']()
        except (OSError, CapturePrivacyError):
            if app.capture_privacy.enabled != previous_capture_mode:
                app.capture_privacy.set_enabled(previous_capture_mode)
            for path, contents in originals.items():
                if (path.read_bytes() if path.exists() else None) == contents:
                    continue
                if contents is None:
                    path.unlink(missing_ok=True)
                else:
                    temporary = path.with_suffix('.rollback.tmp')
                    temporary.write_bytes(contents)
                    temporary.replace(path)
            raise
        if app.reference_controls['snapshot']() != self.references:
            app.qa.reset()
            app.qa_selection = None
            app.qa_status.set('')
            from knowledge_packages import context_changed
            if context_changed(self.references, app.reference_controls['snapshot']()):
                app.reset_knowledge_context()
        if app.auto_qa.get() != app.settings_auto_qa.get():
            app.auto_qa.set(app.settings_auto_qa.get())
            app.toggle_auto_qa()
        app.apply_question_source()
        app.apply_screen_hotkey()
        if hasattr(app, 'apply_manual_cut_mouse'):
            app.apply_manual_cut_mouse()
