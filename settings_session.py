"""Draft lifetime and explicit commit for the main settings dialog."""
from app_paths import DATA
from capture_privacy import CapturePrivacyError


class SettingsSession:
    def __init__(self, app):
        self.app = app
        self.values = [(widget, widget.get()) for widget in
                       (app.model, app.language, app.mode, app.device, app.microphone,
                        app.hotwords, app.dark_mode, app.font_size, app.capture_hidden)]
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
        # Keep a failed multi-file save from leaving only some settings applied.
        files = [DATA / name for name in
                 ('desktop-settings.json', 'recognition-settings.json', 'reference-files.json')]
        originals = {path: path.read_bytes() if path.exists() else None for path in files}
        previous_capture_mode = app.capture_privacy.enabled
        try:
            if app.capture_hidden.get() != previous_capture_mode:
                app.capture_privacy.set_enabled(app.capture_hidden.get())
            app.save_desktop_settings(force=True, raise_errors=True)
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
        if app.auto_qa.get() != app.settings_auto_qa.get():
            app.auto_qa.set(app.settings_auto_qa.get())
            app.toggle_auto_qa()
