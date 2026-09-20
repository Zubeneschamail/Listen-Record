"""Shared send/newline keys for question composition and inline editing."""
from ime_support import is_composing


def bind_question_shortcuts(widget, submit):
    def send(event):
        if not is_composing(widget):
            submit()
        return 'break'

    def newline(event):
        if not is_composing(widget):
            # Use Tk's normal insertion path to replace selections and preserve undo.
            widget.tk.call('tk::TextInsert', widget, '\n')
            if widget.cget('autoseparators'):
                widget.edit_separator()
            widget.see('insert')
        return 'break'

    widget.bind('<Return>', send)
    widget.bind('<Control-Return>', newline)
