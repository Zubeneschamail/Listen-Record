"""Compact, collapsible question composer for the conversation panel."""
import tkinter as tk

import typography
from ime_support import sync_composition_font
from text_shortcuts import bind_question_shortcuts


class QuestionComposer(tk.Frame):
    def __init__(self, parent, submit, font_size=9, on_toggle=None):
        super().__init__(parent, bg='white', height=126, bd=0)
        self.pack_propagate(False)
        self.expanded = True
        self.expanded_height = 126
        self.on_toggle = on_toggle
        self.submit = submit
        self.reference_label = None
        self.bind('<Button-1>', self.expand)
        self.input = tk.Text(self, wrap='word', height=3, width=1, undo=True,
                             bg='white', fg='#4f586b', insertbackground='#007ACC', insertwidth=1,
                             font=(typography.UI_FAMILY, max(10, font_size)), bd=0, highlightthickness=0,
                             padx=0, pady=2, spacing1=2, spacing2=5, spacing3=3, exportselection=False,
                             selectbackground='#E6F2FB', selectforeground='#263044', selectborderwidth=0)
        self.placeholder = tk.Label(self.input, text='输入问题…', bg='white', fg='#a0a5b1',
                                    font=(typography.UI_FAMILY, max(10, font_size)), bd=0, padx=0, pady=0,
                                    cursor='xterm')
        self.placeholder.bind('<Button-1>', self.expand)
        self.input.bind('<<Modified>>', self.changed)
        self.input.bind('<FocusIn>', self.focus_input)
        self.input.bind('<FocusOut>', lambda event: self.update_placeholder())
        bind_question_shortcuts(self.input, self.submit)
        self.input.bind('<Escape>', self.collapse)
        self.actions = tk.Frame(self, bg='white', bd=0)
        self.actions.place(x=16, rely=1, y=-6, anchor='sw', relwidth=1, width=-30, height=32)
        self.middle = tk.Frame(self.actions, bg='white', cursor='xterm', width=1)
        self.middle.bind('<Button-1>', self.expand)
        self.set_expanded(True)
        self.update_placeholder()

    def get(self):
        return self.input.get('1.0', 'end-1c')

    def clear(self):
        self.input.delete('1.0', 'end')
        self.input.edit_reset()
        self.update_placeholder()

    def changed(self, event=None):
        if self.input.edit_modified():
            self.input.edit_modified(False)
            self.update_placeholder()

    def update_placeholder(self):
        if self.get() or self.input.focus_get() is self.input:
            self.placeholder.place_forget()
        else:
            self.placeholder.place(x=0, y=4)

    def focus_input(self, event=None):
        self.update_placeholder()
        sync_composition_font(self.input)

    def set_reference_label(self, label):
        self.reference_label = label
        self.set_expanded(self.expanded)

    def set_expanded(self, expanded):
        changed = expanded != self.expanded
        if self.expanded and not expanded and self.winfo_height() > 40:
            self.expanded_height = self.winfo_height()
        self.expanded = expanded
        if changed:
            self.configure(height=self.expanded_height if expanded else 40)
        if expanded:
            self.input.place(x=16, y=10, relwidth=1, width=-32, relheight=1, height=-52)
            if self.reference_label is not None:
                self.reference_label.pack(fill='both', expand=True)
        else:
            self.input.place_forget()
            if self.reference_label is not None:
                for item in self.reference_label.items:
                    item.hide_tip()
                self.reference_label.pack_forget()
            self.winfo_toplevel().focus_set()
        if changed and self.on_toggle is not None:
            self.on_toggle(expanded)

    def expand(self, event=None):
        self.set_expanded(True)
        self.input.focus_set()
        return 'break'

    def collapse(self, event=None):
        self.set_expanded(False)
        return 'break'

    def set_font_size(self, size):
        self.input.configure(font=(typography.UI_FAMILY, max(10, size)))
        self.placeholder.configure(font=(typography.UI_FAMILY, max(10, size)))
        if self.input.focus_get() is self.input:
            sync_composition_font(self.input)
