"""Compact, collapsible question composer for the conversation panel."""
import tkinter as tk
from PIL import Image, ImageTk
from qa_images import ConversationImage

import typography
from ime_support import sync_composition_font
from text_shortcuts import bind_question_shortcuts


class QuestionComposer(tk.Frame):
    def __init__(self, parent, submit, font_size=9, on_toggle=None, paste_image=None):
        super().__init__(parent, bg='white', height=126, bd=0)
        self.pack_propagate(False)
        self.expanded = True
        self.expanded_height = 126
        self.on_toggle = on_toggle
        self.submit = submit
        self.reference_label = None
        self.context_meter = None
        self.image_item = self.image_preview = self.image_photo = None
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
        if paste_image is not None:
            self.input.bind('<<Paste>>', paste_image)
        self.attachment = tk.Frame(self, bg='white', bd=0)
        self.image_label = tk.Label(self.attachment, bg='white', bd=0)
        self.image_label.pack(fill='both', expand=True)
        self.remove_image_button = tk.Button(self.attachment, text='×', command=self.remove_image,
            bg='white', fg='#737b8c', activebackground='#E6F2FB', relief='flat', bd=0,
            font=(typography.UI_FAMILY, 10), cursor='hand2', padx=0, pady=0)
        self.remove_image_button.place(relx=1, x=0, y=0, anchor='ne', width=16, height=16)
        self.bind('<Configure>', self.layout_image, add='+')
        self.actions = tk.Frame(self, bg='white', bd=0)
        self.actions.place(x=4, rely=1, y=-6, anchor='sw', relwidth=1, width=-8, height=32)
        self.middle = tk.Frame(self.actions, bg='white', cursor='xterm', width=1)
        self.middle.bind('<Button-1>', self.expand)
        self.set_expanded(True)
        self.update_placeholder()

    def get(self):
        return self.input.get('1.0', 'end-1c')

    def clear(self):
        self.remove_image()
        self.input.delete('1.0', 'end')
        self.input.edit_reset()
        self.update_placeholder()

    def set_image(self, item):
        preview = ConversationImage.from_bytes(item.image)
        if preview is None:
            raise ValueError('无法预览这张图片，请重新复制。')
        self.image_item, self.image_preview = item, preview
        self.expand()
        self.layout_image()

    def remove_image(self):
        self.image_item = self.image_preview = self.image_photo = None
        self.image_label.configure(image='')
        self.layout_image()

    def layout_image(self, event=None):
        if event is not None and event.widget is not self:
            return
        offset = 68 if self.image_item is not None else 0
        if self.expanded:
            self.input.place(x=14+offset, y=8, relwidth=1, width=-28-offset, relheight=1, height=-50)
        if self.expanded and self.image_preview is not None:
            height = max(20, min(56, self.winfo_height()-50))
            thumbnail = self.image_preview.source.copy()
            thumbnail.thumbnail((56, height), Image.Resampling.LANCZOS)
            self.image_photo = ImageTk.PhotoImage(thumbnail, master=self)
            self.image_label.configure(image=self.image_photo)
            self.attachment.place(x=14, y=8, width=56, height=height)
        else:
            self.attachment.place_forget()

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
        if self.context_meter is not None:
            if expanded:
                self.context_meter.pack(side='right', padx=(0, 2), before=self.middle)
            else:
                self.context_meter.pack_forget()
        if changed:
            self.configure(height=self.expanded_height if expanded else 40)
        if expanded:
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
        self.layout_image()

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
