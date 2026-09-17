"""Reversible opaque light/dark palettes for the existing Tk interface."""
import sys
import tkinter as tk
from tkinter import ttk

PALETTE = {
    '#15803d': '#4ADE80', '#2563eb': '#60A5FA',
    '#b45309': '#FBBF24', '#b91c1c': '#F87171',
    'white': '#1C2430', '#ffffff': '#1C2430', '#f7f8fa': '#141B25',
    '#e0e3ea': '#303D50', '#eceef3': '#303D50', '#e2e5ed': '#354358',
    '#edf0f7': '#283447', '#f1f3f7': '#222C3B', '#f1f5f9': '#283447',
    '#edf5fb': '#253B50', '#f0f7fc': '#253B50', '#e6f2fb': '#254663',
    '#edf4fa': '#243D54', '#f0f1f3': '#303A48', '#dcecf8': '#315B7C', '#e0e4e9': '#49586C',
    '#263044': '#E3EAF4', '#4f586b': '#CCD7E5', '#737b8c': '#AEBBCD',
    '#858b98': '#A3B1C4', '#9297a4': '#A3B1C4', '#9299aa': '#A3B1C4',
    '#a0a5b1': '#8E9EB2', '#a0a6b2': '#8E9EB2', '#a2a8b5': '#728299',
    '#a3a8b4': '#91A0B4', '#a3adba': '#91A0B4', '#a6acba': '#728299',
    '#527f9e': '#91BDD9', '#668eab': '#91BDD9', '#7896ad': '#91ADC5',
    '#8a929c': '#A3B1C4', '#dce3eb': '#4B5F78', '#9dc3df': '#7FA7C8',
    '#bd544f': '#F39791', '#b65b39': '#F0AA82',
}


def color(value, dark, foreground=False):
    if not dark or (foreground and value.lower() in ('white', '#ffffff')):
        return value
    return PALETTE.get(value.lower(), value)


def apply(root, dark):
    root._dark_theme = dark
    style = ttk.Style(root)
    bg, surface, fg = color('#f7f8fa', dark), color('white', dark), color('#4f586b', dark)
    for name in ('TFrame', 'TLabel', 'TCheckbutton'):
        style.configure(name, background=bg)
    style.configure('TLabel', foreground=color('#858b98', dark))
    style.configure('TCheckbutton', foreground=fg)
    style.map('TCheckbutton', background=[('active', color('#E6F2FB', dark))])
    style.configure('TButton', background=surface, foreground=fg)
    style.map('TButton', background=[('active', color('#E6F2FB', dark))],
              foreground=[('disabled', color('#a6acba', dark))])
    for name in ('Settings.TCombobox', 'Settings.TEntry', 'TCombobox'):
        style.configure(name, fieldbackground=surface, background=surface, foreground=fg,
                        bordercolor=color('#e2e5ed', dark), lightcolor=surface, darkcolor=surface,
                        arrowcolor=color('#9299aa', dark), selectbackground=color('#E6F2FB', dark),
                        selectforeground=fg)
        style.map(name, fieldbackground=[('disabled', color('#f1f3f7', dark)), ('readonly', surface)],
                  background=[('active', color('#F0F7FC', dark)), ('readonly', surface)],
                  foreground=[('disabled', color('#a2a8b5', dark)), ('readonly', fg)],
                  lightcolor=[('focus', surface)], darkcolor=[('focus', surface)])
    style.configure('Slim.Horizontal.TProgressbar', troughcolor=color('#edf0f7', dark))
    for option, value in (('background', surface), ('foreground', fg),
                          ('selectBackground', color('#E6F2FB', dark)), ('selectForeground', fg)):
        root.option_add('*TCombobox*Listbox.' + option, value)

    def visit(widget):
        if getattr(widget, '_custom_theme', False):
            widget.apply_theme(dark)
            return
        # Floating subtitles keep their deliberately transparent, high-contrast style.
        if widget is not root and isinstance(widget, tk.Toplevel) and widget.__class__.__name__ == 'FloatingCaption':
            return
        originals = getattr(widget, '_light_colors', {})
        for option in ('background', 'foreground', 'activebackground', 'activeforeground',
                       'disabledforeground', 'highlightbackground', 'highlightcolor',
                       'selectbackground', 'selectforeground', 'selectcolor', 'insertbackground'):
            if option in widget.keys():
                if option not in originals:
                    originals[option] = str(widget.cget(option))
                value = originals[option]
                if value:
                    widget.configure(**{option: color(value, dark, 'foreground' in option)})
        widget._light_colors = originals
        if isinstance(widget, ttk.Combobox):
            try:
                popup = widget.tk.call('ttk::combobox::PopdownWindow', widget)
                widget.tk.call(popup + '.f.l', 'configure', '-background', surface, '-foreground', fg,
                               '-selectbackground', color('#E6F2FB', dark), '-selectforeground', fg)
            except tk.TclError:
                pass
        if isinstance(widget, tk.Text):
            tags = getattr(widget, '_light_tags', {})
            for tag in widget.tag_names():
                for option in ('foreground', 'background'):
                    key = (tag, option)
                    if key not in tags:
                        tags[key] = widget.tag_cget(tag, option)
                    if tags[key]:
                        widget.tag_configure(tag, **{option: color(tags[key], dark, option == 'foreground')})
            widget._light_tags = tags
        if isinstance(widget, tk.Menu):
            widget.configure(bg=bg, fg=fg, activebackground=color('#E6F2FB', dark), activeforeground=fg)
        if isinstance(widget, (tk.Tk, tk.Toplevel)) and sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            api = ctypes.windll.user32
            api.GetParent.argtypes, api.GetParent.restype = [wintypes.HWND], wintypes.HWND
            hwnd = api.GetParent(widget.winfo_id())
            value = ctypes.c_int(int(dark))
            dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
            dwm.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            dwm(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        for child in widget.winfo_children():
            visit(child)
    visit(root)
