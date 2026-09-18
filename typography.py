"""Shared Windows system typography."""
import tkinter.font as tkfont

UI_FAMILY = 'Microsoft YaHei UI'


def setup(root):
    global UI_FAMILY
    available = list(tkfont.families(root))
    lookup = {name.lower(): name for name in available}
    UI_FAMILY = next((lookup[name.lower()] for name in
        ('Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI') if name.lower() in lookup),
        tkfont.nametofont('TkDefaultFont', root=root).actual('family'))
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont', 'TkTooltipFont'):
        tkfont.nametofont(name, root=root).configure(family=UI_FAMILY, size=10)
    root.option_add('*Font', (UI_FAMILY, 10))
    return UI_FAMILY
