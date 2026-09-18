"""Shared checkbox indicator while retaining ttk keyboard/state behavior."""
from PIL import Image, ImageDraw, ImageTk
import typography


def apply(root, style, dark):
    cache = getattr(root, '_checkbox_images', {})
    if dark not in cache:
        images = {}
        for name in ('off', 'on', 'hover', 'on_hover', 'focus', 'on_focus', 'disabled', 'on_disabled'):
            selected = name.startswith('on')
            disabled = 'disabled' in name
            focused = 'focus' in name
            hover = 'hover' in name
            image = Image.new('RGBA', (80, 80))
            draw = ImageDraw.Draw(image)
            accent = '#60A5FA' if dark else '#007ACC'
            border = '#728299' if dark else '#a3adba'
            fill = '#222C3B' if dark else '#ffffff'
            if selected:
                fill = '#3585D5' if dark else '#007ACC'
                if hover:
                    fill = '#4595E5' if dark else '#006BB3'
                border = fill
            elif hover:
                border = accent
                fill = '#283447' if dark else '#f0f7fc'
            if disabled:
                fill = '#354358' if dark else '#edf0f7'
                border = '#49586C' if dark else '#dce3eb'
            if focused:
                draw.rounded_rectangle((2, 2, 77, 77), radius=16, outline=accent, width=4)
            draw.rounded_rectangle((12, 12, 67, 67), radius=11, fill=fill, outline=border, width=4)
            if selected:
                ink = ('#728299' if dark else '#a3adba') if disabled else '#ffffff'
                points = [(25, 40), (36, 51), (55, 29)]
                draw.line(points, fill=ink, width=7, joint='curve')
                for x, y in points:
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=ink)
            images[name] = ImageTk.PhotoImage(image.resize((20, 20), Image.Resampling.LANCZOS), master=root)
        cache[dark] = images
        root._checkbox_images = cache
    images = cache[dark]
    element = 'WenluDark.Check.indicator' if dark else 'WenluLight.Check.indicator'
    if element not in style.element_names():
        style.element_create(element, 'image', images['off'],
            ('disabled', 'selected', images['on_disabled']), ('disabled', images['disabled']),
            ('focus', 'selected', images['on_focus']), ('focus', images['focus']),
            ('active', 'selected', images['on_hover']), ('active', images['hover']),
            ('selected', images['on']), sticky='w')
    style.layout('TCheckbutton', [('Checkbutton.padding', {'sticky': 'nswe', 'children': [
        (element, {'side': 'left', 'sticky': 'w'}),
        ('Checkbutton.label', {'side': 'left', 'sticky': 'nswe'})]})])
    style.configure('TCheckbutton', font=(typography.UI_FAMILY, 9), padding=(0, 3),
                    indicatorspace=6)
    style.map('TCheckbutton', foreground=[('disabled', '#728299' if dark else '#a3adba')],
              background=[('active', '#141B25' if dark else '#f7f8fa')])
