import tkinter as tk
import unittest
from tkinter import ttk

import theme
from ui_components import SettingsTabs


class SettingsThemeTests(unittest.TestCase):
    def test_dialog_tab_switches_keep_active_theme(self):
        root = tk.Tk()
        window = tk.Toplevel(root)
        tabs = SettingsTabs(window)
        tabs.pack(fill='both', expand=True)
        for title in ('音频与转录', 'AI 与外观', '参考资料', '工具与数据'):
            tabs.add(ttk.Frame(tabs), text=title)
        try:
            for dark in (True, False, True):
                theme.apply(root, dark)
                self.assertEqual(window._dark_theme, dark)
                for selected in tabs.pages:
                    tabs.select(selected)
                    root.update()
                    for key, (_, tab) in tabs.pages.items():
                        expected = '#E6F2FB' if key == selected else '#f7f8fa'
                        self.assertEqual(tab.cget('background'), theme.color(expected, dark))
                        self.assertEqual(tab.cget('activebackground'), theme.color('#E6F2FB', dark))
        finally:
            root.destroy()
