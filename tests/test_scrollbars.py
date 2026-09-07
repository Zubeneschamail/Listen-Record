import tkinter as tk
import unittest
from types import SimpleNamespace
from scrollbars import SlimScrollbar


class ScrollbarTests(unittest.TestCase):
    def test_overflow_drag_wheel_and_auto_hide_without_resizing_text(self):
        root = tk.Tk()
        root.geometry("360x220")
        errors = []
        root.report_callback_exception = lambda *args: errors.append(args)
        text = tk.Text(root, wrap="word", padx=14)
        text.pack(fill="both", expand=True)
        scrollbar = SlimScrollbar(text)
        try:
            root.update()
            self.assertFalse(scrollbar.winfo_manager())
            original_width = text.winfo_width()
            text.insert("end", "\n".join(f"第 {i} 行文字" for i in range(100)))
            root.update()
            self.assertEqual(scrollbar.winfo_manager(), "place")
            self.assertEqual(original_width, text.winfo_width())
            top, size, _, _ = scrollbar.metrics()
            scrollbar.press(SimpleNamespace(y=top+size/2))
            self.assertEqual(scrollbar.itemcget(scrollbar.thumb, "fill"), "#007ACC")
            scrollbar.drag(SimpleNamespace(y=scrollbar.winfo_height()+200))
            root.update()
            self.assertGreater(text.yview()[1], .99)
            scrollbar.release(SimpleNamespace())
            old = text.yview()[0]
            scrollbar.wheel(SimpleNamespace(delta=120))
            root.update()
            self.assertLess(text.yview()[0], old)
            scrollbar.press(SimpleNamespace(y=0))
            scrollbar.release(SimpleNamespace())
            root.update()
            self.assertLess(text.yview()[0], .01)
            text.delete("1.0", "end")
            root.update()
            self.assertFalse(scrollbar.winfo_manager())
            self.assertEqual(original_width, text.winfo_width())
            self.assertFalse(errors)
        finally:
            root.destroy()
