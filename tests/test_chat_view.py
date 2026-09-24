import tkinter as tk
import unittest

from chat_view import ChatView


class ChatViewLayoutTests(unittest.TestCase):
    def test_clear_collapses_old_scroll_height_before_next_bubble(self):
        root = tk.Tk()
        root.geometry("500x500")
        view = ChatView(root, lambda *args: None, lambda *args: None)
        view.pack(fill="both", expand=True)
        try:
            root.update()
            for index in range(40):
                view.add(index, {
                    "id": index,
                    "text": "旧对话内容 " * 35,
                    "source": "system",
                    "time": "09:00:00",
                })
            root.update()
            self.assertGreater(view.inner.winfo_height(), view.canvas.winfo_height())

            view.clear()
            self.assertEqual(view.canvas.bbox("all")[-1], 1)

            view.add(0, {
                "id": 0,
                "text": "新对话的第一句",
                "source": "system",
                "time": "09:01:00",
            })
            root.update()
            self.assertEqual(view.canvas.yview()[0], 0.0)
            self.assertEqual(view.bubbles[0].winfo_y(), 0)
        finally:
            for identifier in root.tk.call("after", "info"):
                root.after_cancel(identifier)
            root.destroy()
