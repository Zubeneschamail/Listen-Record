import tkinter as tk
import unittest
from ui_components import SplitterHandle


class SplitterTests(unittest.TestCase):
    def test_wide_hit_area_and_commit_on_release(self):
        root = tk.Tk()
        root.geometry('600x300')
        errors = []
        root.report_callback_exception = lambda *args: errors.append(args)
        panes = tk.PanedWindow(root, sashwidth=1, opaqueresize=False)
        panes.pack(fill='both', expand=True)
        left, right = tk.Frame(panes), tk.Frame(panes)
        panes.add(left, minsize=150)
        panes.add(right, minsize=150)
        handle = SplitterHandle(panes)
        try:
            root.update()
            panes.sash_place(0, 280, 0)
            handle.position()
            root.update()
            original = panes.sash_coord(0)[0]
            self.assertEqual(handle.winfo_width(), 13)
            self.assertEqual(handle.winfo_x(), original-6)
            # Press six pixels away from the visual one-pixel divider.
            handle.event_generate('<ButtonPress-1>', x=0, y=40, rootx=300, rooty=100)
            handle.event_generate('<B1-Motion>', x=40, y=40, rootx=340, rooty=100)
            root.update()
            self.assertEqual(panes.sash_coord(0)[0], original)
            handle.event_generate('<ButtonRelease-1>', x=40, y=40, rootx=340, rooty=100)
            root.update()
            self.assertEqual(panes.sash_coord(0)[0], original+40)
            self.assertIsNone(root.grab_current())
            panes.forget(right)
            handle.position()
            root.update()
            self.assertFalse(handle.winfo_ismapped())
            self.assertEqual(errors, [])
        finally:
            root.destroy()
