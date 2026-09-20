import ctypes
import sys
import tkinter as tk
import tkinter.font as tkfont
import unittest
from unittest.mock import patch

from ime_support import LOGFONTW, imm_api, sync_composition_font
from qa_composer import QuestionComposer


@unittest.skipUnless(sys.platform == 'win32', 'Windows IME integration')
class CompositionFontTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.composer = QuestionComposer(self.root, lambda: None)
        self.composer.pack(fill='both', expand=True)
        self.root.update()

    def tearDown(self):
        self.root.destroy()

    def read_font(self):
        api = imm_api()
        api.ImmGetCompositionFontW.argtypes = [ctypes.c_void_p, ctypes.POINTER(LOGFONTW)]
        api.ImmGetCompositionFontW.restype = ctypes.c_int
        hwnd = self.root.winfo_id()
        context = api.ImmGetContext(hwnd)
        self.assertTrue(context)
        try:
            font = LOGFONTW()
            self.assertTrue(api.ImmGetCompositionFontW(context, ctypes.byref(font)))
            return font
        finally:
            api.ImmReleaseContext(hwnd, context)

    def test_focus_and_resize_sync_native_composition_font_without_changing_draft(self):
        self.composer.input.insert('1.0', '保留草稿')
        self.root.focus_force()
        self.composer.input.focus_set()
        self.root.update()
        for size in (10, 14):
            self.composer.set_font_size(size)
            font = self.read_font()
            expected = tkfont.Font(root=self.root, font=self.composer.input.cget('font')).actual()
            self.assertEqual(font.lfFaceName, expected['family'])
            self.assertEqual(font.lfHeight, -round(size * self.root.winfo_fpixels('1p')))
            self.assertEqual(font.lfWeight, 400)
            self.assertEqual(self.composer.get(), '保留草稿')

    def test_placeholder_is_hidden_during_empty_focused_input(self):
        self.root.focus_force()
        self.composer.input.focus_set()
        self.root.update()
        self.assertFalse(self.composer.placeholder.winfo_ismapped())
        self.root.focus_set()
        self.root.update()
        self.assertTrue(self.composer.placeholder.winfo_ismapped())

    def test_unavailable_input_context_does_not_block_typing(self):
        with patch('ime_support.imm_api') as factory:
            factory.return_value.ImmGetContext.return_value = None
            self.assertFalse(sync_composition_font(self.composer.input))
            factory.return_value.ImmSetCompositionFontW.assert_not_called()
            factory.return_value.ImmReleaseContext.assert_not_called()


if __name__ == '__main__':
    unittest.main()
