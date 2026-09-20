"""Keep Windows IME composition typography aligned with a Tk text widget."""
import ctypes
from ctypes import wintypes
from functools import lru_cache
import sys
import tkinter as tk
import tkinter.font as tkfont


class LOGFONTW(ctypes.Structure):
    _fields_ = [(name, wintypes.LONG) for name in
                ('lfHeight', 'lfWidth', 'lfEscapement', 'lfOrientation', 'lfWeight')] + [
                (name, wintypes.BYTE) for name in
                ('lfItalic', 'lfUnderline', 'lfStrikeOut', 'lfCharSet', 'lfOutPrecision',
                 'lfClipPrecision', 'lfQuality', 'lfPitchAndFamily')] + [
                ('lfFaceName', wintypes.WCHAR * 32)]


@lru_cache(maxsize=1)
def imm_api():
    api = ctypes.WinDLL('imm32', use_last_error=True)
    api.ImmGetContext.argtypes = [wintypes.HWND]
    api.ImmGetContext.restype = wintypes.HANDLE
    api.ImmReleaseContext.argtypes = [wintypes.HWND, wintypes.HANDLE]
    api.ImmReleaseContext.restype = wintypes.BOOL
    api.ImmSetCompositionFontW.argtypes = [wintypes.HANDLE, ctypes.POINTER(LOGFONTW)]
    api.ImmSetCompositionFontW.restype = wintypes.BOOL
    api.ImmGetCompositionStringW.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    api.ImmGetCompositionStringW.restype = wintypes.LONG
    return api


def is_composing(widget):
    """True while the native IME still has uncommitted composition text."""
    if sys.platform != 'win32':
        return False
    try:
        hwnd = widget.winfo_toplevel().winfo_id()
        api = imm_api()
        context = api.ImmGetContext(hwnd)
        if not context:
            return False
        try:
            return api.ImmGetCompositionStringW(context, 0x0008, None, 0) > 0  # GCS_COMPSTR
        finally:
            api.ImmReleaseContext(hwnd, context)
    except (OSError, tk.TclError):
        return False


def sync_composition_font(widget):
    if sys.platform != 'win32':
        return False
    try:
        font = tkfont.Font(root=widget, font=widget.cget('font')).actual()
        logical = LOGFONTW()
        size = font['size']
        logical.lfHeight = -max(1, round(size * widget.winfo_fpixels('1p') if size > 0 else -size))
        logical.lfWeight = 700 if font['weight'] == 'bold' else 400
        logical.lfItalic = font['slant'] == 'italic'
        logical.lfCharSet = 1  # DEFAULT_CHARSET: let the chosen Unicode font resolve glyphs.
        logical.lfQuality = 5  # CLEARTYPE_QUALITY
        logical.lfFaceName = font['family'][:31]
        # Tk 8.6 positions its native composition window on the toplevel HWND,
        # even when the insertion cursor belongs to a child Text widget.
        hwnd = widget.winfo_toplevel().winfo_id()
        api = imm_api()
        context = api.ImmGetContext(hwnd)
        if not context:
            return False
        try:
            return bool(api.ImmSetCompositionFontW(context, ctypes.byref(logical)))
        finally:
            api.ImmReleaseContext(hwnd, context)
    except (OSError, tk.TclError):
        # IME availability must never prevent ordinary typing.
        return False
