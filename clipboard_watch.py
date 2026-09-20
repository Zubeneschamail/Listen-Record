"""Opt-in Windows clipboard snapshots. Never calls Tk from a worker."""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import io
import os
import threading

from PIL import Image, ImageGrab


@dataclass(frozen=True)
class ClipboardItem:
    text: str
    image: bytes | None = None

    @property
    def fingerprint(self):
        return hashlib.sha256(self.image if self.image else self.text.encode('utf-8')).digest()


def image_item(image):
    if image.width * image.height > 25_000_000:
        raise ValueError('剪贴板图片超过 2500 万像素，请缩小后重新复制。')
    image = image.convert('RGB')
    image.thumbnail((4096, 4096))
    output = io.BytesIO()
    image.save(output, format='PNG')
    data = output.getvalue()
    if len(data) > 8 * 1024 * 1024:
        raise ValueError('剪贴板图片超过 8 MB，请缩小后重新复制。')
    return ClipboardItem(f'【剪贴板图片 {image.width}×{image.height}】请识别图片内容；'
                         '如有问题请直接解答，否则概括要点。', data)


class WindowsClipboard:
    def __init__(self):
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.user.GetClipboardSequenceNumber.restype = wintypes.DWORD
        self.user.GetClipboardOwner.restype = wintypes.HWND
        self.user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user.GetClipboardData.argtypes = [wintypes.UINT]
        self.user.GetClipboardData.restype = wintypes.HANDLE
        self.user.OpenClipboard.argtypes = [wintypes.HWND]
        self.user.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        self.png_format = self.user.RegisterClipboardFormatW('PNG')
        self.kernel.GlobalLock.argtypes = [wintypes.HGLOBAL]
        self.kernel.GlobalLock.restype = ctypes.c_void_p
        self.kernel.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        self.kernel.GlobalSize.argtypes = [wintypes.HGLOBAL]
        self.kernel.GlobalSize.restype = ctypes.c_size_t

    def sequence(self):
        return self.user.GetClipboardSequenceNumber()

    def owned_by_app(self):
        pid = wintypes.DWORD()
        owner = self.user.GetClipboardOwner()
        if owner:
            self.user.GetWindowThreadProcessId(owner, ctypes.byref(pid))
        return pid.value == os.getpid()

    def read(self):
        # ImageGrab supports bitmap and registered PNG clipboard formats on Windows.
        if any(self.user.IsClipboardFormatAvailable(fmt) for fmt in (8, 17, self.png_format)):
            image = ImageGrab.grabclipboard()
            if isinstance(image, Image.Image):
                return image_item(image)
            raise OSError('图片尚未准备好')
        if not self.user.OpenClipboard(None):
            raise OSError('剪贴板暂时被占用')
        try:
            if not self.user.IsClipboardFormatAvailable(13):  # CF_UNICODETEXT
                return None
            handle = self.user.GetClipboardData(13)
            if not handle:
                raise OSError('文字尚未准备好')
            size = self.kernel.GlobalSize(handle)
            if size > 200_000:
                raise ValueError('剪贴板文字超过 2400 字，请缩短后重新复制。')
            pointer = self.kernel.GlobalLock(handle)
            if not pointer:
                raise OSError('无法读取剪贴板')
            try:
                text = ctypes.string_at(pointer, size).decode('utf-16-le').split('\0', 1)[0].strip()
            finally:
                self.kernel.GlobalUnlock(handle)
            if len(text) > 2400:
                raise ValueError('剪贴板文字超过 2400 字，请缩短后重新复制。')
            return ClipboardItem(text) if text else None
        finally:
            self.user.CloseClipboard()


class ClipboardWatcher:
    def __init__(self, events, source=None):
        self.events = events
        self.source = source or WindowsClipboard()
        self.generation = 0
        self.cancel = threading.Event()
        self.repeat = threading.Event()

    def allow_repeat(self):
        self.repeat.set()

    def stop(self):
        self.cancel.set()
        self.generation += 1

    def start(self):
        self.stop()
        cancel = self.cancel = threading.Event()
        repeat = self.repeat = threading.Event()
        generation = self.generation
        baseline = self.source.sequence()

        def watch():
            seen = candidate = baseline
            fingerprint = None
            failures = 0
            while not cancel.wait(.2):
                sequence = self.source.sequence()
                if not sequence or sequence == seen:
                    continue
                if sequence != candidate:
                    candidate, failures = sequence, 0
                    continue  # wait for one stable sample after a copy
                try:
                    item = None if self.source.owned_by_app() else self.source.read()
                    # A newer copy won the race; never label old data as the new copy.
                    if sequence != self.source.sequence():
                        continue
                    seen = sequence
                    if repeat.is_set():
                        fingerprint = None
                        repeat.clear()
                    if item is not None and item.fingerprint != fingerprint:
                        fingerprint = item.fingerprint
                        self.events.put(('clipboard', (generation, item, '')))
                except OSError:
                    failures += 1
                    if failures >= 10:
                        seen = sequence
                        self.events.put(('clipboard', (generation, None, '剪贴板读取失败，请重新复制。')))
                except (ValueError, Image.DecompressionBombError):
                    seen = sequence
                    self.events.put(('clipboard', (generation, None,
                        '剪贴板内容无法发送：文字限 2400 字，图片限 2500 万像素和 8 MB。')))

        threading.Thread(target=watch, daemon=True, name='clipboard-watch').start()
