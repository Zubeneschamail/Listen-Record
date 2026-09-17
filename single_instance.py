"""One app per Windows user; a second launch signals the running UI."""
import ctypes
from ctypes import wintypes

class SingleInstance:
    def __init__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.api.CreateMutexW.restype = wintypes.HANDLE
        self.api.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        self.api.CreateEventW.restype = wintypes.HANDLE
        for name in ('SetEvent', 'CloseHandle'):
            getattr(self.api, name).argtypes = [wintypes.HANDLE]
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        import getpass, hashlib
        suffix = hashlib.sha256(getpass.getuser().encode()).hexdigest()[:16]
        self.mutex = self.api.CreateMutexW(None, False, 'Local\\Wenlu.App.' + suffix)
        if not self.mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        self.duplicate = ctypes.get_last_error() == 183
        self.guard = self.api.CreateMutexW(None, False, "Local\\Wenlu.InstallerGuard")
        self.event = self.api.CreateEventW(None, False, False, 'Local\\Wenlu.Activate.' + suffix)
        if self.duplicate:
            self.api.SetEvent(self.event)

    def poll(self, root):
        if self.api.WaitForSingleObject(self.event, 0) == 0:
            root.deiconify()
            root.lift()
            root.focus_force()
        root.after(250, self.poll, root)

    def close(self):
        self.api.CloseHandle(self.guard)
        self.api.CloseHandle(self.event)
        self.api.CloseHandle(self.mutex)
