"""Optional Qwen process adapter with a speech gate and bounded requests."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np

MAX_SAMPLES = 30 * 16000


def _create_worker_job(process):
    """Make Windows release even a native import stuck with the Python GIL."""
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes
    class Limits(ctypes.Structure):
        _fields_ = [('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),
            ('flags',wintypes.DWORD),('minimum',ctypes.c_size_t),('maximum',ctypes.c_size_t),
            ('active',wintypes.DWORD),('affinity',ctypes.c_size_t),
            ('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('basic',Limits),('io',ctypes.c_uint64*6),
            ('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
            ('peak_process_memory',ctypes.c_size_t),('peak_job_memory',ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=(ctypes.c_void_p,wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype=wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes=(wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD)
    kernel.SetInformationJobObject.restype=wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes=(wintypes.HANDLE,wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype=wintypes.BOOL
    kernel.IsProcessInJob.argtypes=(wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL))
    kernel.IsProcessInJob.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    kernel.CloseHandle.restype=wintypes.BOOL
    job=kernel.CreateJobObjectW(None,None)
    if not job:
        raise RuntimeError('无法创建识别进程的资源管理对象。')
    try:
        limits=ExtendedLimits();limits.basic.flags=0x2000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            raise RuntimeError('无法启用识别进程退出清理。')
        if not kernel.AssignProcessToJobObject(job,process._handle):
            raise RuntimeError('无法管理本次启动的识别进程。')
        return kernel,job
    except Exception:
        kernel.CloseHandle(job)
        raise


def _bind_actual_worker(job, owned):
    if job and owned:
        import ctypes
        from ctypes import wintypes
        kernel,job_handle=job
        active=wintypes.BOOL()
        if not kernel.IsProcessInJob(owned[1],job_handle,ctypes.byref(active)):
            raise RuntimeError('无法验证识别进程的资源绑定。')
        if not active.value and not kernel.AssignProcessToJobObject(job_handle,owned[1]):
            raise RuntimeError('无法绑定实际识别进程的退出清理。')


def _owned_worker_handle(process, pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid == os.getpid():
        raise RuntimeError('Qwen 引擎进程标识无效。')
    if os.name != 'nt':
        if pid != process.pid:
            raise RuntimeError('Qwen 引擎不是本次启动的进程。')
        return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel.TerminateProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100501, False, pid)  # sync/query/quota/terminate our child
    if not handle:
        raise RuntimeError('无法连接 Qwen 识别进程。')
    try:
        if pid != process.pid:
            class BasicInfo(ctypes.Structure):
                _fields_ = [('reserved1', ctypes.c_void_p), ('peb', ctypes.c_void_p),
                            ('reserved2', ctypes.c_void_p*2), ('pid', ctypes.c_size_t),
                            ('parent_pid', ctypes.c_size_t)]
            native = ctypes.WinDLL('ntdll')
            native.NtQueryInformationProcess.argtypes = (wintypes.HANDLE, wintypes.ULONG,
                ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
            native.NtQueryInformationProcess.restype = wintypes.LONG
            info, size = BasicInfo(), wintypes.ULONG()
            status = native.NtQueryInformationProcess(handle, 0, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(size))
            if status < 0 or info.parent_pid != process.pid:
                raise RuntimeError('Qwen 引擎不是本次启动的子进程。')
        return kernel, handle
    except Exception:
        kernel.CloseHandle(handle)
        raise


def _close_process(process, worker_state):
    # venv's Windows python.exe is a redirector: terminating Popen.pid alone
    # can leave the actual GPU process alive. Retain its verified OS handle.
    job = worker_state.pop('job', None)
    if job:
        job[0].CloseHandle(job[1])
    owned = worker_state.pop('handle', None)
    if owned:
        kernel, handle = owned
        try:
            if kernel.WaitForSingleObject(handle, 0) == 258:
                kernel.TerminateProcess(handle, 1)
                kernel.WaitForSingleObject(handle, 1000)
        finally:
            kernel.CloseHandle(handle)
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
    except OSError:
        pass
    for stream in (process.stdin, process.stdout):
        if stream:
            try:
                stream.close()
            except (OSError, ValueError):
                pass


class QwenModel:
    """Whisper-shaped interface; GPU libraries remain in the child process.

    The engine deployment is supplied explicitly. Model downloads, selecting
    this backend and installer integration are managed separately.
    Acoustic hints are currently disabled; caller-side correction rules still
    operate on output through RecognitionContext.normalize as for Whisper.
    """
    supports_acoustic_hints = False

    def __init__(self, engine_root=None, *, command=None, startup_timeout=120,
                 inference_timeout=60):
        if command is None:
            if engine_root is None:
                raise RuntimeError('尚未安装 Qwen 识别引擎。')
            root = Path(engine_root)
            executable = root / 'WenluQwen.exe'
            if not executable.is_file():
                raise RuntimeError('尚未安装 Qwen 识别引擎。')
            command = [str(executable), '--model-root', str(root / 'models')]
        command = list(command)
        if '--parent-pid' not in command:
            command += ['--parent-pid', str(os.getpid())]
        self.process = None
        self._messages = queue.Queue()
        self._lock = threading.Lock()
        self._sequence = 0
        self.inference_timeout = inference_timeout
        self._closed = False
        self._worker_state = {}
        try:
            environment = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT='1')
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=environment)
            self._finalizer = weakref.finalize(self, _close_process, self.process, self._worker_state)
            self._worker_state['job'] = _create_worker_job(self.process)
            process, messages = self.process, self._messages
            def read():
                try:
                    for line in process.stdout:
                        messages.put(('line', line))
                except (OSError, ValueError):
                    pass
                finally:
                    messages.put(('eof', None))
            self._reader = threading.Thread(target=read, daemon=True, name='qwen-output')
            self._reader.start()
            deadline = time.monotonic()+startup_timeout
            starting = self._receive(startup_timeout)
            if starting.get('type') != 'starting' or starting.get('protocol') != 1:
                raise RuntimeError(starting.get('message', 'Qwen 引擎未报告进程标识。'))
            self.worker_pid = starting.get('pid')
            self._worker_state['handle'] = _owned_worker_handle(self.process, self.worker_pid)
            _bind_actual_worker(self._worker_state['job'], self._worker_state['handle'])
            ready = self._receive(deadline-time.monotonic())
            if (ready.get('type') != 'ready' or ready.get('protocol') != 1 or
                    ready.get('device') != 'cuda' or ready.get('warmup') is not True):
                raise RuntimeError(ready.get('message', 'Qwen 引擎未通过启动验证。'))
            self.model = SimpleNamespace(device='cuda')
        except Exception:
            self.close()
            raise

    def _receive(self, timeout):
        try:
            kind, line = self._messages.get(timeout=max(.001, timeout))
        except queue.Empty:
            raise RuntimeError('Qwen 识别引擎响应超时，请检查显存或切换其他模型。') from None
        if kind == 'eof':
            raise RuntimeError('Qwen 识别进程已退出，请重新启动转写或切换模型。')
        try:
            response = json.loads(line)
            if not isinstance(response, dict):
                raise ValueError()
            return response
        except (ValueError, TypeError):
            raise RuntimeError('Qwen 引擎返回了无效数据。') from None

    def _request(self, value):
        if self._closed or self.process.poll() is not None:
            raise RuntimeError('Qwen 识别引擎已关闭，请重新启动转写。')
        self._sequence += 1
        value = dict(value, id=self._sequence)
        line = json.dumps(value)+'\n'
        deadline = time.monotonic()+self.inference_timeout
        sent = queue.Queue(maxsize=1)
        process = self.process
        def write():
            try:
                process.stdin.write(line)
                process.stdin.flush()
                sent.put(None)
            except (OSError, ValueError) as exc:
                sent.put(exc)
        writer = threading.Thread(target=write, daemon=True, name='qwen-input')
        writer.start()
        try:
            error = sent.get(timeout=max(.001, deadline-time.monotonic()))
            if error:
                raise RuntimeError('无法向 Qwen 识别引擎提交音频。') from error
            response = self._receive(deadline-time.monotonic())
            if response.get('type') == 'error':
                raise RuntimeError(response.get('message', 'Qwen 识别失败。'))
            if (response.get('type') != 'result' or response.get('id') != value['id'] or
                    response.get('sha256') != value['sha256']):
                raise RuntimeError('Qwen 识别结果与本次音频不匹配。')
            return response
        except queue.Empty:
            self.close()
            raise RuntimeError('Qwen 识别引擎接收音频超时。') from None
        except Exception:
            self.close()
            raise

    def transcribe(self, audio, language='zh', word_timestamps=True, **options):
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        if self.closed:
            raise RuntimeError('Qwen 识别引擎已关闭，请重新启动转写。')
        sound = np.asarray(audio, dtype='<f4')
        if sound.ndim != 1 or not 0 < len(sound) <= MAX_SAMPLES or not np.isfinite(sound).all():
            raise ValueError('识别音频必须是有效的 16 kHz 单声道样本，且不超过 30 秒。')
        languages = {'zh':'Chinese', 'en':'English', None:None}
        if language not in languages:
            raise ValueError('此识别引擎目前支持中文、英语或自动检测。')
        # An empty waveform can produce a fluent hallucination in Qwen. Keep
        # the same inner VAD parameters as decode_chunk even for direct calls.
        vad = VadOptions(**(options.get('vad_parameters') or dict(
            min_silence_duration_ms=600, min_speech_duration_ms=160, speech_pad_ms=100)))
        if not get_speech_timestamps(sound, vad):
            return iter(()), SimpleNamespace(language=language)
        with self._lock:
            data = sound.tobytes()
            response = self._request(dict(type='transcribe', language=languages[language],
                word_timestamps=bool(word_timestamps), samples=len(sound),
                audio=base64.b64encode(data).decode('ascii'), sha256=hashlib.sha256(data).hexdigest()))
        text, words = response.get('text'), response.get('words')
        if not isinstance(text, str) or (word_timestamps and not isinstance(words, list)):
            self.close()
            raise RuntimeError('Qwen 识别结果缺少文字或词时间戳。')
        if word_timestamps:
            duration, previous = len(sound)/16000, 0.
            try:
                for word in words:
                    start, end = word['start'], word['end']
                    if (not isinstance(word['word'], str) or not word['word'] or
                            not all(isinstance(t, (float, int)) and math.isfinite(t) for t in (start, end)) or
                            not 0 <= start <= end <= duration or start < previous-.001):
                        raise ValueError()
                    previous = end
                if any(char.isalnum() for char in text) and ''.join(w['word'] for w in words) != text:
                    raise ValueError()
            except (TypeError, ValueError, KeyError):
                self.close()
                raise RuntimeError('Qwen 词时间戳或文字完整性验证失败。') from None
            words = [SimpleNamespace(**word) for word in words]
        return iter([SimpleNamespace(text=text, words=words)] if text else []), SimpleNamespace(language=language)

    @property
    def closed(self):
        return self._closed or self.process is None or self.process.poll() is not None

    def close(self):
        self._closed = True
        finalizer = getattr(self, '_finalizer', None)
        if finalizer:
            finalizer()
        reader = getattr(self, '_reader', None)
        if reader and reader is not threading.current_thread():
            reader.join(timeout=.2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
