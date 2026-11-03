"""Local GPU inference with validated CPU fallback."""
import logging
import os
from pathlib import Path
import sys
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel
from app_paths import MODELS, RESOURCES

ROOT = Path(__file__).resolve().parent
_DLL_HANDLES = []
_CUDA_LIBRARIES = []

def prepare_cuda():
    if os.name == 'nt' and not _DLL_HANDLES:
        directories = list((Path(sys.prefix) / 'Lib/site-packages/nvidia').glob('*/bin'))
        directories += list((RESOURCES / 'nvidia').glob('*/bin'))
        directories += list((RESOURCES / 'gpu-runtime/nvidia').glob('*/bin'))
        for directory in directories:
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            # CTranslate2 uses native LoadLibrary, which also needs the process PATH.
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')

def cuda_runtime_errors():
    """Return failing library names and loader errors, including dependencies."""
    if os.name != 'nt' or _CUDA_LIBRARIES:
        return []
    import ctypes
    libraries, errors = [], []
    for name in ('cublasLt64_12.dll', 'cublas64_12.dll', 'cudnn64_9.dll'):
        try:
            libraries.append(ctypes.WinDLL(name))
        except OSError as exc:
            errors.append(f'{name}：{exc}')
    if not errors:
        _CUDA_LIBRARIES.extend(libraries)
    return errors


def cuda_runtime_available():
    """Do not initialize a CUDA model when its delayed-load libraries are absent."""
    errors = cuda_runtime_errors()
    if errors:
        logging.info('CUDA runtime unavailable; using CPU: %s', '; '.join(errors))
    return not errors


def load_model(name, device='auto', status=None):
    from qwen_engine import QWEN_MODEL, INSTALL_HELP, engine_root
    if name == QWEN_MODEL:
        if device == 'cpu':
            raise RuntimeError('中文准确模型需要 NVIDIA GPU；CPU 转录请选择 small 或其他 Whisper 模型。')
        root = engine_root()
        if root is None:
            raise RuntimeError(INSTALL_HELP)
        if status:
            status('正在加载中文准确模型与词时间戳模型 · GPU…')
        from qwen_runtime import QwenModel
        return QwenModel(root)
    def create(target):
        options = dict(device=target, compute_type='float16' if target == 'cuda' else 'int8',
                       cpu_threads=6, num_workers=1, download_root=str(MODELS))
        from model_download import cached_model
        cached = cached_model(name)
        if cached is None:
            raise RuntimeError(f'本机尚未下载 {name} 模型，请在“模型管理”中下载，或选择已有模型。')
        if status:
            status(f'正在加载 {name} · {"GPU" if target == "cuda" else "CPU"}…')
        logging.info('Loading recognition model: %s / %s', target, name)
        model = WhisperModel(str(cached), local_files_only=True, **options)
        # CUDA libraries are lazy: validate inference before decoding queued audio.
        if status:
            status(f'正在初始化 {name} · {"GPU" if target == "cuda" else "CPU"}…')
        segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language='zh',
                                       beam_size=1, vad_filter=False, max_new_tokens=1)
        list(segments)
        logging.info('Recognition backend: %s / %s', target, name)
        return model
    if device == 'cpu':
        return create('cpu')
    try:
        prepare_cuda()
        count = ctranslate2.get_cuda_device_count()
        if device == 'cuda' and count and (errors := cuda_runtime_errors()):
            raise RuntimeError('CUDA 运行库加载失败：' + '; '.join(errors))
        if count and cuda_runtime_available():
            return create('cuda')
        if device == 'cuda':
            raise RuntimeError('未检测到 CUDA 显卡')
    except Exception:
        if device != 'auto':
            raise
        logging.exception('GPU initialization failed; falling back to CPU')
    return create('cpu')


def load_capture_model(name, device='auto', status=None):
    """Capture sessions may end while their warm model remains cached."""
    from qwen_engine import QWEN_MODEL
    if name == QWEN_MODEL:
        return load_model(name, device=device, status=status)
    from whisper_runtime import WhisperCaptureModel
    return WhisperCaptureModel(lambda: load_model(name, device=device, status=status))


def release_model(model):
    """Release either a native Whisper model or a separate GPU engine."""
    if model is None:
        return
    close = getattr(model, 'close', None)
    if callable(close):
        close()
    else:
        backend = getattr(model, 'model', None)
        unload = getattr(backend, 'unload_model', None)
        if callable(unload):
            unload()
