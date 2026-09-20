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
        # CUDA libraries are loaded lazily: actually execute before opening audio streams.
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
