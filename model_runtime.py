"""Local GPU inference with validated CPU fallback."""
import logging
import os
from pathlib import Path
import sys
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel
from huggingface_hub.errors import LocalEntryNotFoundError

ROOT = Path(__file__).resolve().parent
_DLL_HANDLES = []

def prepare_cuda():
    if os.name == 'nt' and not _DLL_HANDLES:
        for directory in (Path(sys.prefix) / 'Lib/site-packages/nvidia').glob('*/bin'):
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            # CTranslate2 uses native LoadLibrary, which also needs the process PATH.
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')

def load_model(name, device='auto'):
    def create(target):
        options = dict(device=target, compute_type='float16' if target == 'cuda' else 'int8',
                       cpu_threads=6, num_workers=1, download_root=str(ROOT / 'models'))
        try:
            model = WhisperModel(name, local_files_only=True, **options)
        except LocalEntryNotFoundError:
            model = WhisperModel(name, **options)
        # CUDA libraries are loaded lazily: actually execute before opening audio streams.
        segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language='zh',
                                       beam_size=1, vad_filter=False, max_new_tokens=1)
        list(segments)
        logging.info('Recognition backend: %s / %s', target, name)
        return model
    if device == 'cpu':
        return create('cpu')
    try:
        prepare_cuda()
        if ctranslate2.get_cuda_device_count():
            return create('cuda')
        if device == 'cuda':
            raise RuntimeError('未检测到 CUDA 显卡')
    except Exception:
        if device != 'auto':
            raise
        logging.exception('GPU initialization failed; falling back to CPU')
    return create('cpu')
