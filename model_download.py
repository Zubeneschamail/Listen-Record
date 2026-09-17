"""Offline cache discovery and an independently cancellable download worker."""
from pathlib import Path
from app_paths import MODELS, RESOURCES, legacy_models

BUNDLED_MODELS = RESOURCES / 'bundled-models'

MODEL_FILES = ['config.json', 'preprocessor_config.json', 'model.bin', 'tokenizer.json', 'vocabulary.*']


def progress_text(completed, total):
    completed = max(0, min(completed, total))
    unit, scale = ('GiB', 1024**3) if total >= 1024**3 else ('MiB', 1024**2)
    percent = completed / total * 100 if total else 0
    return f'{percent:.1f}% · {completed / scale:.2f} / {total / scale:.2f} {unit}'


def complete_model(path):
    required = ('model.bin', 'config.json', 'tokenizer.json')
    return (all((path / file).is_file() and (path / file).stat().st_size > 0 for file in required)
            and any(file.stat().st_size for file in path.glob('vocabulary.*') if file.is_file()))


def cached_model(name):
    from faster_whisper.utils import _MODELS
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    bundled = BUNDLED_MODELS / name
    if name in _MODELS and complete_model(bundled):
        return bundled
    for cache in dict.fromkeys((MODELS, legacy_models())):
        try:
            path = Path(snapshot_download(_MODELS[name], cache_dir=str(cache), local_files_only=True,
                                          allow_patterns=MODEL_FILES))
            if complete_model(path):
                return path
        except (LocalEntryNotFoundError, OSError):
            pass
        # Downloads pinned to a commit can be complete without a refs/main file.
        snapshots = cache / ('models--' + _MODELS[name].replace('/', '--')) / 'snapshots'
        for snapshot in sorted(snapshots.glob('*'), key=lambda item: item.stat().st_mtime, reverse=True):
            if snapshot.is_dir() and complete_model(snapshot):
                return snapshot
    return None


def download_worker(name, pipe):
    """Spawned process: closing the dialog can stop network waits immediately."""
    import io
    import logging
    import time
    import threading
    from huggingface_hub import HfApi, snapshot_download
    from faster_whisper import WhisperModel
    from faster_whisper.utils import _MODELS
    from tqdm.auto import tqdm
    try:
        path = cached_model(name)
        if path is None:
            pipe.send(('connecting', '正在连接模型服务…'))
            info = HfApi().model_info(_MODELS[name], timeout=10)
            pipe.send(('progress', '正在下载模型文件，可随时取消…'))
            class Progress(tqdm):
                def __init__(self, *args, **kwargs):
                    self.report_bytes = kwargs.get('unit') == 'B' and 'transfer' not in kwargs.get('name', '') and kwargs.get('desc') != 'Downloading bytes'
                    self.sent_at = 0
                    self.report_lock = threading.Lock()
                    kwargs['file'] = io.StringIO()  # windowed executables have no stderr
                    super().__init__(*args, **kwargs)
                def update(self, n=1):
                    result = super().update(n)
                    if self.report_bytes and self.total:
                        with self.report_lock:
                            now = time.monotonic()
                            if now - self.sent_at >= 0.15 or self.n >= self.total:
                                pipe.send(('bytes', (min(self.n, self.total), self.total)))
                                self.sent_at = now
                    return result
            path = snapshot_download(_MODELS[name], revision=info.sha, cache_dir=str(MODELS),
                                     etag_timeout=10, tqdm_class=Progress,
                                     allow_patterns=MODEL_FILES)
        pipe.send(('progress', '正在校验模型…'))
        model = WhisperModel(str(path), device='cpu', compute_type='int8', local_files_only=True)
        del model
        pipe.send(('done', '模型已就绪，可离线使用。'))
    except Exception as exc:
        logging.exception('Model download or validation failed')
        pipe.send(('error', '下载或校验失败，请检查网络后重试。\n' + str(exc)[:180]))
    finally:
        pipe.close()
