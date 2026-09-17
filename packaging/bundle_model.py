"""Stage a pinned, validated small model before freezing the offline installer."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_paths import MODELS, legacy_models
from model_download import MODEL_FILES, complete_model

REPO = 'Systran/faster-whisper-small'
REVISION = '536b0662742c02347bc0e980a01041f333bce120'


def prepare():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    source = None
    for cache in dict.fromkeys((MODELS, legacy_models())):
        try:
            candidate = Path(snapshot_download(REPO, revision=REVISION, cache_dir=str(cache),
                                              local_files_only=True, allow_patterns=MODEL_FILES))
            if complete_model(candidate):
                source = candidate
                break
        except (OSError, LocalEntryNotFoundError):
            pass
    if source is None:
        source = Path(snapshot_download(REPO, revision=REVISION, cache_dir=str(MODELS),
                                        allow_patterns=MODEL_FILES, etag_timeout=15))
    if not complete_model(source):
        raise RuntimeError('Bundled small model is incomplete; refusing to build an offline installer')
    target = ROOT / 'build/bundled-models/small'
    target.mkdir(parents=True, exist_ok=True)
    for pattern in MODEL_FILES:
        for file in source.glob(pattern):
            shutil.copy2(file, target / file.name, follow_symlinks=True)
    (target / 'source.json').write_text(json.dumps({'repository': REPO, 'revision': REVISION}), encoding='utf-8')
    from faster_whisper import WhisperModel
    model = WhisperModel(str(target), device='cpu', compute_type='int8', local_files_only=True)
    del model
    print(f'Bundled small model ready: {sum(p.stat().st_size for p in target.iterdir()) / 1024**2:.1f} MiB')


if __name__ == '__main__':
    prepare()
