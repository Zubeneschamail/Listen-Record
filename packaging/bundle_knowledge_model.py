"""Stage only the three pinned public embedding files, never customer knowledge packages."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app_paths import MODELS
from knowledge_embedding import FILES, prepare_model, sha256


def bundle():
    target = ROOT / 'build' / 'bundled-knowledge-model'
    local = MODELS / 'bge-small-zh-v1.5'
    for name, expected in FILES.items():
        cached = local / name
        destination = target / name
        if cached.is_file() and sha256(cached) == expected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, destination)
    prepare_model(target)


if __name__ == '__main__':
    bundle()
