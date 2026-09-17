"""Read-only application resources and persistent per-user data."""
import json
import os
from pathlib import Path
import shutil
import sys

RESOURCES = Path(__file__).resolve().parent
DATA = Path(os.environ.get('WENLU_DATA_DIR', str(Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Wenlu')))
DATA.mkdir(parents=True, exist_ok=True)
MODELS = DATA / 'models'
RECORDINGS = DATA / 'recordings'
LOGS = DATA / 'logs'
for directory in (MODELS, RECORDINGS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)


def migrate_legacy(source=RESOURCES, target=DATA):
    """Copy missing user data, never overwrite newer data or delete originals."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        return
    target.mkdir(parents=True, exist_ok=True)
    if (source / 'models').is_dir() and not (target / 'legacy-model-cache.json').exists():
        (target / 'legacy-model-cache.json').write_text(json.dumps(str(source / 'models')), encoding='utf-8')
    for name in ('recognition-settings.json', 'recordings'):
        old, new = source / name, target / name
        if old.is_file() and not new.exists():
            shutil.copy2(old, new)
        elif old.is_dir():
            new.mkdir(exist_ok=True)
            for file in old.iterdir():
                if file.is_file() and file.suffix in ('.txt', '.jsonl') and not (new / file.name).exists():
                    shutil.copy2(file, new / file.name)


def preferences():
    try:
        value = json.loads((DATA / 'desktop-settings.json').read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_desktop(values):
    temporary = DATA / 'desktop-settings.tmp'
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(DATA / 'desktop-settings.json')


def legacy_models():
    try:
        value = json.loads((DATA / 'legacy-model-cache.json').read_text(encoding='utf-8'))
        return Path(value) if isinstance(value, str) and Path(value).is_dir() else RESOURCES / 'models'
    except (OSError, ValueError):
        return RESOURCES / 'models'
