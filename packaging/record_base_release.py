"""Record the validated reusable program build, without customer data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from version import VERSION


def main():
    executable = ROOT / 'dist/Wenlu/Wenlu.exe'
    with executable.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    (executable.parent / 'base-release.json').write_text(json.dumps({
        'version': VERSION, 'delivery_protocol': 1, 'exe_sha256': digest
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
