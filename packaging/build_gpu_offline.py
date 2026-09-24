"""Build a self-contained GPU add-on from hash-pinned official Windows wheels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import os

import bundle_gpu

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Wenlu-GPU-Offline-cu12.4-cudnn9.1-win64'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(compiler):
    # Use the same URL/hash pins as the online installer; reject altered cache.
    pins = (ROOT / 'packaging/gpu-components.iss').read_text(encoding='utf-8')
    for key in ('GpuCublasUrl', 'GpuCudnnUrl', 'GpuNvrtcUrl'):
        url = re.search(r'#define ' + key + r' "([^"]+)"', pins)[1]
        expected = re.search(r'Source: "\{#' + key + r'\}";[^\n]+Hash: "([a-f0-9]+)"', pins)[1]
        wheel = ROOT / 'build/gpu-wheels' / url.rsplit('/', 1)[-1]
        if not wheel.is_file():
            raise RuntimeError('Missing cached wheel: ' + str(wheel))
        if sha256(wheel) != expected:
            raise RuntimeError('Wheel SHA256 mismatch: ' + wheel.name)
        print('Verified ' + wheel.name, flush=True)
    bundle_gpu.stage()
    runtime = ROOT / 'build/gpu-runtime'
    manifest = json.loads((runtime / 'manifest.json').read_text(encoding='utf-8'))
    entries = []
    # Only include files freshly emitted by stage(), never old staging residue.
    for name in [*manifest['sha256'], 'manifest.json']:
        source = runtime / name
        if name != 'manifest.json' and sha256(source) != manifest['sha256'][name]:
            raise RuntimeError('Staged file SHA256 mismatch: ' + name)
        destination = '{app}\\_internal\\gpu-runtime'
        if Path(name).parent != Path('.'):
            destination += '\\' + str(Path(name).parent)
        entries.append(f'Source: "{source}"; DestDir: "{destination}"; Flags: ignoreversion')
    include = ROOT / 'build/gpu-offline-files.iss'
    include.write_text('\n'.join(entries) + '\n', encoding='utf-8-sig')
    subprocess.run([str(compiler), '/Q', '/DRuntimeFiles=' + str(include),
                    str(ROOT / 'packaging/gpu-offline.iss')], check=True)
    artifact = ROOT / 'release' / (NAME + '.exe')
    artifact.with_suffix('.exe.sha256').write_text(sha256(artifact) + '  ' + artifact.name + '\n', encoding='ascii')
    print(f'Built {artifact} ({artifact.stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iscc', type=Path, default=Path(os.environ['LOCALAPPDATA']) / 'Programs/Inno Setup 6/ISCC.exe')
    build(parser.parse_args().iscc)
