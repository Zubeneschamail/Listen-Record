"""Install the real offline EXE in an isolated app copy, then run CUDA inference."""
import hashlib
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    installer = ROOT / 'release/Wenlu-GPU-Offline-cu12.4-cudnn9.1-win64.exe'
    folder = Path(tempfile.mkdtemp(prefix='gpu-offline-check-', dir=ROOT / 'build'))
    app = folder / 'Wenlu'
    shutil.copytree(ROOT / 'dist/Wenlu', app,
                    ignore=shutil.ignore_patterns('gpu-runtime', 'nvidia'))
    exe_hash = digest(app / 'Wenlu.exe')
    env = dict(os.environ, WENLU_DATA_DIR=str(folder / 'user-data'),
               HF_HUB_OFFLINE='1', HTTP_PROXY='http://127.0.0.1:1',
               HTTPS_PROXY='http://127.0.0.1:1', ALL_PROXY='http://127.0.0.1:1')
    env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep)
                                if not any(s in p.lower() for s in ('nvidia', 'cuda', 'site-packages')))
    flags = subprocess.CREATE_NO_WINDOW
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    for name, destination in [('invalid', folder / 'invalid'), ('locked', app), ('valid', app)]:
        handle = None
        if name == 'locked':
            handle = api.CreateFileW(str(app / 'Wenlu.exe'), 0x80000000, 1, None, 3, 0x80, None)
            assert handle != ctypes.c_void_p(-1).value
        try:
            result = subprocess.run([str(installer), '/VERYSILENT', '/SUPPRESSMSGBOXES',
                '/NORESTART', '/DIR=' + str(destination), '/LOG=' + str(folder / (name + '.log'))],
                env=env, timeout=240, creationflags=flags)
        finally:
            if handle is not None:
                api.CloseHandle(handle)
        if name == 'invalid':
            assert result.returncode != 0, 'Invalid target was accepted'
            assert not list(destination.rglob('*.dll')), 'Invalid target was modified'
        elif name == 'locked':
            assert result.returncode != 0, 'Locked application was accepted'
            assert not (app / '_internal/gpu-runtime').exists()
        else:
            assert result.returncode == 0, f'Installer failed: {result.returncode}; see {folder}'
        print(f'PASS {name}: exit={result.returncode}', flush=True)
    runtime = app / '_internal/gpu-runtime'
    manifest = json.loads((runtime / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        assert digest(runtime / name) == expected, name
    assert digest(app / 'Wenlu.exe') == exe_hash, 'Main application was changed'
    report = folder / 'gpu-self-test.json'
    subprocess.run([str(app / 'Wenlu.exe'), '--self-test', str(report), '--verify-gpu'],
                   env=env, timeout=180, check=True, creationflags=flags)
    value = json.loads(report.read_text(encoding='utf-8'))
    assert value['ok'] and value['gpu_verified'], value
    print(json.dumps({'installed_files_verified': len(manifest['sha256']),
                      'gpu': value, 'evidence': str(folder)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
