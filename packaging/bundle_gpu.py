"""Stage official NVIDIA wheels as an optional, application-local installer component."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {'cublas': ('nvidia-cublas-cu12', '12.4.5.8'),
            'cudnn': ('nvidia-cudnn-cu12', '9.1.0.70')}
REQUIRED = {'cublas64_12.dll', 'cublasLt64_12.dll', 'cudnn64_9.dll',
            'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll', 'cudnn_graph64_9.dll'}


def stage():
    wheels = ROOT / 'build/gpu-wheels'
    target = ROOT / 'build/gpu-runtime'
    wheels.mkdir(parents=True, exist_ok=True)
    files = {}
    for component, (package, version) in PACKAGES.items():
        matches = list(wheels.glob(package.replace('-', '_') + '-' + version + '-*win_amd64.whl'))
        if not matches:
            subprocess.run([sys.executable, '-m', 'pip', 'download', '--index-url', 'https://pypi.org/simple',
                            '--only-binary=:all:', '--no-deps', '--dest', str(wheels), package + '==' + version], check=True)
            matches = list(wheels.glob(package.replace('-', '_') + '-' + version + '-*win_amd64.whl'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected one Windows x64 wheel for {package}')
        licenses = 0
        with zipfile.ZipFile(matches[0]) as archive:
            for member in archive.namelist():
                path = PurePosixPath(member)
                if len(path.parts) == 4 and path.parts[:3] == ('nvidia', component, 'bin') and path.suffix == '.dll':
                    destination = target.joinpath(*path.parts)
                elif 'license' in path.name.lower() and not member.endswith('/'):
                    destination = target / 'licenses' / component / path.name
                    licenses += 1
                else:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(member)
                destination.write_bytes(data)
                files[str(destination.relative_to(target)).replace('\\', '/')] = hashlib.sha256(data).hexdigest()
        if not licenses:
            raise RuntimeError(f'NVIDIA license missing from {package}')
    if not REQUIRED.issubset({Path(name).name for name in files}):
        raise RuntimeError('GPU runtime is incomplete')
    (target / 'manifest.json').write_text(json.dumps({'packages': PACKAGES, 'sha256': files}, indent=2), encoding='utf-8')
    print(f'Optional GPU runtime ready: {sum(p.stat().st_size for p in target.rglob("*.dll")) / 1024**2:.0f} MiB')


if __name__ == '__main__':
    stage()
