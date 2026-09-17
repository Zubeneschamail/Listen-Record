# Build using: python -m PyInstaller packaging/wenlu.spec --noconfirm
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata
from pathlib import Path
root = Path(SPECPATH).parent
assets = [(str(root / 'assets'), 'assets')]
bundled = root / 'build/bundled-models/small'
if not (bundled / 'model.bin').is_file():
    raise RuntimeError('Run packaging/bundle_model.py before building the offline installer')
assets.append((str(bundled), 'bundled-models/small'))
for package in ['faster_whisper', 'opencc', 'certifi']:
    assets += collect_data_files(package)
for package in ['faster-whisper', 'huggingface-hub', 'tokenizers', 'tqdm']:
    assets += copy_metadata(package)
binaries = []
for package in ['ctranslate2', 'onnxruntime', 'pyaudiowpatch']:
    binaries += collect_dynamic_libs(package)
a = Analysis([str(root / 'launcher.py')], pathex=[str(root)], binaries=binaries, datas=assets,
             hiddenimports=['pystray._win32', 'PIL.Image', 'scipy.special._cdflib'],
             excludes=['torch', 'tensorflow', 'matplotlib', 'pytest'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Wenlu', console=False,
          icon=str(root / 'assets/wenlu.ico'), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name='Wenlu', upx=False)
