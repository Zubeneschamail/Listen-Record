# Build using a separate pinned PyTorch/Transformers environment.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = []
for package in ('transformers', 'tokenizers', 'librosa', 'certifi'):
    datas += collect_data_files(package)
for distribution in ('torch', 'transformers', 'accelerate', 'safetensors', 'tokenizers',
                     'huggingface-hub', 'librosa', 'numpy', 'soundfile'):
    datas += copy_metadata(distribution)
hidden = collect_submodules('transformers.models.qwen3_asr')
hidden += ['transformers.models.qwen2.tokenization_qwen2']
a = Analysis([str(root/'qwen_worker.py')], pathex=[str(root)], datas=datas,
    hiddenimports=hidden, excludes=['tensorflow', 'torchvision', 'torchaudio', 'matplotlib',
        'pytest', 'tkinter', 'pyaudiowpatch', 'faster_whisper', 'ctranslate2'], noarchive=False)
# Keep this opt-in observer ahead of the entry script without importing GPU
# packages into the desktop app or changing the recognizer's model profile.
a.scripts.insert(0, ('qwen_diagnostics', str(root/'packaging/qwen_diagnostics.py'), 'PYSOURCE'))
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='WenluQwen', console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name='WenluQwen', upx=False)
