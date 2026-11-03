"""Discovery and integrity checks for the optional offline recognition engine."""
import hashlib
import json
from pathlib import Path, PurePosixPath

QWEN_MODEL = 'qwen3-asr'
QWEN_LABEL = 'qwen3-asr · 中文准确'
MODEL_LABELS = ('small · 轻量准确', 'large-v3-turbo · 高性能', 'base · 均衡', 'tiny · 更快', QWEN_LABEL)
ASR_REVISION = '7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c'
ALIGNER_REVISION = 'c07281df297b9905d24a508279258cccf987a064'
REQUIRED_FILES = ('WenluQwen.exe', '_internal/python312.dll',
    '_internal/torch/lib/torch_cuda.dll') + tuple(
        f'models/{model}/{name}'
        for model in ('qwen3-asr-0.6b', 'qwen3-forced-aligner-0.6b')
        for name in ('model.safetensors', 'config.json', 'processor_config.json',
                     'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'README.md'))
INSTALL_HELP = ('请运行包含“中文准确识别”组件的完整版离线安装包，并选中该组件。'
                '此模型需要支持 BF16 的 NVIDIA 显卡，建议 8 GB 显存；也可选择现有 Whisper 模型。')
QWEN_HELP = ('偏重中文和中英技术术语，速度较慢；需要支持 BF16 的 NVIDIA 显卡。'
             '普通热词不参与本模型解码，纠错规则仍然生效。')


def complete_engine(folder, verify_hashes=False):
    folder = Path(folder)
    try:
        manifest = folder/'engine.json'
        if manifest.stat().st_size > 2*1024*1024:
            return False
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if (info['protocol'] != 1 or info['asr_revision'] != ASR_REVISION or
                info['aligner_revision'] != ALIGNER_REVISION):
            return False
        files = info['files']
        if not isinstance(files, dict) or not set(REQUIRED_FILES).issubset(files) or len(files) > 10000:
            return False
        root = folder.resolve()
        for relative in files:
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or ':' in relative or '\\' in relative:
                return False
        # The UI checks only essential presence/sizes. Full hashing of the
        # multi-GB engine is an explicit build/install verification step.
        for relative in files if verify_hashes else REQUIRED_FILES:
            item = files[relative]
            path = PurePosixPath(relative)
            file = folder/path
            if not file.resolve().is_relative_to(root):
                return False
            if not file.is_file() or file.stat().st_size != item['bytes']:
                return False
            if verify_hashes:
                with file.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def engine_root(resources=None, data=None):
    if resources is None or data is None:
        from app_paths import RESOURCES, DATA
        resources = RESOURCES if resources is None else resources
        data = DATA if data is None else data
    for folder in (Path(resources)/'qwen-engine', Path(data)/'engines/qwen'):
        if complete_engine(folder):
            return folder
    return None


def default_model_label():
    return QWEN_LABEL if engine_root() is not None else MODEL_LABELS[0]
