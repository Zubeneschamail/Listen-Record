"""Pinned BGE ONNX encoder; documents and queries use the same fingerprint."""
from pathlib import Path
import hashlib
import os
import threading

import httpx

from app_paths import RESOURCES, MODELS
BUNDLED_MODEL_DIR = RESOURCES / 'bundled-knowledge-model'
STAGED_MODEL_DIR = RESOURCES / 'build' / 'bundled-knowledge-model'
DEFAULT_MODEL_DIR = (BUNDLED_MODEL_DIR if BUNDLED_MODEL_DIR.is_dir() else
                     STAGED_MODEL_DIR if STAGED_MODEL_DIR.is_dir() else
                     MODELS / 'bge-small-zh-v1.5')
REPO = 'Xenova/bge-small-zh-v1.5'
REVISION = '75c43b069aac4d136ba6bc1122f995fedcfd2781'
FILES = {
    'tokenizer.json': '48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26',
    'config.json': 'd4193ead3a810fd694fa8a31d7fc72fbaebc0668b603e398734bf2f6538ff42f',
    'onnx/model_quantized.onnx': '15b717c382bcb518ba457b93ea6850ede7f4f1cd8937454aa06972366cd19bcc',
}
QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'


class Cancelled(RuntimeError):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled('已取消知识包检索。')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_model(directory=DEFAULT_MODEL_DIR, progress=print, cancel=None):
    """Explicit network operation. Build/search never download implicitly."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=15)) as client:
        for name, expected in FILES.items():
            check_cancel(cancel)
            target = directory / name
            if target.is_file() and sha256(target) == expected:
                progress(f'已校验：{name}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
            try:
                progress(f'下载模型文件：{name}')
                url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    size = int(response.headers.get('content-length', 0))
                    done, previous_mb = 0, -1
                    with temporary.open('wb') as stream:
                        for block in response.iter_bytes(256 * 1024):
                            check_cancel(cancel)
                            stream.write(block)
                            done += len(block)
                            mb = done // (1024 * 1024)
                            if mb != previous_mb:
                                progress(f'{name}：{done / 1048576:.1f} MB' +
                                         (f' / {size / 1048576:.1f} MB' if size else ''))
                                previous_mb = mb
                if sha256(temporary) != expected:
                    raise ValueError(f'模型文件校验失败：{name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    progress('模型已下载并通过 SHA256 校验，可以离线构建。')


class Encoder:
    dimension = 512
    max_tokens = 512

    def __init__(self, directory=DEFAULT_MODEL_DIR, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        directory = Path(directory)
        for name, expected in FILES.items():
            path = directory / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError('知识包向量模型缺失或校验失败，请运行 prepare_knowledge_model.py 或重新安装闻录。')
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'onnx/model_quantized.onnx'),
                                            sess_options=options, providers=['CPUExecutionProvider'])
        self.lock = threading.Lock()
        self.signature = {
            'model': 'BAAI/bge-small-zh-v1.5', 'onnx_repository': REPO,
            'revision': REVISION, 'files': FILES, 'dimension': self.dimension,
            'pooling': 'cls', 'normalization': 'l2', 'precision': 'dynamic-int8',
            'query_prefix': QUERY_PREFIX, 'max_tokens': self.max_tokens,
        }

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def windows(self, text, budget, overlap):
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        offsets = encoded.offsets
        if not offsets:
            return []
        result = []
        for start in range(0, len(offsets), budget - overlap):
            stop = min(start + budget, len(offsets))
            a, b = offsets[start][0], offsets[stop - 1][1]
            result.append((text[a:b], a, b))
            if stop == len(offsets):
                break
        return result

    def encode(self, texts, query=False, batch_size=8, cancel=None, progress=lambda _: None):
        import numpy as np
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = []
        for offset in range(0, len(texts), batch_size):
            check_cancel(cancel)
            batch = [(QUERY_PREFIX + text if query else text) for text in texts[offset:offset + batch_size]]
            with self.lock:
                encoded = self.tokenizer.encode_batch(batch)
                if any(len(item.ids) > self.max_tokens for item in encoded):
                    raise ValueError('文本超过模型 token 上限，请缩短问题或调整切块；未静默截断。')
                length = max(len(item.ids) for item in encoded)
                ids = np.zeros((len(batch), length), dtype=np.int64)
                mask = np.zeros_like(ids)
                types = np.zeros_like(ids)
                for i, item in enumerate(encoded):
                    ids[i, :len(item.ids)] = item.ids
                    mask[i, :len(item.ids)] = 1
                    types[i, :len(item.ids)] = item.type_ids
                values = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types}
                inputs = {item.name: values[item.name] for item in self.session.get_inputs()}
                output = self.session.run(None, inputs)[0]
            output = output[:, 0, :] if output.ndim == 3 else output
            output = np.asarray(output, dtype=np.float32)
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            if output.shape != (len(batch), self.dimension) or not np.isfinite(output).all() or (norms <= 0).any():
                raise ValueError('模型返回了无效向量。')
            vectors.append(output / norms)
            progress(f'向量化：{min(offset + batch_size, len(texts))}/{len(texts)} 个片段')
        check_cancel(cancel)
        return np.concatenate(vectors)
