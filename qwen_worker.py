"""Optional isolated Qwen GPU engine. The desktop process never imports torch.

This module is an engine entry point, not an alternative application launcher.
Only the desktop adapter should submit audio after its speech-activity check.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

MAX_SAMPLES = 30 * 16000
MAX_MESSAGE = 3 * 1024 * 1024


def watch_parent(pid):
    """Exit even during loading when a cancelled diagnostic parent is killed."""
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise ValueError('Invalid desktop process')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not handle:
            raise OSError('无法监测主程序，识别引擎未启动。')
        def wait():
            try:
                kernel.WaitForSingleObject(handle, 0xffffffff)
            finally:
                kernel.CloseHandle(handle)
            os._exit(0)
    else:
        import time
        def wait():
            while os.getppid() == pid:
                time.sleep(.2)
            os._exit(0)
    threading.Thread(target=wait, daemon=True, name='qwen-parent-watch').start()


def serve(model_root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    import numpy as np
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForTokenClassification
    from qwen_alignment import aligned_words

    def send(value):
        print(json.dumps(value), flush=True)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Qwen 中文准确模型需要支持 BF16 的 NVIDIA 显卡；请切换为 Whisper 模型。')
    common = dict(dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa',
                  local_files_only=True, trust_remote_code=False)
    asr_path = model_root / 'qwen3-asr-0.6b'
    aligner_path = model_root / 'qwen3-forced-aligner-0.6b'
    processor = AutoProcessor.from_pretrained(asr_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(asr_path, **common).eval()
    align_processor = AutoProcessor.from_pretrained(aligner_path, local_files_only=True, trust_remote_code=False)
    aligner = AutoModelForTokenClassification.from_pretrained(aligner_path, **common).eval()

    def recognize(sound, language, timestamps, limit=512):
        inputs = processor.apply_transcription_request(audio=sound, language=language,
            processor_kwargs={'sampling_rate':16000}).to(model.device, model.dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=limit, do_sample=False, num_beams=1)
        generated = ids[:, inputs['input_ids'].shape[1]:]
        if limit == 512 and generated.shape[1] >= limit:
            raise RuntimeError('识别输出达到长度上限，未提交不完整字幕。')
        text = processor.decode(generated, return_format='transcription_only')[0]
        words = None
        if timestamps:
            words = []
            if any(char.isalnum() for char in text):
                align_inputs, word_lists = align_processor.prepare_forced_aligner_inputs(
                    audio=sound, transcript=text, language=language,
                    processor_kwargs={'sampling_rate':16000})
                align_inputs = align_inputs.to(aligner.device, aligner.dtype)
                with torch.inference_mode(): output = aligner(**align_inputs)
                alignment = align_processor.decode_forced_alignment(logits=output.logits,
                    input_ids=align_inputs['input_ids'], word_lists=word_lists,
                    timestamp_token_id=aligner.config.timestamp_token_id)[0]
                words = aligned_words(text, alignment, len(sound)/16000)
        return dict(text=text, words=words)

    # Exercise both networks before reporting ready. Warmup text is discarded;
    # empty audio can elicit a plausible phrase even with no actual speaker.
    recognize(np.zeros(8000, dtype=np.float32), 'Chinese', False, limit=8)
    warm, word_lists = align_processor.prepare_forced_aligner_inputs(
        audio=np.zeros(8000, dtype=np.float32), transcript='好', language='Chinese',
        processor_kwargs={'sampling_rate':16000})
    with torch.inference_mode(): aligner(**warm.to(aligner.device, aligner.dtype))
    torch.cuda.synchronize()
    send(dict(type='ready', protocol=1, device='cuda', warmup=True,
              gpu=torch.cuda.get_device_name(0), acoustic_hints=False))
    while True:
        line = sys.stdin.buffer.readline(MAX_MESSAGE+1)
        if not line:
            return
        if len(line) > MAX_MESSAGE:
            raise ValueError('Engine request is too large')
        request = json.loads(line)
        request_id = request['id']
        try:
            if request.get('type') == 'close':
                return
            language = request.get('language')
            if language not in (None, 'Chinese', 'English'):
                raise ValueError('Unsupported recognition language')
            count = request['samples']
            if not isinstance(count, int) or not 0 < count <= MAX_SAMPLES:
                raise ValueError('Invalid audio length')
            data = base64.b64decode(request['audio'], validate=True)
            if len(data) != count*4 or hashlib.sha256(data).hexdigest() != request['sha256']:
                raise ValueError('Audio payload verification failed')
            sound = np.frombuffer(data, dtype='<f4')
            if not np.isfinite(sound).all():
                raise ValueError('Audio contains non-finite samples')
            result = recognize(sound, language, bool(request.get('word_timestamps', True)))
            send(dict(result, type='result', id=request_id, sha256=request['sha256']))
        except Exception as exc:
            send(dict(type='error', id=request_id, message=str(exc)[:500]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--parent-pid', type=int, default=os.getppid())
    args = parser.parse_args()
    try:
        watch_parent(args.parent_pid)
        # Identify the real interpreter before importing GPU libraries. A
        # Windows venv launcher may have a different PID from this process.
        print(json.dumps(dict(type='starting', protocol=1, pid=os.getpid())), flush=True)
        serve(args.model_root)
    except Exception as exc:
        print(json.dumps(dict(type='error', message=str(exc)[:500])), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
