"""Single-writer record appends with rollback for recoverable write errors.

This protects a previously complete file from a torn append reported by Python.
It is not a power-loss transaction: abrupt termination and failed truncation
still require recovery. Appending never scans or rewrites the existing history.
"""
import codecs
import os
from pathlib import Path


def append_bytes(path, payload, *, initial_prefix=b'', ensure_newline=False):
    # Unbuffered I/O prevents unwritten buffered bytes from being retried by
    # close() after rollback. A short write is completed or rolled back too.
    with Path(path).open('a+b' if ensure_newline else 'ab', buffering=0) as stream:
        # Append mode already opens at EOF. Re-seeking a read/write Windows
        # handle to EOF adds unnecessary filesystem work for each caption.
        original_size = stream.tell()
        prefix = initial_prefix if not original_size else b''
        if original_size and ensure_newline:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b'\n':
                prefix = b'\n'
        pending = memoryview(prefix + payload)
        try:
            while pending:
                count = stream.write(pending)
                if count is None or count <= 0 or count > len(pending):
                    raise OSError('记录追加未能完成，请导出当前文字。')
                pending = pending[count:]
            stream.flush()
        except BaseException:
            try:
                stream.truncate(original_size)
            except OSError as rollback_error:
                raise OSError('记录追加失败，且未能回滚文件尾部；请保留原文件并导出当前文字。') from rollback_error
            raise


def append_text(path, text, *, bom=False):
    # Match the previous TextIO newline and UTF-8-sig behavior byte for byte.
    payload = text.replace('\n', os.linesep).encode('utf-8')
    append_bytes(path, payload, initial_prefix=codecs.BOM_UTF8 if bom else b'')
