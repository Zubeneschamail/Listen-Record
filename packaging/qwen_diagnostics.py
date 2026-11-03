"""Opt-in startup stack diagnostics; no audio or transcript content is logged."""
import os

path = os.environ.get('WENLU_QWEN_DIAGNOSTIC_LOG')
if path:
    import faulthandler
    _qwen_diagnostic_stream = open(path, 'w', encoding='utf-8')
    faulthandler.enable(file=_qwen_diagnostic_stream)
    faulthandler.dump_traceback_later(20, repeat=True, file=_qwen_diagnostic_stream)
