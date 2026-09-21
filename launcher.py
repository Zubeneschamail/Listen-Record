"""Executable entry point, with a no-audio packaged smoke check."""
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    if '--self-test' in sys.argv:
        report = Path(sys.argv[sys.argv.index('--self-test') + 1])
        try:
            if '--verify-bundled-model' in sys.argv:
                import os
                os.environ['HF_HUB_OFFLINE'] = '1'
            import app
            import tkinter as tk
            import numpy as np
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            from updates import version_tuple
            from app_paths import RESOURCES
            root = tk.Tk()
            root.withdraw()
            image = tk.PhotoImage(file=str(RESOURCES / 'assets/logo-24.png'))
            get_speech_timestamps(np.zeros(16000, dtype=np.float32), VadOptions())
            from echo_cancellation import make_processor
            echo = make_processor()
            echo_audio = echo.process(np.zeros(160, np.float32), np.zeros(160, np.float32))
            if echo_audio.shape != (160,) or not np.isfinite(echo_audio).all():
                raise RuntimeError('Packaged echo cancellation self-test failed')
            root.destroy()
            if '--verify-bundled-model' in sys.argv:
                from model_download import cached_model, BUNDLED_MODELS
                from model_runtime import load_model
                if cached_model('small') != BUNDLED_MODELS / 'small':
                    raise RuntimeError('Bundled model missing; refusing cache fallback')
                model = load_model('small', device='cpu')
                del model
            if '--verify-download-worker' in sys.argv:
                from model_download import download_worker
                context = multiprocessing.get_context('spawn')
                receiver, sender = context.Pipe(duplex=False)
                worker = context.Process(target=download_worker, args=('small', sender), daemon=True)
                worker.start()
                sender.close()
                try:
                    while receiver.poll(60):
                        kind, message = receiver.recv()
                        if kind == 'error':
                            raise RuntimeError(message)
                        if kind == 'done':
                            break
                    else:
                        raise TimeoutError('Download worker did not finish')
                finally:
                    worker.join(2)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(2)
                    receiver.close()
            if '--verify-model' in sys.argv:
                from model_runtime import load_model
                model = load_model(sys.argv[sys.argv.index('--verify-model') + 1], device='cpu')
                segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language='zh', beam_size=1, max_new_tokens=1)
                list(segments)
            if '--verify-auto-models' in sys.argv:
                from model_runtime import load_model
                for name in ('small', 'large-v3-turbo', 'small'):
                    model = load_model(name)
                    del model
            gpu_verified = False
            if '--verify-gpu' in sys.argv:
                from model_runtime import load_model
                model = load_model('small', device='cuda')
                if model.model.device != 'cuda':
                    raise RuntimeError('GPU verification unexpectedly fell back to CPU')
                del model
                gpu_verified = True
            report.write_text(json.dumps({'ok': True, 'version': app.VERSION, 'echo_verified': True,
                                         'gpu_verified': gpu_verified,
                                         'bundled_model': 'small' if '--verify-bundled-model' in sys.argv else None}), encoding='utf-8')
        except Exception:
            import traceback
            report.write_text(traceback.format_exc(), encoding='utf-8')
            raise
    else:
        import runpy
        # A real import keeps app visible to the freezer's dependency analysis.
        import app
        runpy.run_module('app', run_name='__main__')
