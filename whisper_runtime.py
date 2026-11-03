"""Keep capture-model CUDA state on a long-lived thread, including destruction."""
import atexit
from concurrent.futures import Future
import queue
import threading
from types import SimpleNamespace
import weakref

from tokenizers import Tokenizer

_ACTIVE = weakref.WeakSet()


def _detached_error(exc):
    # Passing the original traceback would retain worker frames and the native
    # model on the caller, potentially destroying CUDA resources on that thread.
    error = RuntimeError(str(exc))
    error.worker_error_type = type(exc).__name__
    return error


def _run(factory_box, commands, ready):
    native = None
    closing = None
    try:
        factory = factory_box.pop()
        try:
            native = factory()
        finally:
            factory = None  # Do not retain a status callback's Transcriber.
        tokenizer = getattr(native, 'hf_tokenizer', None)
        metadata = dict(device=str(native.model.device),
                        max_length=getattr(native, 'max_length', 448),
                        tokenizer=tokenizer.to_str() if isinstance(tokenizer, Tokenizer) else None)
        tokenizer = None
        ready.set_result(metadata)
        while True:
            operation, args, options, response = commands.get()
            if operation == 'close':
                closing = response
                break
            segments = info = None
            try:
                segments, info = native.transcribe(*args, **options)
                # Exhaust lazy generators here; their frames retain the native
                # model and would otherwise run inference on the caller thread.
                response.set_result((list(segments), info))
            except BaseException as exc:
                response.set_exception(_detached_error(exc))
            finally:
                segments = info = args = options = response = None
    except BaseException as exc:
        if not ready.done():
            ready.set_exception(_detached_error(exc))
    finally:
        failure = None
        try:
            if native is not None:
                native.model.unload_model()
        except BaseException as exc:
            failure = _detached_error(exc)
        finally:
            native = None  # Drop the final model reference before its thread exits.
        if closing is not None:
            if failure is None:
                closing.set_result(None)
            else:
                closing.set_exception(failure)


class WhisperCaptureModel:
    """Synchronous Whisper interface with a persistent native-model owner."""
    def __init__(self, factory):
        self.closed = False
        self._lock = threading.Lock()
        self._commands = queue.Queue(maxsize=1)
        ready = Future()
        # Thread._args outlives the factory invocation. A consumed box prevents
        # it from retaining the Transcriber -> adapter -> thread reference cycle.
        self._thread = threading.Thread(target=_run, args=([factory], self._commands, ready),
                                        daemon=True, name='wenlu-whisper-model')
        try:
            self._thread.start()
            metadata = ready.result()
        except BaseException:
            self.closed = True
            if self._thread.ident is not None:
                # Startup may have been interrupted on the caller while native
                # loading continues. Arrange owner cleanup before joining it.
                self._commands.put(('close', (), {}, Future()))
                self._thread.join()
            raise
        try:
            self.model = SimpleNamespace(device=metadata['device'])
            self.max_length = metadata['max_length']
            # Prompt budgeting still receives a real tokenizer, without exposing
            # the Whisper model or a lazy iterator outside its owning thread.
            self.hf_tokenizer = Tokenizer.from_str(metadata['tokenizer']) if metadata['tokenizer'] else None
        except BaseException:
            self.close()
            raise
        _ACTIVE.add(self)

    def transcribe(self, *args, **options):
        with self._lock:
            if self.closed:
                raise RuntimeError('Whisper 识别模型已关闭，请重新开始转写。')
            response = Future()
            self._commands.put(('transcribe', args, options, response))
            segments, info = response.result()
        return iter(segments), info

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            response = Future()
            self._commands.put(('close', (), {}, response))
            try:
                response.result()
            finally:
                self._thread.join()
                _ACTIVE.discard(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def close_capture_models():
    """Also release models retained by non-GUI tools before Python shutdown."""
    for model in list(_ACTIVE):
        try:
            model.close()
        except Exception:
            pass


atexit.register(close_capture_models)
