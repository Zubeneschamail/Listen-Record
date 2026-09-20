"""Bounded startup checks; no Tk calls from workers or user content in probes."""
import copy
import multiprocessing as mp
import queue
import threading
import time

LABELS = {'model': '转录模型', 'gpu': 'GPU 加速', 'deepseek': '答疑模型'}


def check_local(name, report):
    from model_download import cached_model
    from model_runtime import load_model, prepare_cuda, cuda_runtime_errors
    import ctranslate2

    report('model', 'checking', f'正在检查 {name} 模型…')
    if cached_model(name) is None:
        report('model', 'error', f'未找到完整的 {name} 模型，请下载或切换模型。')
        report('gpu', 'blocked', '待转录模型就绪后验证 GPU 推理。')
        return
    report('gpu', 'checking', '正在检查显卡和运行库…')
    gpu_error = None
    try:
        prepare_cuda()
        if not ctranslate2.get_cuda_device_count():
            raise RuntimeError('未检测到可用的 CUDA 显卡，请检查 NVIDIA 显卡及驱动。')
        errors = cuda_runtime_errors()
        if errors:
            raise RuntimeError('运行库或其依赖加载失败：' + '、'.join(e.split('：', 1)[0] for e in errors))
        model = load_model(name, device='cuda', status=lambda s: report('gpu', 'checking', s))
        if model.model.device != 'cuda':
            raise RuntimeError('模型未使用 GPU。')
    except Exception as exc:
        gpu_error = str(exc)
    if gpu_error:
        report('gpu', 'error', gpu_error)
        report('model', 'checking', f'正在验证 {name} 的 CPU 加载…')
        try:
            model = load_model(name, device='cpu')
        except Exception as exc:
            report('model', 'error', f'{name} 加载失败：{exc}')
            return
        report('model', 'success', f'{name} 加载和推理正常，可使用 CPU 转写。')
    else:
        report('model', 'success', f'{name} 加载和推理正常。')
        report('gpu', 'success', f'{name} 已通过 GPU 推理检测。')


def local_worker(name, sender):
    try:
        check_local(name, lambda *item: sender.send(item))
    except Exception as exc:
        sender.send(('model', 'error', f'本地检测失败：{exc}'))
        sender.send(('gpu', 'blocked', '本地检测未完成。'))
    finally:
        sender.close()


class StartupChecks:
    def __init__(self, context=None, clock=time.monotonic):
        self.context = context or mp.get_context('spawn')
        self.clock = clock
        self.results = {key: ('waiting', '等待检测') for key in LABELS}
        self.messages = queue.Queue()
        self.cancel = threading.Event()
        self.process = self.receiver = None
        self.active = False
        self.started = 0

    def start(self, name, profile):
        self.active = True
        self.started = self.clock()
        self.results = {key: ('checking', '正在检测…') for key in LABELS}
        sender = None
        try:
            self.receiver, sender = self.context.Pipe(duplex=False)
            self.process = self.context.Process(target=local_worker, args=(name, sender), daemon=True)
            self.process.start()
        except Exception as exc:
            self.process = None
            if self.receiver:
                self.receiver.close()
                self.receiver = None
            self.results['model'] = ('error', f'无法启动本地检测：{exc}')
            self.results['gpu'] = ('blocked', '本地检测未完成。')
        finally:
            if sender:
                sender.close()
        profile = copy.deepcopy(profile)
        def network():
            from qa_provider import APIProvider
            try:
                backend = APIProvider('deepseek', profile)
                backend.check_connection(self.cancel, timeout=20)
                result = ('success', f'{profile.get("model", "答疑模型")} 已成功响应。')
            except Exception as exc:
                result = ('error', str(exc))
            if not self.cancel.is_set():
                self.messages.put(('deepseek', *result))
        threading.Thread(target=network, daemon=True, name='startup-deepseek').start()

    def _close_local(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.2)
            self.process = None
        if self.receiver:
            self.receiver.close()
            self.receiver = None

    def poll(self):
        if not self.active:
            return
        if self.receiver:
            try:
                while self.receiver.poll():
                    key, state, detail = self.receiver.recv()
                    self.results[key] = (state, detail)
            except (EOFError, OSError):
                pass
            local_finished = all(self.results[k][0] not in ('waiting', 'checking') for k in ('model', 'gpu'))
            if local_finished:
                self._close_local()
            elif not self.process.is_alive() or self.clock() - self.started > 120:
                message = '本地检测超时，请检查模型和显卡。' if self.process.is_alive() else '本地检测进程异常退出，请检查模型和运行库。'
                for key in ('model', 'gpu'):
                    if self.results[key][0] in ('waiting', 'checking'):
                        self.results[key] = ('error', message)
                self._close_local()
        while not self.messages.empty():
            key, state, detail = self.messages.get_nowait()
            if self.results[key][0] == 'checking':
                self.results[key] = (state, detail)
        if self.results['deepseek'][0] == 'checking' and self.clock() - self.started > 25:
            self.cancel.set()
            self.results['deepseek'] = ('error', '答疑模型连接检测超时，请检查网络或服务配置。')
        if all(state not in ('waiting', 'checking') for state, _ in self.results.values()):
            self.active = False

    def close(self):
        self.cancel.set()
        self.active = False
        self._close_local()
