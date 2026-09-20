"""Isolated GPU inference check: native crashes must not close the application."""
import multiprocessing as mp
import logging
import time


def probe_gpu(name, progress):
    import ctranslate2
    from model_download import cached_model
    from model_runtime import prepare_cuda, cuda_runtime_errors, load_model

    progress('正在检查 NVIDIA 显卡与驱动…')
    prepare_cuda()
    count = ctranslate2.get_cuda_device_count()
    if not count:
        raise RuntimeError('未检测到可用的 CUDA 显卡。请检查 NVIDIA 显卡及驱动。')
    progress(f'发现 {count} 个 CUDA 设备，正在检查运行库…')
    errors = cuda_runtime_errors()
    if errors:
        logging.warning('GPU runtime check failed: %s', '; '.join(errors))
        names = [error.split('：', 1)[0] for error in errors]
        raise RuntimeError('CUDA 运行库加载失败。以下文件或其依赖不可用：\n'
                           + '\n'.join(names)
                           + '\n请安装或修复闻录 GPU 运行库后重新检测。')
    if cached_model(name) is None:
        raise RuntimeError(f'尚未下载 {name} 模型，无法验证 GPU 推理。请先在模型管理中下载。')
    # load_model exhausts the warmup generator: success requires real inference.
    model = load_model(name, device='cuda', status=progress)
    if model.model.device != 'cuda':
        raise RuntimeError('实际模型未使用 CUDA，GPU 检测未通过。')
    return f'{name} 已在 GPU 上完成模型加载和测试推理。检测结果不代表当前转写已切换。'


def gpu_worker(name, sender):
    try:
        result = probe_gpu(name, lambda message: sender.send(('progress', message)))
        sender.send(('success', result))
    except Exception as exc:
        sender.send(('error', str(exc)))
    finally:
        sender.close()


class GPUCheck:
    def __init__(self, events, context=None, clock=time.monotonic, timeout=120):
        self.events, self.clock, self.timeout = events, clock, timeout
        self.context = context or mp.get_context('spawn')
        self.process = self.receiver = None
        self.started = 0
        self.revision = 0

    @property
    def active(self):
        return self.process is not None

    def publish(self, state, message):
        self.events.put(('gpu_check', (self.revision, state, message)))

    def start(self, name):
        if self.active:
            return
        self.revision += 1
        self.receiver, sender = self.context.Pipe(duplex=False)
        self.process = self.context.Process(target=gpu_worker, args=(name, sender), daemon=True)
        try:
            self.process.start()
            self.started = self.clock()
            self.publish('progress', f'正在检测 {name} 的 GPU 加速…')
        except Exception as exc:
            self.process = None
            self.receiver.close()
            self.receiver = None
            self.publish('error', '检测进程启动失败：' + str(exc))
        finally:
            sender.close()

    def poll(self):
        if not self.active:
            return
        try:
            while self.receiver.poll():
                state, message = self.receiver.recv()
                if state in ('success', 'error'):
                    self.close()
                    self.publish(state, message)
                    return
                self.publish(state, message)
        except (EOFError, OSError):
            pass
        if not self.process.is_alive():
            code = self.process.exitcode
            self.close()
            self.publish('error', f'GPU 检测进程异常退出（{code}），请检查显卡驱动与运行库。')
        elif self.clock() - self.started > self.timeout:
            self.close()
            self.publish('error', f'GPU 检测超时（{self.timeout} 秒），请检查显卡驱动、显存和模型文件。')

    def close(self):
        self.revision += 1
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.2)
            self.process = None
        if self.receiver is not None:
            self.receiver.close()
            self.receiver = None
