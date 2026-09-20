"""API preflight and explicit end-to-end checks, separate from user questions."""
import threading


def request_failure(error):
    text = str(error).lower()
    if any(word in text for word in ('401', 'unauthorized', 'authentication', '密钥')):
        return 'unauthenticated', 'API Key 无效，请在模型服务设置中检查密钥。'
    if any(word in text for word in ('429', 'quota', 'usage limit', 'rate limit', '额度')):
        return 'unavailable', '答疑模型额度不足或请求受限，请稍后重试。'
    if any(word in text for word in ('timeout', 'timed out', '超时')):
        return 'unavailable', '答疑模型响应超时，请检查网络后重新检测。'
    return 'unavailable', '答疑模型请求失败，请检查网络和服务配置后重新检测。'


class QAConnection:
    def __init__(self, events, preflight=None, probe=None, name='答疑模型', failure=request_failure):
        self.events = events
        self.name = name
        self.failure = failure
        self.preflight = preflight or (lambda: ('unauthenticated', '请先配置答疑模型。'))
        self.probe = probe
        self.state, self.detail = 'unknown', f'尚未检测 {name}。'
        self.revision = 0
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def _publish(self, revision, state, detail):
        with self.lock:
            if revision != self.revision or self.cancel.is_set():
                return
            self.state, self.detail = state, detail
            self.events.put(('qa_connection', None))

    def check(self, probe=False):
        with self.lock:
            if self.state == 'checking':
                return
            self.cancel.set()
            self.cancel = threading.Event()
            self.revision += 1
            revision, cancel = self.revision, self.cancel
            self.state, self.detail = 'checking', f'正在检测 {self.name}…'
            self.events.put(('qa_connection', None))

        def work():
            try:
                state, detail = self.preflight()
                if cancel.is_set():
                    return
                if state == 'authenticated' and probe:
                    self._publish(revision, 'checking', '正在验证服务响应…')
                    answer = self.probe(cancel)
                    if not answer or not answer.strip():
                        raise RuntimeError('Empty response')
                    state, detail = 'verified', '连接正常，测试请求已成功返回。'
                self._publish(revision, state, detail)
            except Exception as exc:
                self._publish(revision, *self.failure(exc))
        threading.Thread(target=work, daemon=True, name='qa-connection').start()

    def record(self, error=None):
        with self.lock:
            self.cancel.set()
            self.cancel = threading.Event()
            self.revision += 1
            self.state, self.detail = self.failure(error) if error else (
                'verified', f'最近一次请求成功，{self.name} 响应正常。')
            self.events.put(('qa_connection', None))

    def close(self):
        with self.lock:
            self.cancel.set()
            self.revision += 1
