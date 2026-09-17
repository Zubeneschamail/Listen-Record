"""Login preflight and explicit end-to-end checks, separate from user questions."""
import subprocess
import threading
from codex_qa import find_codex, CodexQA


def login_status():
    try:
        executable = find_codex()
    except RuntimeError:
        return 'missing', '未安装 Codex，请安装并登录后重新检测。'
    try:
        result = subprocess.run([executable, 'login', 'status'], capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=8,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        output = (result.stdout + result.stderr).lower()
        if result.returncode == 0 and 'logged in' in output:
            return 'authenticated', '已登录；尚未验证服务响应。点击检测连接可发送一条测试请求。'
        if 'not logged in' in output or 'not authenticated' in output:
            return 'unauthenticated', '未登录 Codex，请在 Codex 中登录后重新检测。'
        return 'unavailable', '无法读取 Codex 登录状态，请检查安装或重新登录。'
    except subprocess.TimeoutExpired:
        return 'unavailable', 'Codex 登录检测超时，请重试。'
    except OSError:
        return 'unavailable', '无法启动 Codex，请检查安装后重试。'


def request_failure(error):
    text = str(error).lower()
    if any(word in text for word in ('401', 'unauthorized', 'not logged in', 'authentication', '未登录')):
        return 'unauthenticated', '登录已失效，请重新登录 Codex 后检测连接。'
    if any(word in text for word in ('429', 'quota', 'usage limit', 'rate limit', '额度')):
        return 'unavailable', 'Codex 额度不足或请求受限，请稍后重试。'
    if any(word in text for word in ('timeout', 'timed out', '超时')):
        return 'unavailable', 'Codex 响应超时，请检查网络后重新检测。'
    return 'unavailable', 'Codex 请求失败，请检查网络、登录状态后重新检测。'


class CodexConnection:
    def __init__(self, events, login=login_status, probe=None, name='Codex', failure=request_failure):
        self.events = events
        self.name = name
        self.failure = failure
        self.login = login
        self.probe = probe or (lambda cancel: CodexQA(None)._run(
            '只回复 OK，不调用任何工具。', cancel, timeout=30))
        self.state, self.detail = 'unknown', f'尚未检测 {name}。'
        self.revision = 0
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def _publish(self, revision, state, detail):
        with self.lock:
            if revision != self.revision or self.cancel.is_set():
                return
            self.state, self.detail = state, detail
            self.events.put(('codex_connection', None))

    def check(self, probe=False):
        with self.lock:
            if self.state == 'checking':
                return
            self.cancel.set()
            self.cancel = threading.Event()
            self.revision += 1
            revision, cancel = self.revision, self.cancel
            self.state, self.detail = 'checking', f'正在检测 {self.name}…'
            self.events.put(('codex_connection', None))

        def work():
            try:
                state, detail = self.login()
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
        threading.Thread(target=work, daemon=True, name='codex-connection').start()

    def record(self, error=None):
        with self.lock:
            self.cancel.set()
            self.cancel = threading.Event()
            self.revision += 1
            self.state, self.detail = self.failure(error) if error else (
                'verified', f'最近一次请求成功，{self.name} 响应正常。')
            self.events.put(('codex_connection', None))

    def close(self):
        with self.lock:
            self.cancel.set()
            self.revision += 1
