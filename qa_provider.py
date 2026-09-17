"""Text-only streaming API providers and per-user encrypted credentials."""
import asyncio
import base64
import ctypes
from ctypes import wintypes
import json
import time
from urllib.parse import urlsplit

import httpx
from app_paths import DATA

PROVIDERS = {'codex': 'Codex', 'deepseek': 'DeepSeek', 'compatible': '兼容 API'}
DEFAULTS = {
    'codex': {'base_url': '', 'model': ''},
    'deepseek': {'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash'},
    'compatible': {'base_url': '', 'model': ''},
}


class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def protect(value, decrypt=False):
    """DPAPI binds credentials to the current Windows user; never falls back to plaintext."""
    raw = base64.b64decode(value, validate=True) if decrypt else value.encode('utf-8')
    buffer = ctypes.create_string_buffer(raw)
    source = Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise RuntimeError('无法读取或加密 API Key，请重新填写后保存。')
    try:
        output = ctypes.string_at(result.data, result.size)
        return output.decode('utf-8') if decrypt else base64.b64encode(output).decode('ascii')
    finally:
        kernel.LocalFree(result.data)


def load_settings(path=None):
    settings = {'provider': 'codex', 'profiles': {}}
    try:
        stored = json.loads((path or DATA / 'qa-settings.json').read_text(encoding='utf-8'))
        if stored.get('provider') in PROVIDERS:
            settings['provider'] = stored['provider']
        for name, profile in stored.get('profiles', {}).items():
            if name not in PROVIDERS or not isinstance(profile, dict):
                continue
            item = {k: str(profile.get(k, v)) for k, v in DEFAULTS[name].items()}
            try:
                item['api_key'] = protect(profile['encrypted_key'], decrypt=True) if profile.get('encrypted_key') else ''
            except (ValueError, RuntimeError, UnicodeError):
                item['api_key'] = ''
            settings['profiles'][name] = item
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return settings


def save_settings(settings, path=None):
    target = path or DATA / 'qa-settings.json'
    profiles = {}
    for name, profile in settings['profiles'].items():
        profiles[name] = {k: profile.get(k, v) for k, v in DEFAULTS[name].items()}
        key = profile.get('api_key', '')
        profiles[name]['encrypted_key'] = protect(key) if key else ''
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps({'provider': settings['provider'], 'profiles': profiles},
                                    ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(target)


def endpoint(base_url):
    base_url = base_url.strip().rstrip('/')
    url = urlsplit(base_url)
    if (not url.hostname or url.username or url.password or url.query or url.fragment
            or any(char.isspace() for char in base_url)
            or (url.scheme != 'https' and not (
                url.scheme == 'http' and url.hostname in ('localhost', '127.0.0.1', '::1')))):
        raise ValueError('请填写 HTTPS API 地址；仅本机服务允许 HTTP。')
    return base_url if url.path.endswith('/chat/completions') else base_url + '/chat/completions'


class APIProvider:
    def __init__(self, provider, profile, transport=None):
        self.provider = provider
        self.profile = dict(profile)
        self.transport = transport

    def preflight(self):
        try:
            endpoint(self.profile.get('base_url', ''))
        except ValueError as exc:
            return 'unavailable', str(exc)
        if not self.profile.get('model', '').strip():
            return 'unavailable', '请在模型服务中填写模型名称。'
        if not self.profile.get('api_key', '').strip():
            return 'unauthenticated', '请在设置 → 模型服务中填写 API Key。'
        return 'authenticated', '已配置；点击检查连接可验证服务响应。'

    @staticmethod
    def failure(error):
        return 'unavailable', str(error)

    def run(self, prompt, cancel, partial=None, timeout=90):
        state, detail = self.preflight()
        if state != 'authenticated':
            raise RuntimeError(detail)

        async def request():
            system, separator, user = prompt.partition('\n')
            messages = ([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
                        if separator else [{'role': 'user', 'content': prompt}])
            body = {'model': self.profile['model'], 'messages': messages,
                    'stream': True, 'max_tokens': 1024}
            if self.provider == 'deepseek':
                body['thinking'] = {'type': 'disabled'}
            headers = {'Authorization': 'Bearer ' + self.profile['api_key'],
                       'Accept': 'text/event-stream'}
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=8),
                                         transport=self.transport, follow_redirects=False) as client:
                async with client.stream('POST', endpoint(self.profile['base_url']),
                                         headers=headers, json=body) as response:
                    if response.status_code != 200:
                        messages = {401: 'API Key 无效，请重新填写。', 403: '服务拒绝访问，请检查权限。',
                                    402: 'API 余额不足，请到服务商充值。', 404: '接口或模型不存在，请检查地址和模型名。',
                                    429: '请求过于频繁或额度不足，请稍后重试。'}
                        raise RuntimeError(messages.get(response.status_code,
                            f'模型服务请求失败（HTTP {response.status_code}），请检查配置或稍后重试。'))
                    answer, last_emit, complete = '', 0, False
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if not data:
                            continue
                        if data == '[DONE]':
                            complete = True
                            break
                        try:
                            event = json.loads(data)
                            if event.get('error'):
                                raise RuntimeError('模型服务返回错误，请检查模型配置或稍后重试。')
                            choices = event.get('choices') or []
                            if not choices:
                                continue
                            choice = choices[0]
                            content = choice.get('delta', {}).get('content') or ''
                            if isinstance(content, str):
                                answer += content
                            reason = choice.get('finish_reason')
                            if reason:
                                if reason != 'stop':
                                    raise RuntimeError('回答被截断或过滤，请缩短问题后重试。')
                                complete = True
                            now = time.monotonic()
                            if partial and answer and now - last_emit >= 0.12:
                                partial(answer)
                                last_emit = now
                        except (ValueError, AttributeError, TypeError, IndexError) as exc:
                            raise RuntimeError('模型返回格式不兼容，请使用 Chat Completions 流式接口。') from exc
                    if not complete:
                        raise RuntimeError('回答连接中断，请重新发送问题。')
                    if not answer.strip():
                        raise RuntimeError('模型未返回答案，请检查模型名称或重试。')
                    return answer.strip()

        async def cancellable():
            task = asyncio.create_task(request())
            deadline = time.monotonic() + timeout
            try:
                while not task.done():
                    if cancel.is_set():
                        raise RuntimeError('已取消')
                    if time.monotonic() >= deadline:
                        raise RuntimeError('模型响应超时，请稍后重试。')
                    await asyncio.wait({task}, timeout=0.1)
                return await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        try:
            return asyncio.run(cancellable())
        except httpx.TimeoutException:
            raise RuntimeError('模型响应超时，请检查网络后重试。') from None
        except (httpx.HTTPError, UnicodeError, ValueError):
            raise RuntimeError('无法连接模型服务，请检查网络和 API 配置。') from None
