"""Authenticated local handoff and managed storage for Builder knowledge packages."""
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import sys
import tempfile
import threading
import time

from app_paths import DATA

PROTOCOL = 1


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def register_launcher(data=DATA):
    """Remember the last supported installation, including a development checkout."""
    root = Path(__file__).resolve().parent
    if getattr(sys, 'frozen', False):
        target = Path(sys.executable)
    else:
        target = root / 'launcher.py'
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    write_json(data / 'knowledge-launcher.json', {'protocol': PROTOCOL, 'target': str(target)})


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def import_into_app(app, source, data=DATA):
    from knowledge_packages import inspect_package, validate_selection, webpage_sources
    if app.settings_visible:
        raise ValueError('闻录设置窗口正在编辑，请先保存或取消设置，再点击导入。')
    source = Path(source).resolve(strict=True)
    manifest = inspect_package(source)
    webpage_sources(manifest)
    folder = Path(data) / 'knowledge-packages'
    folder.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps([manifest['customer_id'], manifest['package_id']]).encode()).hexdigest()[:32]
    target = folder / (identity + '.wlkb')
    paths = []
    for value in app.reference_controls['snapshot']()['paths']:
        if Path(value).suffix.lower() == '.wlkb':
            old = inspect_package(value)
            if (old['customer_id'], old['package_id']) == (manifest['customer_id'], manifest['package_id']):
                continue
        paths.append(value)
    # Validate using the incoming path before publishing the managed copy.
    validate_selection(paths + [str(source)])
    if len(paths) >= 32:
        raise ValueError('闻录已有 32 项资料，请先移除不再使用的资料。')
    handle, name = tempfile.mkstemp(suffix='.wlkb', prefix='import-', dir=folder)
    os.close(handle)
    temporary = Path(name)
    backup = None
    published = False
    try:
        before = _hash(source)
        shutil.copyfile(source, temporary)
        if _hash(temporary) != before or _hash(source) != before or inspect_package(temporary) != manifest:
            raise ValueError('知识包在导入时发生变化，请生成完成后重试。')
        if target.exists():
            backup = folder / (temporary.stem + '.backup')
            target.replace(backup)
        temporary.replace(target)
        published = True
        try:
            app.reference_controls['import_paths'](paths + [str(target)])
        except Exception:
            target.unlink(missing_ok=True)
            published = False
            raise
    finally:
        temporary.unlink(missing_ok=True)
        if backup is not None:
            if published:
                backup.unlink(missing_ok=True)
            else:
                backup.replace(target)
    return {'name': manifest['name'], 'path': str(target), 'customer_id': manifest['customer_id']}


class ImportServer:
    """Network threads only enqueue work; package/settings/UI changes run on Tk's thread."""
    def __init__(self, root, callback, data=DATA):
        self.root, self.callback, self.data = root, callback, Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.requests = queue.Queue(maxsize=8)
        self.token = secrets.token_urlsafe(32)
        self.closed = False
        self.timer = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                body = json.dumps(value, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def authorized(self):
                if not hmac.compare_digest(self.headers.get('Authorization', '').encode('utf-8'), ('Bearer ' + owner.token).encode('utf-8')):
                    self.reply(403, {'ok': False, 'error': '未授权的导入请求。'})
                    return False
                return True

            def do_GET(self):
                if self.authorized():
                    self.reply(200 if self.path == '/health' else 404,
                               {'ok': self.path == '/health', 'protocol': PROTOCOL})

            def do_POST(self):
                if not self.authorized():
                    return
                if self.path != '/import':
                    self.reply(404, {'ok': False, 'error': '未知操作。'})
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 1 <= length <= 16384:
                        raise ValueError()
                    self.connection.settimeout(3)
                    value = json.loads(self.rfile.read(length))
                    path = value['path']
                    if not isinstance(path, str) or not 1 <= len(path) <= 4096:
                        raise ValueError()
                    request = {'path': path, 'done': threading.Event(), 'expires': time.monotonic() + 25}
                    owner.requests.put_nowait(request)
                except (ValueError, KeyError, TypeError, OSError, queue.Full):
                    self.reply(400, {'ok': False, 'error': '导入请求无效或闻录正忙。'})
                    return
                if request['done'].wait(30):
                    self.reply(200, request['result'])
                else:
                    self.reply(408, {'ok': False, 'error': '闻录未及时处理导入，请检查闻录窗口。'})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = self.data / 'knowledge-import.json'
        try:
            write_json(self.endpoint, {'protocol': PROTOCOL, 'port': self.server.server_port,
                                       'token': self.token, 'pid': os.getpid()})
            self.poll()
        except Exception:
            self.close()
            raise

    def poll(self):
        self.timer = None
        if self.closed:
            return
        try:
            request = self.requests.get_nowait()
        except queue.Empty:
            pass
        else:
            try:
                if time.monotonic() >= request['expires']:
                    raise ValueError('导入请求已过期，请重新点击导入。')
                request['result'] = dict(self.callback(request['path']), ok=True)
            except Exception as exc:
                request['result'] = {'ok': False, 'error': str(exc)}
            finally:
                request['done'].set()
        self.timer = self.root.after(100, self.poll)

    def close(self):
        self.closed = True
        if self.timer is not None:
            try:
                self.root.after_cancel(self.timer)
            except Exception:
                pass
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        try:
            if json.loads(self.endpoint.read_text(encoding='utf-8')).get('token') == self.token:
                self.endpoint.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def start_import_server(app):
    def receive(path):
        result = import_into_app(app, path)
        app.open_qa()
        app.root.deiconify()
        app.root.lift()
        app.qa_status.set('已导入知识包：' + result['name'])
        return result
    register_launcher()
    return ImportServer(app.root, receive)
