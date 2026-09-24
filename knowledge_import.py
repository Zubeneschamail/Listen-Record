"""Authenticated local handoff and managed storage for Builder knowledge packages."""
import hashlib
from concurrent.futures import Future
from dataclasses import dataclass
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
    snapshot = import_snapshot(app)
    prepared = prepare_import(source, snapshot, data)
    return prepared.commit(app)


def import_snapshot(app):
    if app.settings_visible:
        raise ValueError('闻录设置窗口正在编辑，请先保存或取消设置，再点击导入。')
    return app.reference_controls['snapshot']()


@dataclass
class PreparedImport:
    temporary: Path
    target: Path
    manifest: dict
    paths: list
    snapshot: dict

    def discard(self):
        self.temporary.unlink(missing_ok=True)

    def commit(self, app):
        backup = None
        published = False
        try:
            if import_snapshot(app) != self.snapshot:
                raise ValueError('导入过程中资料设置发生变化，请重新点击导入。')
            if self.target.exists():
                backup = self.temporary.with_suffix('.backup')
                self.target.replace(backup)
            self.temporary.replace(self.target)
            published = True
            try:
                app.reference_controls['import_paths'](self.paths + [str(self.target)])
            except Exception:
                self.target.unlink(missing_ok=True)
                published = False
                raise
        finally:
            self.discard()
            if backup is not None:
                if published:
                    backup.unlink(missing_ok=True)
                else:
                    backup.replace(self.target)
        return {'name': self.manifest['name'], 'path': str(self.target),
                'customer_id': self.manifest['customer_id']}


def prepare_import(source, snapshot, data=DATA):
    """Validate and copy on a worker; publish only after returning to the UI thread."""
    from knowledge_packages import inspect_package, validate_selection, webpage_sources
    source = Path(source).resolve(strict=True)
    manifest = inspect_package(source)
    webpage_sources(manifest)
    folder = Path(data) / 'knowledge-packages'
    folder.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps([manifest['customer_id'], manifest['package_id']]).encode()).hexdigest()[:32]
    target = folder / (identity + '.wlkb')
    paths = []
    for value in snapshot['paths']:
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
    try:
        before = _hash(source)
        shutil.copyfile(source, temporary)
        if _hash(temporary) != before or _hash(source) != before or inspect_package(temporary) != manifest:
            raise ValueError('知识包在导入时发生变化，请生成完成后重试。')
        return PreparedImport(temporary, target, manifest, paths, snapshot)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class ImportServer:
    """Disk preparation runs in a worker; final package/settings changes run on Tk."""
    def __init__(self, root, callback, data=DATA, prepare=None):
        self.root, self.callback, self.data = root, callback, Path(data)
        self.prepare, self.active = prepare, None
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
                if request['done'].wait(600):
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
        if self.active is not None:
            request, future = self.active
            if future.done():
                self.active = None
                prepared = None
                try:
                    prepared = future.result()
                    request['result'] = dict(self.callback(prepared), ok=True)
                except Exception as exc:
                    request['result'] = {'ok': False, 'error': str(exc)}
                finally:
                    if prepared is not None:
                        prepared.discard()
                    request['done'].set()
            self.timer = self.root.after(100, self.poll)
            return
        try:
            request = self.requests.get_nowait()
        except queue.Empty:
            pass
        else:
            try:
                if time.monotonic() >= request['expires']:
                    raise ValueError('导入请求已过期，请重新点击导入。')
                if self.prepare is None:
                    request['result'] = dict(self.callback(request['path']), ok=True)
                else:
                    work = self.prepare(request['path'])
                    future = Future()
                    def run():
                        try:
                            future.set_result(work())
                        except BaseException as exc:
                            future.set_exception(exc)
                    self.active = request, future
                    threading.Thread(target=run, daemon=True).start()
            except Exception as exc:
                request['result'] = {'ok': False, 'error': str(exc)}
            finally:
                if 'result' in request:
                    request['done'].set()
        self.timer = self.root.after(100, self.poll)

    def close(self):
        self.closed = True
        if self.active is not None:
            request, future = self.active
            request['result'] = {'ok': False, 'error': '闻录已关闭，导入未完成。'}
            request['done'].set()
            def discard(finished):
                try:
                    finished.result().discard()
                except BaseException:
                    pass
            future.add_done_callback(discard)
            self.active = None
        while True:
            try:
                pending = self.requests.get_nowait()
            except queue.Empty:
                break
            pending['result'] = {'ok': False, 'error': '闻录已关闭，导入未完成。'}
            pending['done'].set()
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
    def prepare(path):
        snapshot = import_snapshot(app)
        app.qa_status.set('正在校验并复制知识包…')
        return lambda: prepare_import(path, snapshot)

    def receive(prepared):
        result = prepared.commit(app)
        app.open_qa()
        app.root.deiconify()
        app.root.lift()
        app.qa_status.set('已导入知识包：' + result['name'])
        return result
    register_launcher()
    return ImportServer(app.root, receive, prepare=prepare)
