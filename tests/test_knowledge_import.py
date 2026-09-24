from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from knowledge_embedding import Encoder
from knowledge_import import ImportServer, import_into_app, import_snapshot, prepare_import
from test_knowledge_packages import write_package


class KnowledgeImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = Encoder()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.package = self.folder / 'source.wlkb'
        write_package(self.package, self.encoder)
        self.settings = {'enabled': False, 'paths': []}
        def apply(paths):
            self.settings.update(enabled=True, paths=list(paths))
        self.app = SimpleNamespace(settings_visible=False, reference_controls={
            'snapshot': lambda: dict(self.settings), 'import_paths': apply})

    def test_copy_enable_repeat_update_and_original_removal(self):
        result = import_into_app(self.app, self.package, self.folder / 'data')
        managed = Path(result['path'])
        self.assertEqual(managed.read_bytes(), self.package.read_bytes())
        self.assertTrue(self.settings['enabled'])
        self.assertEqual(self.settings['paths'], [str(managed)])
        repeat = import_into_app(self.app, self.package, self.folder / 'data')
        self.assertEqual(repeat['path'], str(managed))
        self.assertEqual(len(self.settings['paths']), 1)
        newer = self.folder / 'new.wlkb'
        write_package(newer, self.encoder, text='新版增加人工告警和重试记录。')
        import_into_app(self.app, newer, self.folder / 'data')
        self.assertEqual(managed.read_bytes(), newer.read_bytes())
        self.package.unlink()
        newer.unlink()
        self.assertTrue(managed.is_file())

    def test_save_failure_restores_existing_package(self):
        managed = Path(import_into_app(self.app, self.package, self.folder)['path'])
        original = managed.read_bytes()
        newer = self.folder / 'new.wlkb'
        write_package(newer, self.encoder, text='新版内容')
        self.app.reference_controls['import_paths'] = lambda _: (_ for _ in ()).throw(OSError('disk full'))
        with self.assertRaisesRegex(OSError, 'disk full'):
            import_into_app(self.app, newer, self.folder)
        self.assertEqual(managed.read_bytes(), original)
        self.assertEqual(list(managed.parent.glob('*.backup')), [])
        self.assertEqual(len(list(managed.parent.glob('*.wlkb'))), 1)

    def test_invalid_cross_customer_and_settings_draft_leave_selection_unchanged(self):
        import_into_app(self.app, self.package, self.folder)
        original = dict(self.settings)
        other = self.folder / 'other.wlkb'
        write_package(other, self.encoder, customer='other')
        with self.assertRaisesRegex(ValueError, '不同客户'):
            import_into_app(self.app, other, self.folder)
        self.app.settings_visible = True
        with self.assertRaisesRegex(ValueError, '设置'):
            import_into_app(self.app, self.package, self.folder)
        self.app.settings_visible = False
        bad = self.folder / 'bad.wlkb'
        bad.write_bytes(b'not sqlite')
        with self.assertRaises(ValueError):
            import_into_app(self.app, bad, self.folder)
        self.assertEqual(self.settings, original)

    def test_package_count_limit(self):
        for index in range(4):
            package = self.folder / f'{index}.wlkb'
            manifest = write_package(package, self.encoder)
            manifest['package_id'] = str(index)
            with closing(sqlite3.connect(package)) as connection:
                connection.execute('UPDATE metadata SET value=?', (json.dumps(manifest),))
                connection.commit()
            import_into_app(self.app, package, self.folder)
        with self.assertRaisesRegex(ValueError, '4'):
            import_into_app(self.app, self.package, self.folder)
        self.assertEqual(len(self.settings['paths']), 4)

    def test_authenticated_handoff_applies_on_ui_thread_and_removes_endpoint(self):
        root = tk.Tk()
        root.withdraw()
        calls = []
        def receive(path):
            import threading
            self.assertIs(threading.current_thread(), threading.main_thread())
            calls.append(path)
            return import_into_app(self.app, path, self.folder)
        server = ImportServer(root, receive, self.folder)
        try:
            endpoint = json.loads((self.folder / 'knowledge-import.json').read_text(encoding='utf-8'))
            url = f'http://127.0.0.1:{endpoint["port"]}'
            with httpx.Client(trust_env=False) as client:
                self.assertEqual(client.post(url + '/import', json={'path': str(self.package)}).status_code, 403)
                headers = {'Authorization': 'Bearer ' + endpoint['token']}
                self.assertTrue(client.get(url + '/health', headers=headers).json()['ok'])
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(client.post, url + '/import', headers=headers, json={'path': str(self.package)})
                    deadline = time.monotonic() + 10
                    while not future.done():
                        self.assertLess(time.monotonic(), deadline)
                        root.update()
                        time.sleep(.01)
                    result = future.result().json()
                    self.assertTrue(result['ok'], result)
                    self.assertEqual(self.settings['paths'], [result['path']])
            self.assertEqual(calls, [str(self.package)])
        finally:
            server.close()
            root.destroy()
        self.assertFalse((self.folder / 'knowledge-import.json').exists())

    def test_real_app_import_enables_references_and_resets_context(self):
        from app import App
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'), \
                patch('reference_settings.load_settings', return_value={'enabled': False, 'paths': []}), \
                patch('reference_settings.save_settings') as save:
            root = tk.Tk()
            root.withdraw()
            app = App(root)
            try:
                app.qa_history.append(('旧问题', '旧回答'))
                result = import_into_app(app, self.package, self.folder)
                self.assertEqual(save.call_args.args[0], {'enabled': True, 'paths': [result['path']]})
                self.assertEqual(app.reference_controls['paths'](), [result['path']])
                self.assertTrue(app.reference_controls['enabled'].get())
                self.assertEqual(app.knowledge_history_start, 1)
                app.qa_history.append(('后续问题', '后续回答'))
                import_into_app(app, self.package, self.folder)
                self.assertEqual(app.knowledge_history_start, 2)
            finally:
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()

    def test_background_copy_keeps_ui_alive_and_commits_on_ui_thread(self):
        self.check_background_import(change_settings=False)

    def test_settings_changed_during_copy_discards_pending_import(self):
        self.check_background_import(change_settings=True)

    def test_close_during_copy_never_publishes_and_removes_temporary(self):
        root = tk.Tk()
        root.withdraw()
        release, cleaned = threading.Event(), threading.Event()
        def prepare(path):
            snapshot = import_snapshot(self.app)
            def work():
                if not release.wait(5):
                    raise RuntimeError('release timeout')
                return prepare_import(path, snapshot, self.folder / 'data')
            return work
        server = ImportServer(root, lambda _: self.fail('published after closing'),
                              self.folder / 'data', prepare=prepare)
        request = {'path': str(self.package), 'done': threading.Event(), 'expires': time.monotonic() + 25}
        server.requests.put(request)
        try:
            deadline = time.monotonic() + 5
            while server.active is None:
                self.assertLess(time.monotonic(), deadline)
                root.update()
                time.sleep(.01)
            future = server.active[1]
            server.close()
            future.add_done_callback(lambda _: cleaned.set())
            release.set()
            self.assertTrue(cleaned.wait(5))
            self.assertFalse(request['result']['ok'])
            self.assertEqual(self.settings['paths'], [])
            self.assertEqual(list((self.folder / 'data' / 'knowledge-packages').iterdir()), [])
        finally:
            release.set()
            if not server.closed:
                server.close()
            root.destroy()

    def check_background_import(self, change_settings):
        root = tk.Tk()
        root.withdraw()
        started, release = threading.Event(), threading.Event()
        heartbeat = []
        def prepare(path):
            self.assertIs(threading.current_thread(), threading.main_thread())
            snapshot = import_snapshot(self.app)
            def work():
                self.assertIsNot(threading.current_thread(), threading.main_thread())
                started.set()
                if not release.wait(5):
                    raise RuntimeError('UI blocked during copy')
                return prepare_import(path, snapshot, self.folder / 'data')
            return work
        def commit(prepared):
            self.assertIs(threading.current_thread(), threading.main_thread())
            return prepared.commit(self.app)
        def tick():
            if started.is_set():
                heartbeat.append(True)
                if change_settings:
                    self.settings['enabled'] = True
                release.set()
            else:
                root.after(10, tick)
        server = ImportServer(root, commit, self.folder / 'data', prepare=prepare)
        try:
            root.after(10, tick)
            with httpx.Client(trust_env=False, timeout=10) as client, ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(client.post, f'http://127.0.0.1:{server.server.server_port}/import',
                                     headers={'Authorization': 'Bearer ' + server.token}, json={'path': str(self.package)})
                deadline = time.monotonic() + 10
                while not future.done():
                    self.assertLess(time.monotonic(), deadline)
                    root.update()
                    time.sleep(.01)
                result = future.result().json()
                self.assertEqual(result['ok'], not change_settings, result)
                self.assertTrue(heartbeat)
                if change_settings:
                    self.assertIn('设置发生变化', result['error'])
                    self.assertEqual(self.settings['paths'], [])
                else:
                    self.assertEqual(self.settings['paths'], [result['path']])
                self.assertEqual(list((self.folder / 'data' / 'knowledge-packages').glob('import-*')), [])
        finally:
            release.set()
            server.close()
            root.destroy()
