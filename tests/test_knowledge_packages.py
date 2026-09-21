import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import tkinter as tk
import unittest
from unittest.mock import patch

import httpx
import numpy as np

import knowledge_packages as kb
from knowledge_embedding import Encoder
from qa_provider import APIProvider
from qa_worker import make_prompt
from reference_files import ReferenceTools, validate_source


def response(text):
    return httpx.Response(200, text='data: ' + json.dumps({'choices': [
        {'delta': {'content': text}, 'finish_reason': 'stop'}]}) + '\n\n')


def write_package(path, encoder, customer='test-customer', text='数据库提交后消息未发出：采用 outbox 同事务记录，后台补发。', source='项目.md'):
    vector = encoder.encode([text])[0]
    manifest = {'format': kb.FORMAT, 'customer_id': customer, 'package_id': 'test-package',
                'name': '回归测试资料', 'embedding': kb.expected_embedding(), 'version': 1,
                'fingerprint': hashlib.sha256(text.encode()).hexdigest(),
                'chunk_count': 1, 'keyword_tokenizer': 'chinese-bigram-ascii-v1'}
    connection = sqlite3.connect(path)
    try:
        connection.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE chunks(id TEXT PRIMARY KEY, source TEXT, section TEXT, page INTEGER,
                line_start INTEGER, line_end INTEGER, text TEXT, content_hash TEXT, vector BLOB);
            CREATE VIRTUAL TABLE keywords USING fts5(terms, tokenize='unicode61');
        ''')
        connection.execute('INSERT INTO metadata VALUES (?,?)', ('manifest', json.dumps(manifest)))
        connection.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)',
                           ('chunk1', source, '故障处理', None, 10, 15, text, 'test', vector.astype('<f4').tobytes()))
        connection.execute('INSERT INTO keywords(rowid,terms) VALUES (1,?)', (' '.join(kb._terms(text)),))
        connection.commit()
    finally:
        connection.close()
    return manifest


class KnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = Encoder()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / 'test.wlkb'
        write_package(self.path, self.encoder)
        self.settings = {'enabled': True, 'paths': [str(self.path)]}

    def provider(self, handler, provider='deepseek', model='deepseek-flash'):
        return APIProvider(provider, dict(base_url='https://example.com', model=model, api_key='fake'),
                           httpx.MockTransport(handler), lambda: self.settings)

    def test_import_and_actual_semantic_context(self):
        self.assertEqual(validate_source(self.path), str(self.path.resolve()))
        value = kb.KnowledgeLibrary().context(self.settings, make_prompt('', '保存成功但提醒没发出怎么解决？'), [], threading.Event())
        self.assertIn('outbox', value)
        self.assertIn('kb1/test.wlkb/项目.md', value)
        self.assertLess(len(value.encode()), kb.MAX_CONTEXT_BYTES + 300)

    def test_disabled_and_explicit_no_read(self):
        library = kb.KnowledgeLibrary()
        with patch('knowledge_packages.get_encoder', side_effect=AssertionError('should not load model')):
            self.assertEqual(library.context(dict(self.settings, enabled=False), '问题', [], threading.Event()), '')
            self.assertEqual(library.context(self.settings, make_prompt('', '不用读取资料，只解释概念'), [], threading.Event()), '')

    def test_only_wlkb_packages_are_accepted(self):
        for extension in ('.wenlukb', '.sqlite', '.db'):
            with self.subTest(extension=extension):
                old = self.path.with_suffix(extension)
                old.write_bytes(self.path.read_bytes())
                with self.assertRaisesRegex(ValueError, '仅支持 .wlkb'):
                    kb.inspect_package(old)
                with self.assertRaises(ValueError):
                    validate_source(old)
                settings = dict(self.settings, paths=[str(old)])
                self.assertEqual(kb.package_paths(settings), [])
                self.assertEqual(ReferenceTools(settings, threading.Event()).roots, {})
        upper = self.path.with_suffix('.WLKB')
        upper.write_bytes(self.path.read_bytes())
        self.assertEqual(validate_source(upper), str(upper.resolve()))

    def test_both_providers_receive_data_with_alternating_history(self):
        for provider in ('deepseek', 'compatible'):
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                self.assertIn('outbox', body['messages'][-1]['content'])
                self.assertNotIn('outbox', body['messages'][0]['content'])
                self.assertIn('禁止把团队成果', body['messages'][0]['content'])
                self.assertEqual([m['role'] for m in body['messages']], ['system', 'user', 'assistant', 'user'])
                self.assertNotEqual(body.get('tool_choice'), 'required')
                self.assertNotIn('read_reference_file', str(body.get('tools')))
                return response('采用 outbox [kb1/test.wlkb/项目.md:10]。')
            result = self.provider(handler, provider).run(make_prompt('', '不要联网，读取知识包说明通知遗漏如何处理'), threading.Event(),
                history=[{'role': 'user', 'content': '说说你的项目'}, {'role': 'assistant', 'content': '我负责提醒模块。'}])
            self.assertIn('outbox', result)
            self.assertEqual(len(requests), 1)

    def test_probe_never_loads_or_transmits_package(self):
        def handler(request):
            self.assertNotIn('outbox', request.content.decode())
            self.assertNotIn('知识包检索资料', request.content.decode())
            return response('OK')
        with patch('knowledge_packages.KnowledgeLibrary.context', side_effect=AssertionError('probe accessed package')):
            self.assertEqual(self.provider(handler).check_connection(threading.Event()), 'OK')

    def test_corrupt_and_cross_customer_rejected(self):
        other = self.folder / 'other.wlkb'
        write_package(other, self.encoder, customer='other-customer')
        with self.assertRaisesRegex(ValueError, '不同客户'):
            kb.validate_selection([str(self.path), str(other)])
        self.settings['paths'].append(str(other))
        with self.assertRaisesRegex(RuntimeError, '客户身份'):
            kb.KnowledgeLibrary().context(self.settings, '故障', [], threading.Event())
        broken = self.folder / 'broken.wlkb'
        broken.write_text('not SQLite', encoding='utf-8')
        with self.assertRaises(ValueError):
            validate_source(broken)

    def test_source_paths_never_escape_and_bad_model_rejected(self):
        malicious = self.folder / 'bad.wlkb'
        write_package(malicious, self.encoder, source='../private.txt')
        with self.assertRaisesRegex(RuntimeError, '来源路径'):
            kb.KnowledgeLibrary().context({'enabled': True, 'paths': [str(malicious)]}, '故障', [], threading.Event())
        connection = sqlite3.connect(self.path)
        try:
            value = kb.inspect_package(self.path)
            value['embedding']['dimension'] = 768
            connection.execute('UPDATE metadata SET value=?', (json.dumps(value),))
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(ValueError, '模型不兼容'):
            validate_source(self.path)

    def test_plain_reference_numbering_survives_mixed_selection(self):
        text = self.folder / 'notes.txt'
        text.write_text('普通资料', encoding='utf-8')
        settings = dict(self.settings, paths=[str(self.path), str(text)])
        references = ReferenceTools(settings, threading.Event())
        self.assertEqual(list(references.roots), ['r2'])

    def test_disk_replacement_refreshes_vectors_and_context(self):
        library = kb.KnowledgeLibrary()
        library.context(self.settings, '通知故障', [], threading.Event())
        newer = self.folder / 'new.wlkb'
        write_package(newer, self.encoder, text='新版采用事务发件箱，并增加人工告警。')
        newer.replace(self.path)
        result = library.context(self.settings, '通知故障', [], threading.Event())
        self.assertIn('人工告警', result)
        self.assertNotIn('outbox', result)

    def test_flash_extracts_first_then_pro_receives_knowledge(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                self.assertNotIn('知识包检索资料', request.content.decode())
                return response('题目：数据库提交后消息丢失，如何解决？')
            self.assertIn('outbox', request.content.decode())
            return response('使用事务发件箱。')
        self.provider(handler, model='deepseek-v4-pro').run(make_prompt('', '分析这张图'), threading.Event(), image=b'fake-image')
        self.assertEqual(len(requests), 2)

    def test_cancellation_makes_no_network_request(self):
        cancel = threading.Event()
        cancel.set()
        def handler(_):
            self.fail('cancelled request reached network')
        with self.assertRaisesRegex(RuntimeError, '取消'):
            self.provider(handler).run(make_prompt('', '项目怎么处理消息'), cancel)

    def test_settings_import_save_and_context_boundary(self):
        from app import App
        with patch('app.GlobalHotkey'), patch('app.App.start_tray'), patch('qa_connection.QAConnection.check'), \
                patch('reference_settings.load_settings', return_value={'enabled': False, 'paths': []}), \
                patch('reference_settings.save_settings') as save:
            root = tk.Tk()
            root.withdraw()
            app = App(root)
            try:
                app.qa_history.append(('旧客户问题', '旧客户答案'))
                app.rows.append({'source': 'system', 'text': '旧客户转写'})
                with patch('reference_settings.filedialog.askopenfilenames', return_value=(str(self.path),)) as picker:
                    app.reference_controls['add_packages']()
                self.assertEqual(picker.call_args.kwargs['filetypes'], [('闻录知识包', '*.wlkb')])
                self.assertIn('知识包', app.reference_controls['listing'].get(0))
                self.assertEqual(save.call_args.args[0]['paths'], [str(self.path.resolve())])
                app.auto_qa.set(True)
                self.assertEqual(app.session_context_snapshot(), ([], []))
                self.assertEqual(len(app.qa_history), 1)  # Visible history retained.
                app.qa_history.append(('新问题', '新答案'))
                app.rows.append({'source': 'system', 'text': '新转写'})
                rows, exchanges = app.session_context_snapshot()
                self.assertEqual(rows, [('system', '新转写')])
                self.assertEqual(exchanges, [('新问题', '新答案')])
                app.reference_controls['remove_path'](str(self.path.resolve()))
                self.assertTrue(self.path.exists())
                self.assertEqual(app.session_context_snapshot(), ([], []))
            finally:
                app.qa.set_enabled(False)
                app.qa_connection.close()
                app.hotkey.close()
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()


if __name__ == '__main__':
    unittest.main()
