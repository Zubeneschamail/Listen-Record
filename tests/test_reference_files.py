import json
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile

from reference_files import ReferenceTools, load_settings, save_settings


class ReferenceFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'docs'
        self.root.mkdir()
        (self.root/'design.md').write_text('项目资料\n精度要求为 0.25 毫米。\n不能超过 0.3 毫米。', encoding='utf-8')
        self.cancel = threading.Event()
        self.tools = ReferenceTools({'enabled': True, 'paths': [str(self.root)]}, self.cancel)

    def call(self, name, **args):
        return json.loads(self.tools.execute(name, json.dumps(args)))

    def test_list_search_read_with_line_citations(self):
        self.assertEqual(self.call('list_reference_files')['files'], ['r1/design.md'])
        result = self.call('search_reference_files', query='精度')
        self.assertEqual(result['matches'][0]['line'], 2)
        self.assertIn('0.25', result['matches'][0]['text'])
        result = self.call('read_reference_file', path='r1/design.md', start_line=2, limit=1)
        self.assertEqual(result['next_line'], 3)
        self.assertIn('2: 精度', result['lines'][0])

    def test_only_selected_file_and_no_traversal_or_unknown_tools(self):
        outside = self.base/'outside.txt'
        outside.write_text('DO-NOT-READ', encoding='utf-8')
        for path in ('../outside.txt', 'r1/../outside.txt', str(outside), 'r1/C:/outside.txt'):
            result = self.call('read_reference_file', path=path)
            self.assertIn('error', result)
            self.assertNotIn('DO-NOT-READ', str(result))
        self.assertIn('error', self.call('execute_shell', command='echo bad'))
        self.tools = ReferenceTools({'enabled': True, 'paths': [str(self.root/'design.md')]}, self.cancel)
        self.assertIn('lines', self.call('read_reference_file', path='r1/design.md'))
        self.assertIn('error', self.call('read_reference_file', path='r1/other.md'))

    def test_excluded_files_and_nested_symlinks(self):
        (self.root/'.env').write_text('PRIVATE', encoding='utf-8')
        ignored = self.root/'node_modules'
        ignored.mkdir()
        (ignored/'private.md').write_text('PRIVATE', encoding='utf-8')
        self.assertEqual(self.call('list_reference_files')['files'], ['r1/design.md'])
        self.assertIn('error', self.call('read_reference_file', path='r1/.env'))
        outside = self.base/'outside.md'
        outside.write_text('PRIVATE', encoding='utf-8')
        try:
            (self.root/'link.md').symlink_to(outside)
        except OSError:
            return  # Symlink privilege is optional on Windows.
        self.assertNotIn('r1/link.md', self.call('list_reference_files')['files'])
        self.assertIn('error', self.call('read_reference_file', path='r1/link.md'))

    def test_changed_files_are_not_stale_and_budget_is_bounded(self):
        first = self.call('read_reference_file', path='r1/design.md')
        (self.root/'design.md').write_text('updated contents with a different length', encoding='utf-8')
        second = self.call('read_reference_file', path='r1/design.md')
        self.assertNotEqual(first, second)
        (self.root/'big.txt').write_text(('"\\'*500+'\n')*100, encoding='utf-8')
        for _ in range(10):
            raw = self.tools.execute('read_reference_file', '{"path":"r1/big.txt"}')
            self.assertLessEqual(len(raw.encode('utf-8')), 10000)
            json.loads(raw)
        self.assertGreaterEqual(self.tools.remaining, 0)

    def test_docx_extracts_paragraphs_and_pdf_pages(self):
        with zipfile.ZipFile(self.root/'notes.docx', 'w') as archive:
            archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>测试资料</w:t></w:r></w:p></w:body></w:document>')
        self.assertIn('测试资料', str(self.call('read_reference_file', path='r1/notes.docx')))
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({
            NameObject('/F1'): DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 10 100 Td (Reference fact 8421) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        writer.write(self.root/'notes.pdf')
        result = self.call('read_reference_file', path='r1/notes.pdf')
        self.assertIn('8421', str(result))
        self.assertIn('第 1 页', str(result))

    def test_persistence_disabled_and_cancellation(self):
        path = self.base/'settings.json'
        data = {'enabled': True, 'paths': [str(self.root)]}
        save_settings(data, path)
        self.assertEqual(load_settings(path), data)
        self.assertFalse(ReferenceTools({'enabled': False, 'paths': [str(self.root)]}, self.cancel).roots)
        self.cancel.set()
        with self.assertRaisesRegex(RuntimeError, '已取消'):
            self.call('list_reference_files')
