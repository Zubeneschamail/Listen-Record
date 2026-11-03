import codecs
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qa_records import append_qa, load_qa, replace_qa


class QARecordTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/'interview.qa.jsonl'

    def test_append_and_reload_keep_duplicate_turns_independent(self):
        first=append_qa(self.path,'one','为什么？','因为有唯一约束。')
        prefix=self.path.read_bytes()
        append_qa(self.path,'two','为什么？','因为有唯一约束。')
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        records=load_qa(self.path)
        self.assertEqual([r.key for r in records],['one','two'])
        replace_qa(first,'one','第一个为什么？','更新第一个回答。')
        result=load_qa(self.path)
        self.assertEqual([(r.question,r.answer) for r in result],
                         [('第一个为什么？','更新第一个回答。'),('为什么？','因为有唯一约束。')])

    def test_legacy_duplicates_have_stable_independent_origins_without_migration_write(self):
        line=json.dumps(dict(time='旧时间',question='为什么？',answer='旧回答',extra=True),ensure_ascii=False)
        self.path.write_bytes(codecs.BOM_UTF8+(line+'\r\n\r\n'+line).encode('utf-8'))
        raw=self.path.read_bytes()
        records=load_qa(self.path)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertNotEqual(records[0].key,records[1].key)
        self.assertEqual([r.key for r in records],[r.key for r in load_qa(self.path)])
        replace_qa(records[0].origin,records[0].key,'修订第一题','新回答')
        result=self.path.read_bytes()
        self.assertTrue(result.startswith(codecs.BOM_UTF8))
        self.assertTrue(result.endswith(('\r\n\r\n'+line).encode('utf-8')))
        value=json.loads(result.decode('utf-8-sig').splitlines()[0])
        self.assertTrue(value['extra'])
        self.assertEqual(value['time'],'旧时间')
        self.assertIn('updated_at',value)

    def test_other_turn_append_and_reorder_do_not_redirect_an_identified_revision(self):
        origin=append_qa(self.path,'one','第一题','第一答')
        append_qa(self.path,'two','第二题','第二答')
        lines=self.path.read_bytes().splitlines(keepends=True)
        self.path.write_bytes(b''.join(reversed(lines)))
        replace_qa(origin,'one','第一题修订','修订答案')
        self.assertEqual([r.question for r in load_qa(self.path)],['第二题','第一题修订'])

    def test_stale_deleted_or_replaced_turn_never_overwrites_another(self):
        origin=append_qa(self.path,'one','第一题','第一答')
        append_qa(self.path,'two','第二题','第二答')
        replace_qa(origin,'one','外部修订','新的答案')
        newer=self.path.read_bytes()
        with self.assertRaisesRegex(ValueError,'发生变化'):
            replace_qa(origin,'one','过期窗口','不应写入')
        self.assertEqual(self.path.read_bytes(),newer)
        self.path.write_bytes(newer.splitlines(keepends=True)[1])
        remaining=self.path.read_bytes()
        with self.assertRaises(ValueError):
            replace_qa(origin,'one','第一题','不应添加到末尾')
        self.assertEqual(self.path.read_bytes(),remaining)

    def test_malformed_and_duplicate_identity_journals_are_not_rewritten(self):
        valid=dict(id='same',question='问题',answer='答案')
        for content in ('{','[]','{"question":1,"answer":"a"}',
                        json.dumps(valid)+'\n'+json.dumps(valid),'{"question":"q","answer":null}'):
            with self.subTest(content=content):
                self.path.write_text(content,encoding='utf-8')
                original=self.path.read_bytes()
                with self.assertRaises(ValueError):
                    load_qa(self.path)
                self.assertEqual(self.path.read_bytes(),original)

    def test_atomic_replace_failure_preserves_journal_and_removes_temporary_file(self):
        origin=append_qa(self.path,'one','问题','原回答')
        original=self.path.read_bytes()
        with patch.object(Path,'replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                replace_qa(origin,'one','问题','新回答')
        self.assertEqual(self.path.read_bytes(),original)
        self.assertEqual(list(self.path.parent.glob('*.tmp')),[])

    def test_append_after_a_valid_unterminated_line_and_blank_lines(self):
        self.path.write_text(json.dumps(dict(question='旧问题',answer='旧答案')),encoding='utf-8')
        append_qa(self.path,'new','新问题','新答案')
        self.assertEqual([r.question for r in load_qa(self.path)],['旧问题','新问题'])

    def test_missing_journal_is_empty(self):
        self.assertEqual(load_qa(self.path),[])

    def test_append_does_not_rescan_a_long_journal(self):
        append_qa(self.path,'one','问题','答案')
        with patch.object(Path,'read_bytes',side_effect=AssertionError('Do not rescan on append')):
            append_qa(self.path,'two','下一题','下一答')
        self.assertEqual(len(load_qa(self.path)),2)
