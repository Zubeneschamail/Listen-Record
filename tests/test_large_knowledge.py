from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

import numpy as np

import knowledge_packages as kb
from knowledge_embedding import Encoder
from knowledge_storage import semantic_top, SCAN_BATCH
from test_knowledge_packages import write_package


class LargeKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = Encoder()

    def test_streamed_scan_matches_dense_ranking_and_cancels(self):
        # Deterministic normalized numerical fixture, independent of model quality.
        rng = np.random.default_rng(19)
        vectors = rng.normal(size=(SCAN_BATCH * 3 + 7, 512)).astype('<f4')
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        query = vectors[-1]
        with closing(sqlite3.connect(':memory:')) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('CREATE TABLE chunks(vector BLOB)')
            connection.executemany('INSERT INTO chunks VALUES (?)', ((v.tobytes(),) for v in vectors))
            rows, scores = semantic_top(connection, query)
            expected = np.argsort(-(vectors @ query), kind='stable')[:20] + 1
            self.assertEqual(rows, expected.tolist())
            self.assertEqual(rows[0], len(vectors))
            np.testing.assert_allclose([scores[int(i)] for i in expected], (vectors @ query)[expected - 1], atol=1e-6)
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                semantic_top(connection, query, check=cancel)

    def test_large_package_and_legacy_oversized_manifest(self):
        # Exercise all former reader limits in one real SQLite package (>256 MB,
        # >50k chunks, >32M text characters, >1M manifest characters).
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'large.wlkb'
            manifest = write_package(path, self.encoder)
            body = 'cache document ' * 140
            vector = self.encoder.encode([body])[0].astype('<f4').tobytes()
            count = 70001
            with closing(sqlite3.connect(path)) as connection:
                connection.executemany('INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)',
                    ((f'chunk{i}', 'bulk.md', '缓存', None, 1, 1, body, 'bulk', vector) for i in range(2, count + 1)))
                manifest.update(chunk_count=count, skipped=[{'path': 'x' * 1_100_000, 'reason': 'legacy'}])
                connection.execute('UPDATE metadata SET value=?', (json.dumps(manifest),))
                connection.commit()
            self.assertGreater(path.stat().st_size, 256 * 1024**2)
            self.assertEqual(kb.inspect_package(path)['chunk_count'], count)
            result = kb.KnowledgeLibrary().context({'enabled': True, 'paths': [str(path)]},
                                                  '数据库提交后消息未发出怎么办？', [], threading.Event())
            self.assertIn('outbox', result)
            self.assertLess(len(result.encode()), kb.MAX_CONTEXT_BYTES + 300)
