"""Bounded-memory SQLite vector ranking."""
import heapq

MAX_SIZE = 8 * 1024 ** 3
MAX_CHUNKS = 1_000_000
MAX_MANIFEST = 64 * 1024 ** 2
SCAN_BATCH = 512


def semantic_top(connection, query_vector, top_k=20, check=lambda: None, row_ids=None):
    import numpy as np
    candidates = []
    count = 0
    if row_ids is None:
        cursor = connection.execute('SELECT rowid,vector FROM chunks ORDER BY rowid')
    else:
        row_ids = list(dict.fromkeys(row_ids))
        if not 1 <= len(row_ids) <= SCAN_BATCH:
            raise ValueError('知识包候选片段数量无效。')
        cursor = connection.execute(
            'SELECT rowid,vector FROM chunks WHERE rowid IN (' + ','.join('?' for _ in row_ids) + ') ORDER BY rowid',
            row_ids)
    while True:
        check()
        rows = cursor.fetchmany(SCAN_BATCH)
        if not rows:
            break
        count += len(rows)
        if count > MAX_CHUNKS:
            raise ValueError('知识包超过 100 万片段，请按项目拆分。')
        if any(not isinstance(row['vector'], bytes) or len(row['vector']) != 2048 for row in rows):
            raise ValueError('知识包向量长度无效。')
        matrix = np.frombuffer(b''.join(row['vector'] for row in rows), dtype='<f4').reshape(len(rows), 512)
        if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.01):
            raise ValueError('知识包向量无效。')
        scores = matrix @ query_vector
        for index in np.argsort(-scores, kind='stable')[:top_k]:
            candidate = (float(scores[index]), -rows[index]['rowid'])
            if len(candidates) < top_k:
                heapq.heappush(candidates, candidate)
            elif candidate > candidates[0]:
                heapq.heapreplace(candidates, candidate)
    check()
    if not count:
        raise ValueError('知识包为空。')
    if row_ids is not None and count != len(row_ids):
        raise ValueError('知识包关键词索引与原文不一致。')
    ranked = sorted(candidates, reverse=True)
    return [-rowid for _, rowid in ranked], {-rowid: score for score, rowid in ranked}
