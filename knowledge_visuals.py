"""Bounded, package-embedded page images for locally retrieved text evidence."""
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3

from knowledge_packages import _open, package_paths

FORMAT = 'wenlu-kb-visual-pages-v1'
MAX_IMAGE_BYTES = 3_000_000


def visual_evidence(settings, context, cancel=None, limit=2):
    """Read only images linked to exact retrieved chunk ranges, never disk paths in text."""
    try:
        selected = json.loads(context).get('片段', [])
    except (ValueError, AttributeError, TypeError):
        return []
    if not isinstance(selected, list):
        return []
    result, seen = [], set()
    for index, value in enumerate(package_paths(settings), 1):
        if cancel is not None and cancel.is_set():
            raise RuntimeError('已取消')
        path = Path(value).resolve(strict=True)
        prefix = f'kb{index}/{path.name}/'
        candidates = [item for item in selected if isinstance(item, dict)
                      and isinstance(item.get('来源'), str) and item['来源'].startswith(prefix)]
        if not candidates:
            continue
        connection, manifest = _open(path, cancel)
        try:
            if manifest.get('visual_attachment_format') != FORMAT:
                continue
            for table in ('kb_visual_pages', 'kb_chunk_visuals'):
                row = connection.execute('SELECT type,sql FROM sqlite_master WHERE name=?', (table,)).fetchone()
                if row is None or row['type'] != 'table' or 'virtual table' in row['sql'].lower():
                    raise ValueError('知识包原图索引无效，请重新生成知识包。')
            for item in candidates:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError('已取消')
                start, end = item.get('起始行'), item.get('结束行')
                if type(start) is not int or type(end) is not int:
                    continue
                rows = connection.execute(
                    'SELECT p.document,p.page,p.sha256,length(p.image) AS bytes FROM kb_chunk_visuals v '
                    'JOIN kb_visual_pages p ON p.document=v.document AND p.page=v.page '
                    'WHERE v.source=? AND v.line_start=? AND v.line_end=? ORDER BY p.page LIMIT 2',
                    (item['来源'][len(prefix):], start, end)).fetchall()
                for row in rows:
                    if (not isinstance(row['document'], str) or not 1 <= len(row['document']) <= 128
                            or type(row['page']) is not int or not 1 <= row['page'] <= 10000
                            or not isinstance(row['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', row['sha256'])):
                        raise ValueError('知识包原图元数据无效。')
                    key = (str(path), row['document'], row['page'])
                    if key in seen:
                        continue
                    if type(row['bytes']) is not int or not 0 < row['bytes'] <= MAX_IMAGE_BYTES:
                        raise ValueError('知识包原图超过读取限制。')
                    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_IMAGE_BYTES + 65536)
                    data = connection.execute('SELECT image FROM kb_visual_pages WHERE document=? AND page=?',
                                              (row['document'], row['page'])).fetchone()[0]
                    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != row['sha256']:
                        raise ValueError('知识包原图校验失败。')
                    from PIL import Image
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format != 'JPEG' or image.width*image.height > 8_000_000:
                            raise ValueError('知识包原图格式或尺寸无效。')
                        image.verify()
                    seen.add(key)
                    result.append({'label': f"{manifest['name']} / {row['document']}原第{row['page']}页",
                                   'mime': 'image/jpeg', 'data': data})
                    if len(result) >= max(1, min(limit, 2)):
                        return result
        finally:
            connection.close()
    return result
