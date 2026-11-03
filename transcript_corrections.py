"""User-confirmed transcript overlays; original JSONL/TXT files stay intact."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile

MAX_CORRECTION = 6000


@dataclass(frozen=True)
class RowOrigin:
    path: Path
    index: int
    fingerprint: str


@dataclass
class Recording:
    rows: list
    origins: list
    corrected: set
    warnings: list


def fingerprint(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def row_origin(path, index, row):
    return RowOrigin(Path(path).resolve(), index, fingerprint(row))


def read_rows(path):
    # Validate everything needed for display AND subtitle export before the
    # caller clears the current practice. A wall clock can hide bad relative
    # times until a much later SRT export.
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'转录记录第 {index} 行不是完整的 JSON，请核对文件。') from exc
        if not isinstance(row, dict) or not isinstance(row.get('text'), str):
            raise ValueError(f'转录记录第 {index} 行的文字格式不正确。')
        if row.get('source', 'system') not in ('system', 'microphone'):
            raise ValueError(f'转录记录第 {index} 行的声音来源不正确。')
        if 'incomplete' in row and not isinstance(row['incomplete'], bool):
            raise ValueError(f'转录记录第 {index} 行的中断标记格式不正确。')
        try:
            times = [row.get(key) for key in ('start', 'end')]
            if (any(type(value) not in (int, float) or not math.isfinite(value*1000) for value in times)
                    or times[1] < times[0]):
                raise ValueError()
            clock = row.get('captured_at')
            if clock is not None and clock != '':
                if not isinstance(clock, str):
                    raise ValueError()
                datetime.fromisoformat(clock)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'转录记录第 {index} 行的时间格式不正确，请核对开始、结束和采集时间。') from exc
        rows.append(row)
    return rows


def corrections_path(path):
    return Path(path).with_suffix('.corrections.json')


def _read_entries(path):
    try:
        value = json.loads(corrections_path(path).read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return {}
    except (ValueError, UnicodeError) as exc:
        raise ValueError('无法读取已有修正，原记录仍保留。') from exc
    if not isinstance(value, dict) or value.get('version') != 1 or not isinstance(value.get('rows'), dict):
        raise ValueError('修正记录格式不正确，原记录仍保留。')
    for key, entry in value['rows'].items():
        if (not isinstance(key, str) or not key.isascii() or not key.isdecimal() or
                len(key) > 10 or key != str(int(key)) or
                not isinstance(entry, dict) or not isinstance(entry.get('original_sha256'), str) or
                len(entry['original_sha256']) != 64 or not isinstance(entry.get('text'), str) or
                not entry['text'].strip() or len(entry['text']) > MAX_CORRECTION):
            raise ValueError('修正记录格式不正确，原记录仍保留。')
    return value['rows']


def load_recording(path):
    original = read_rows(path)
    rows = [dict(row) for row in original]
    origins = [row_origin(path, index, row) for index, row in enumerate(original)]
    warnings, corrected = [], set()
    try:
        entries = _read_entries(path)
    except (OSError, ValueError) as exc:
        return Recording(rows, origins, corrected, [str(exc)])
    for key, entry in entries.items():
        index = int(key)
        if index >= len(rows) or entry['original_sha256'] != origins[index].fingerprint:
            warnings.append('有修正与原转录不匹配，已保留原文。')
            continue
        if entry['text'] != original[index]['text']:
            rows[index]['text'] = entry['text']
            corrected.add(index)
    return Recording(rows, origins, corrected, warnings)


def original_row(origin):
    rows = read_rows(origin.path)
    if origin.index < 0 or origin.index >= len(rows) or fingerprint(rows[origin.index]) != origin.fingerprint:
        raise ValueError('原转录已发生变化，请重新打开记录后再修正。')
    return rows[origin.index]


def save_correction(origin, text):
    text = text.strip()
    if not text or len(text) > MAX_CORRECTION:
        raise ValueError(f'请输入修正后的文字，最多 {MAX_CORRECTION} 字。')
    original = original_row(origin)
    entries = _read_entries(origin.path)
    key = str(origin.index)
    if key in entries and entries[key]['original_sha256'] != origin.fingerprint:
        raise ValueError('已有修正与原转录不匹配，请先核对记录文件。')
    if text == original['text']:
        entries.pop(key, None)
    else:
        entries[key] = dict(original_sha256=origin.fingerprint, text=text,
                            edited_at=datetime.now().astimezone().isoformat())
    path = corrections_path(origin.path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            json.dump(dict(version=1, rows=entries), file, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return text != original['text']
