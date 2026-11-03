"""Local Q&A journal with stable identities and guarded, atomic revisions."""
import codecs
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from recording_storage import append_bytes


def fingerprint(line):
    return hashlib.sha256(line.rstrip('\r\n').encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class QAOrigin:
    path: Path
    key: str | None
    line: int
    fingerprint: str


@dataclass(frozen=True)
class QARecord:
    key: str
    question: str
    answer: str
    time: str
    origin: QAOrigin


def _parse(path, raw):
    path = Path(path).resolve()
    lines = raw.decode('utf-8-sig').splitlines(keepends=True)
    records, seen = [], set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if (not isinstance(value, dict) or
                    not isinstance(value.get('question'), str) or not value['question'].strip() or
                    not isinstance(value.get('answer'), str) or
                    not isinstance(value.get('time', ''), str)):
                raise ValueError()
            key = value.get('id')
            if key is not None and (not isinstance(key, str) or not key.strip()):
                raise ValueError()
            origin = QAOrigin(path, key, index, fingerprint(line))
            key = key or 'legacy-'+hashlib.sha256(
                f'{path}\0{index}\0{origin.fingerprint}'.encode('utf-8')).hexdigest()
            if key in seen:
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise ValueError(f'问答记录第 {index+1} 行格式不正确；原文件未改动。') from exc
        seen.add(key)
        records.append(QARecord(key, value['question'], value['answer'], value.get('time', ''), origin))
    return lines, records


def load_qa(path):
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    return _parse(path, raw)[1]


def append_qa(path, key, question, answer, *, timestamp=None):
    path = Path(path).resolve()
    record = dict(id=key, time=timestamp if timestamp is not None else datetime.now().astimezone().isoformat(), question=question, answer=answer)
    line = json.dumps(record, ensure_ascii=False)
    # Append without scanning or rewriting an increasingly long interview.
    append_bytes(path, (line+'\n').encode('utf-8'), ensure_newline=True)
    return QAOrigin(path, key, -1, fingerprint(line))


def replace_qa(origin, key, question, answer):
    raw = origin.path.read_bytes()
    lines, records = _parse(origin.path, raw)
    matches = [record for record in records if
               (record.origin.key == origin.key if origin.key is not None else record.origin.line == origin.line)]
    if len(matches) != 1 or matches[0].origin.fingerprint != origin.fingerprint:
        raise ValueError('这条问答已在文件中发生变化，请重新打开记录后再修改；原文件未改动。')
    index = matches[0].origin.line
    previous = lines[index]
    value = json.loads(previous)
    value.update(id=key, question=question, answer=answer,
                 updated_at=datetime.now().astimezone().isoformat())
    line = json.dumps(value, ensure_ascii=False)
    newline = '\r\n' if previous.endswith('\r\n') else '\n' if previous.endswith('\n') else ''
    lines[index] = line+newline
    updated = (codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b'')+''.join(lines).encode('utf-8')
    _write_atomic(origin.path, updated)
    return QAOrigin(origin.path, key, index, fingerprint(line))


def save_qa_copy(path, records):
    """Write a validated snapshot; callers choose and protect the destination."""
    path = Path(path).resolve()
    if not records:
        raise ValueError('尚无已完成问答可保存。')
    raw = ''.join(json.dumps(record, ensure_ascii=False)+'\n' for record in records).encode('utf-8')
    _parse(path, raw)
    _write_atomic(path, raw)


def _write_atomic(path, raw):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            file.write(raw)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
