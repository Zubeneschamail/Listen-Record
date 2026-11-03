"""Explicitly saved correction aliases; never infer rules from transcript edits."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile

from recognition import LEGACY_DEFAULT_HOTWORDS, MAX_HOTWORD_CHARS, MAX_HOTWORD_ENTRIES


@dataclass(frozen=True)
class RulePreferences:
    hotwords: str
    fingerprint: str | None
    revision: str | None = None


class InvalidRulePreferences(ValueError):
    def __init__(self, message, revision):
        super().__init__(message)
        self.revision = revision


def _read(path):
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return {}, None, None
    revision = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw.decode('utf-8'))
        if not isinstance(value, dict) or not isinstance(value.get('hotwords', ''), str):
            raise ValueError('Invalid hotwords')
    except (ValueError, UnicodeError) as exc:
        raise InvalidRulePreferences('现有词库无法读取，请先在设置中核对；原文件未改动。', revision) from exc
    # Ordinary app saves may change JSON indentation/key order without changing
    # the word list. Detect changed data, not those harmless serialization edits.
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return value, hashlib.sha256(canonical).hexdigest(), revision


def load_rule_preferences(path):
    value, fingerprint, revision = _read(path)
    hotwords = value.get('hotwords', '')
    return RulePreferences('' if hotwords == LEGACY_DEFAULT_HOTWORDS else hotwords, fingerprint, revision)


def _entries(value):
    entries, seen = [], set()
    for entry in re.split(r'[,，;；\n]+', value):
        entry = ' '.join(entry.split())
        if entry and entry.casefold() not in seen:
            entries.append(entry)
            seen.add(entry.casefold())
    if len(entries) > MAX_HOTWORD_ENTRIES or len(', '.join(entries)) > MAX_HOTWORD_CHARS:
        raise ValueError('词库已满，请先在设置中精简；不会自动移除已有词或规则。')
    return entries


def prepare_rule(value, wrong, right, replace=False):
    for word in (wrong, right):
        if not isinstance(word, str) or re.search(r'[,，;；\r\n=]', word):
            raise ValueError('错词和正确词不能包含逗号、分号、换行或等号。')
    wrong, right = (' '.join(word.split()) for word in (wrong, right))
    if not wrong or not right:
        raise ValueError('请填写完整的错词和正确词。')
    if wrong == right:
        raise ValueError('错词和正确词相同，无需新增规则。')
    entry = wrong+'='+right
    if len(entry) > MAX_HOTWORD_CHARS:
        raise ValueError('这条规则太长，请只填写需要纠正的词语。')
    entries = _entries(value)
    aliases = [item for item in entries if '=' in item and
               item.split('=', 1)[0].strip().casefold() == wrong.casefold()]
    if aliases and all(item.split('=', 1)[1].strip() == right for item in aliases):
        return ', '.join(entries)
    if aliases and not replace:
        raise ValueError('该错词已有规则：'+ '、'.join(aliases)+'。如需更新，请勾选替换已有规则。')
    others = [item for item in entries if item not in aliases]
    return ', '.join(_entries(', '.join([entry]+others)))


def save_rule(path, wrong, right, expected, replace=False):
    path = Path(path)
    value, fingerprint, _ = _read(path)
    if fingerprint != expected:
        raise ValueError('词库已在其他地方发生变化，请重新打开此窗口后保存。')
    old = value.get('hotwords', '')
    base = '' if old == LEGACY_DEFAULT_HOTWORDS else old
    updated = prepare_rule(base, wrong, right, replace)
    if updated == old:
        return updated
    value['hotwords'] = updated
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return updated
