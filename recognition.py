"""Bounded terminology/context guidance for local speech decoding."""
import json
import hashlib
from pathlib import Path
import re
import threading
import tempfile

LEGACY_DEFAULT_HOTWORDS = "大模型, 人工智能, LangChain, LangGraph, MCP, Agent, Skill, Codex, Whisper, VibeVoice"
DEFAULT_HOTWORDS = ""
MAX_HOTWORD_CHARS = 256
MAX_HOTWORD_ENTRIES = 24
INTERVIEW_HOTWORDS = {
    "AI / 大模型": "大语言模型, 检索增强生成, 向量数据库, 嵌入向量, 提示词, 微调, 推理, 多模态, RAG, LangChain, LangGraph, MCP, Agent, Embedding",
    "后端开发": "微服务, 消息队列, 异步, 幂等性, 分布式, 事务, 索引, 缓存, PostgreSQL, MySQL, Redis, FastAPI, Docker, Kubernetes",
    "前端开发": "TypeScript, JavaScript, React, Vue.js, Next.js, Node.js, Webpack, Vite, 组件, 路由, 状态管理, 服务端渲染",
    "GIS / 三维": "GIS, WebGIS, 三维GIS, 空间索引, 坐标系, 经纬度, 点云, 倾斜摄影, 数字孪生, Cesium, PostGIS, GeoServer",
}


def parse_hotwords(value):
    terms, seen = [], set()
    for term in re.split(r"[,，;；\n]+", value):
        term = " ".join(term.split())
        if len(term) > MAX_HOTWORD_CHARS:
            continue  # Reject oversized entries instead of inventing a partial alias.
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            if len(", ".join(terms + [term])) > MAX_HOTWORD_CHARS or len(terms) >= MAX_HOTWORD_ENTRIES:
                break
            terms.append(term)
    return terms


def add_interview_hotwords(value, topic):
    """Keep the user's aliases first, then add relevant terms within the budget."""
    return ", ".join(parse_hotwords(value + ", " + INTERVIEW_HOTWORDS[topic]))


def read_preferences(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("hotwords"), str):
            return DEFAULT_HOTWORDS
        # Older releases prefilled this list. It can turn background noise into
        # the same suggested word on every chunk, so migrate the untouched default.
        if value["hotwords"] == LEGACY_DEFAULT_HOTWORDS:
            return ""
        return ", ".join(parse_hotwords(value["hotwords"]))
    except (OSError, ValueError):
        return DEFAULT_HOTWORDS


class PreferenceConflictError(ValueError):
    """An explicit settings draft must not replace a newer saved word list."""


def preferences_revision(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def load_wordlist_snapshot(path):
    # Unlike the initial UI fallback, capture must not silently ignore a damaged
    # saved word list. Loading is read-only, including legacy-default migration.
    from recognition_rules import load_rule_preferences
    saved = load_rule_preferences(path)
    return ', '.join(parse_hotwords(saved.hotwords)), saved.revision


def load_capture_preferences(path):
    return load_wordlist_snapshot(path)[0]


def save_preferences(path, hotwords):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError, UnicodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    value['hotwords'] = ', '.join(parse_hotwords(hotwords))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CorrectionUpdates:
    """One bounded, immutable preference snapshot shared by capture sources."""
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def reset(self):
        with self._lock:
            self._value = None

    def publish(self, value):
        if not isinstance(value, str):
            raise ValueError('Correction preferences must be text')
        value = ', '.join(parse_hotwords(value))
        with self._lock:
            self._value = value

    def snapshot(self):
        with self._lock:
            return self._value

    def __deepcopy__(self, memo):
        # Drafts copy their decoding context but follow this capture's updates.
        return self


class RecognitionContext:
    def __init__(self, hotwords="", *, correction_source=None):
        entries = parse_hotwords(hotwords)
        self.corrections = {}
        self.terms = []
        self.hint_terms = []
        for entry in entries:
            if '=' in entry:
                wrong, right = (part.strip() for part in entry.split('=', 1))
                if wrong and right:
                    self.corrections[wrong] = right
                    self.terms.append(right)
            else:
                self.terms.append(entry)
                self.hint_terms.append(entry)
        # Explicit correction targets win over conflicting canonical hotword
        # spellings, both live and after restarting the capture.
        self.terms = self.hint_terms + list(self.corrections.values())
        self.rtf = 0.0
        self.correction_source = correction_source
        self._live_preferences = None

    def options(self, start, backlog=0, draft=False):
        # Do not feed recognized text back as an initial prompt. A hallucinated
        # phrase can otherwise reinforce itself across separate audio chunks.
        # Explicit replacement rules correct output only. Adding a correction
        # must not change the decoder prompt and damage unrelated words.
        # A standalone term explicitly opts in to acoustic guidance.
        return dict(hotwords=", ".join(self.hint_terms) or None,
                    beam_size=1 if draft or backlog >= 1.5 or self.rtf >= 0.65 else 3)

    def commit(self, rows, end, elapsed, duration):
        self.rtf = 0.7 * self.rtf + 0.3 * elapsed / max(0.1, duration)

    def normalize(self, text):
        # The UI only publishes immutable preferences after a successful save.
        # Each worker normalization uses one complete snapshot; hints and RTF
        # remain those of the active capture, including during a slow decode.
        if self.correction_source is not None:
            value = self.correction_source()
            if value is not None and value != self._live_preferences:
                replacement = RecognitionContext(value)
                self.corrections = replacement.corrections
                self.terms = self.hint_terms + list(self.corrections.values())
                self._live_preferences = value
        # Apply explicit aliases before canonical spacing; otherwise a hotword
        # such as LangChain can hide the user's "lang chain=LangGraph" rule.
        if self.corrections:
            aliases = []
            for word in sorted(self.corrections, key=len, reverse=True):
                pattern = re.escape(word)
                if re.match(r"[A-Za-z0-9.+#/_-]", word):
                    pattern = r"(?<![A-Za-z0-9.+#/_-])" + pattern
                if re.search(r"[A-Za-z0-9.+#/_-]$", word):
                    # Keep qualified names/versions intact, but allow the period
                    # that ends a spoken sentence (net. vs net.io).
                    pattern += r"(?![A-Za-z0-9+#/_-]|\.(?=[A-Za-z0-9]))"
                aliases.append(pattern)
            replacements = {word.casefold(): value for word, value in self.corrections.items()}
            text = re.sub('|'.join(aliases), lambda match: replacements[match.group().casefold()],
                          text, flags=re.I | re.ASCII)
        # Restore only spelling/case/spacing of explicit Latin terms. No fuzzy
        # replacement: similar sounds can denote different legitimate words.
        for term in self.terms:
            if (not re.fullmatch(r"[A-Za-z0-9.+#/_ -]+", term) or
                    not re.search(r"[A-Za-z]", term)):
                continue
            compact = re.sub(r"\s+", "", term)
            pattern = r"(?<![A-Za-z0-9.+#/_-])" + r"\s*".join(map(re.escape, compact)) + r"(?![A-Za-z0-9+#/_-])"
            text = re.sub(pattern, lambda _: term, text, flags=re.I)
        return text
