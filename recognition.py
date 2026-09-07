"""Bounded terminology/context guidance for local speech decoding."""
from collections import deque
import json
from pathlib import Path
import re

DEFAULT_HOTWORDS = "大模型, 人工智能, LangChain, LangGraph, MCP, Agent, Skill, Codex, Whisper, VibeVoice"


def parse_hotwords(value):
    terms, seen = [], set()
    for term in re.split(r"[,，;；\n]+", value):
        term = " ".join(term.split())[:40]
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            if len(", ".join(terms + [term])) > 256 or len(terms) >= 24:
                break
            terms.append(term)
    return terms


def read_preferences(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("hotwords"), str):
            return DEFAULT_HOTWORDS
        return ", ".join(parse_hotwords(value["hotwords"]))
    except (OSError, ValueError):
        return DEFAULT_HOTWORDS


def save_preferences(path, hotwords):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"hotwords": ", ".join(parse_hotwords(hotwords))},
                                    ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class RecognitionContext:
    def __init__(self, hotwords=""):
        self.terms = parse_hotwords(hotwords)
        self.history = deque(maxlen=3)
        self.last_end = None
        self.rtf = 0.0

    def options(self, start, backlog=0, draft=False):
        # A new topic after a long pause must not inherit the old wording.
        expired = self.last_end is not None and start - self.last_end > 20
        if expired and not draft:
            self.history.clear()
        return dict(hotwords=", ".join(self.terms) or None,
                    initial_prompt=(" ".join(self.history)[-160:] or None) if not expired else None,
                    beam_size=1 if draft or backlog >= 0.5 or self.rtf >= 0.65 else 3)

    def commit(self, rows, end, elapsed, duration):
        for _, _, text, _ in rows:
            text = text.strip()
            if text and (not self.history or self.history[-1] != text):
                self.history.append(text[-160:])
        self.last_end = end
        self.rtf = 0.7 * self.rtf + 0.3 * elapsed / max(0.1, duration)

    def normalize(self, text):
        # Restore only spelling/case/spacing of explicit Latin terms. No fuzzy
        # replacement: similar sounds can denote different legitimate words.
        for term in self.terms:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 -]*", term):
                continue
            compact = re.sub(r"\s+", "", term)
            pattern = r"(?<![A-Za-z0-9])" + r"\s*".join(map(re.escape, compact)) + r"(?![A-Za-z0-9])"
            text = re.sub(pattern, lambda _: term, text, flags=re.I)
        return text
