import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as Item
import unittest
from unittest.mock import Mock
import numpy as np

from recognition import RecognitionContext, parse_hotwords, read_preferences, save_preferences
from segmentation import AudioChunk, WordAssembler, DraftPreview, PauseSegmenter, decode_chunk, RATE


class RecognitionTests(unittest.TestCase):
    def test_hotwords_are_bounded_and_empty_can_disable_them(self):
        self.assertEqual(parse_hotwords("大模型，LangChain; langchain\nMCP"), ["大模型", "LangChain", "MCP"])
        self.assertLessEqual(len(", ".join(parse_hotwords(",".join("词"*30+str(i) for i in range(99))))), 256)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            save_preferences(path, "")
            self.assertEqual(read_preferences(path), "")
            save_preferences(path, "MCP，LangGraph")
            self.assertEqual(read_preferences(path), "MCP, LangGraph")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["hotwords"], "MCP, LangGraph")

    def test_context_is_bounded_expires_and_backlog_reduces_beam(self):
        context = RecognitionContext("大模型")
        for i in range(6):
            context.commit([(0, 1, str(i)*100, 0)], end=10, elapsed=0.1, duration=2)
        self.assertLessEqual(len(context.options(11)["initial_prompt"]), 160)
        self.assertEqual(len(context.history), 3)
        self.assertEqual(context.options(11)["beam_size"], 3)
        self.assertEqual(context.options(11, backlog=2)["beam_size"], 1)
        context.rtf = 1
        self.assertEqual(context.options(11)["beam_size"], 1)
        self.assertIsNone(context.options(40)["initial_prompt"])
        self.assertFalse(context.history)

    def test_term_normalization_does_not_replace_similar_words(self):
        context = RecognitionContext("LangChain, MCP, Agent, 大模型")
        self.assertEqual(context.normalize("lang chain与m c p"), "LangChain与MCP")
        self.assertEqual(context.normalize("Agents agentic 大模形"), "Agents agentic 大模形")

    def test_final_uses_guidance_but_preview_never_commits_it(self):
        context = RecognitionContext("LangChain")
        segmenter = PauseSegmenter(lambda a: [{"start": 0, "end": len(a)}])
        segmenter.push(np.ones(RATE*2, dtype=np.float32), 0, 100)
        model = Mock()
        segment = Item(text="lang chain", words=[Item(start=0, end=1, word="lang chain")])
        model.transcribe.return_value = ([segment], None)
        assembler = WordAssembler()
        preview = DraftPreview(clock=lambda: 0)
        self.assertEqual(preview.render(model, assembler, segmenter, "zh", guidance=context), "LangChain")
        self.assertFalse(context.history)
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 1)
        rows = decode_chunk(model, assembler, segmenter.snapshot(), "zh", guidance=context)
        self.assertEqual(rows[0][2], "LangChain")
        self.assertEqual(list(context.history), ["LangChain"])
        self.assertEqual(model.transcribe.call_args.kwargs["hotwords"], "LangChain")
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 3)

    def test_expensive_draft_increases_cooldown(self):
        clock = [0.0]
        segmenter = PauseSegmenter(lambda a: [{"start": 0, "end": len(a)}])
        segmenter.push(np.ones(RATE*2, dtype=np.float32), 0, 100)
        model = Mock()
        def infer(*args, **kwargs):
            clock[0] += 3
            return [Item(text="草稿")], None
        model.transcribe.side_effect = infer
        preview = DraftPreview(clock=lambda: clock[0])
        preview.render(model, WordAssembler(), segmenter, "zh")
        self.assertEqual(preview.cooldown, 4.5)
        self.assertEqual(preview.last_finished, 3)


if __name__ == "__main__":
    unittest.main()
