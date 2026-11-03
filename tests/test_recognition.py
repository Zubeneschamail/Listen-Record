import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as Item
import unittest
from unittest.mock import Mock
import numpy as np

from recognition import (RecognitionContext, LEGACY_DEFAULT_HOTWORDS,
                         parse_hotwords, read_preferences, save_preferences,
                         INTERVIEW_HOTWORDS, add_interview_hotwords)
from segmentation import AudioChunk, WordAssembler, DraftPreview, PauseSegmenter, decode_chunk, RATE


class RecognitionTests(unittest.TestCase):
    def test_correction_rules_do_not_change_decoder_guidance(self):
        before = RecognitionContext('lang chain=LangGraph, C++, C#, .NET')
        after = RecognitionContext('秋陵=丘陵, '+ 'lang chain=LangGraph, C++, C#, .NET')
        self.assertEqual(before.options(0), after.options(0))
        self.assertEqual(after.options(0)['hotwords'], 'C++, C#, .NET')
        self.assertEqual(after.normalize('市区西北部则是秋陵'), '市区西北部则是丘陵')
        self.assertEqual(RecognitionContext('密等性=幂等性').options(0)['hotwords'], None)
        self.assertEqual(RecognitionContext('密等性=幂等性, 幂等性').options(0)['hotwords'], '幂等性')
        self.assertEqual(after.normalize('lang chain与langgraph'), 'LangGraph与LangGraph')

    def test_long_explicit_alias_is_preserved_instead_of_truncating_its_target(self):
        wrong = '项目中检索增强生成资料处理流水线的技术名称'
        right = '项目中检索增强生成资料处理流水线的正确技术名称'
        value = wrong+'='+right
        self.assertGreater(len(value), 40)
        self.assertEqual(parse_hotwords(value), [value])
        self.assertEqual(RecognitionContext(value).normalize(wrong), right)
        self.assertEqual(parse_hotwords('词'*300+', Redis'), ['Redis'])

    def test_interview_presets_preserve_user_aliases_and_fit_prompt_limits(self):
        for topic, terms in INTERVIEW_HOTWORDS.items():
            self.assertEqual(", ".join(parse_hotwords(terms)), terms)
            merged = add_interview_hotwords("密等性=幂等性, LangChain", topic)
            self.assertTrue(merged.startswith("密等性=幂等性, LangChain"))
            self.assertLessEqual(len(merged), 256)
            self.assertEqual(add_interview_hotwords(merged, topic), merged)

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

    def test_untouched_legacy_hotwords_are_cleared(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            self.assertEqual(read_preferences(path), "")
            path.write_text(json.dumps({"hotwords": LEGACY_DEFAULT_HOTWORDS}), encoding="utf-8")
            self.assertEqual(read_preferences(path), "")
            path.write_text(json.dumps({"hotwords": "大模型, 自定义词"}), encoding="utf-8")
            self.assertEqual(read_preferences(path), "大模型, 自定义词")

    def test_recognized_text_is_not_replayed_and_backlog_reduces_beam(self):
        context = RecognitionContext("大模型")
        for i in range(6):
            context.commit([(0, 1, str(i)*100, 0)], end=10, elapsed=0.1, duration=2)
        self.assertNotIn("initial_prompt", context.options(11))
        self.assertEqual(context.options(11)["hotwords"], "大模型")
        self.assertEqual(context.options(11)["beam_size"], 3)
        self.assertEqual(context.options(11, backlog=2)["beam_size"], 1)
        context.rtf = 1
        self.assertEqual(context.options(11)["beam_size"], 1)
        self.assertNotIn("initial_prompt", context.options(40))

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
        self.assertEqual(context.rtf, 0)
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 1)
        rows = decode_chunk(model, assembler, segmenter.snapshot(), "zh", guidance=context)
        self.assertEqual(rows[0][2], "LangChain")
        self.assertEqual(model.transcribe.call_args.kwargs["hotwords"], "LangChain")
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 3)
        self.assertNotIn("initial_prompt", model.transcribe.call_args.kwargs)

    def test_technical_terms_with_symbols_restore_spelling_without_rewriting_other_names(self):
        context = RecognitionContext("C++, C#, .NET, Vue.js, GPT-4o, RAG")
        self.assertEqual(context.normalize("c + +、c #、. net、vue . js、gpt - 4o与r a g"),
                         "C++、C#、.NET、Vue.js、GPT-4o与RAG")
        self.assertEqual(context.normalize("C++17、C#Script、Vue.jsx、GraphRAG、file.NET、ragged"),
                         "C++17、C#Script、Vue.jsx、GraphRAG、file.NET、ragged")

    def test_explicit_aliases_take_priority_and_respect_latin_word_boundaries(self):
        context = RecognitionContext("lang chain=LangGraph, LangChain, sql=SQL数据库, 密等性=幂等性")
        self.assertEqual(context.normalize("LANG CHAIN、SQL、MySQL、NoSQL、密等性"),
                         "LangGraph、SQL数据库、MySQL、NoSQL、幂等性")
        self.assertEqual(RecognitionContext('I=接口').normalize('i、ı、İ'), '接口、ı、İ')

    def test_explicit_aliases_preserve_qualified_names_and_sentence_periods(self):
        context = RecognitionContext('net=Network, .NET=DotNet, C++=Cpp')
        self.assertEqual(context.normalize('file.NET net.io .NET.dll C++17 file.C++'),
                         'file.NET net.io .NET.dll C++17 file.C++')
        self.assertEqual(context.normalize('net. NET？.NET。C++.'), 'Network. Network？DotNet。Cpp.')

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
