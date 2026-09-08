import unittest
from types import SimpleNamespace
from tokenizers import Tokenizer, models, pre_tokenizers
from faster_whisper import WhisperModel
from segmentation import budget_prompt

class PromptBudgetTests(unittest.TestCase):
    def test_combined_hotwords_and_history_leave_room_for_output(self):
        tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
        model = SimpleNamespace(hf_tokenizer=tokenizer, max_length=448)
        wrapper = SimpleNamespace(sot_prev=1, sot_sequence=[2,3,4], no_timestamps=5,
                                  encode=lambda text: tokenizer.encode(text).ids)
        for history in [None, '长上下文'*160]:
            for terms in ['', '大模型,LangChain,MCP,'*100]:
                original = dict(initial_prompt=history, hotwords=terms, beam_size=3)
                options = budget_prompt(model, original)
                self.assertEqual(original['initial_prompt'], history)
                for draft in [False, True]:
                    prompt = WhisperModel.get_prompt(model, wrapper,
                        options.get('initial_prompt') or [], without_timestamps=draft,
                        hotwords=options.get('hotwords'))
                    self.assertLessEqual(len(prompt)+256, 448)
                self.assertEqual(options['beam_size'], 3)
