import unittest

from tools.asr_metrics import contains_term, mixed_tokens, score_text


class ASRMetricsTests(unittest.TestCase):
    def test_symbol_error_is_visible_when_character_metric_hides_it(self):
        result = score_text('用C++实现。', '用C#实现。', ['C++'])
        self.assertEqual(result['cer'], 0)
        self.assertEqual(result['mixed_edits'], 1)
        self.assertEqual(result['terms_matched'], 0)
        self.assertEqual(result['term_recall'], 0)

    def test_chinese_and_english_have_comparable_token_units(self):
        self.assertEqual(mixed_tokens('我们用 Redis 缓存'), ['我', '们', '用', 'redis', '缓', '存'])
        result = score_text('我们用Redis缓存', '我们用MySQL缓存')
        self.assertEqual(result['mixed_edits'], 1)
        self.assertEqual(result['mixed_tokens'], 6)

    def test_numbers_decimals_and_negation_are_not_silently_rewritten(self):
        self.assertNotEqual(mixed_tokens('不能超过1.5秒'), mixed_tokens('能超过15秒'))
        self.assertGreater(score_text('不能超过1.5秒', '能超过15秒')['mixed_edits'], 0)
        self.assertGreater(score_text('三次', '3次')['mixed_edits'], 0)

    def test_technical_tokens_and_normal_spelling_equivalences(self):
        self.assertEqual(mixed_tokens('C++、C#、.NET、Vue.js、GPT-4o、99.9%'),
                         ['c++', 'c#', '.net', 'vue.js', 'gpt-4o', '99.9%'])
        self.assertEqual(score_text('Ｒｅｄｉｓ，缓存。', 'Redis 缓存')['mixed_edits'], 0)
        self.assertTrue(contains_term('c + +与. net', 'C++'))
        self.assertTrue(contains_term('c + +与. net', '.NET'))

    def test_terms_do_not_match_inside_other_technical_names(self):
        for text, term in (('MySQL', 'SQL'), ('NoSQL', 'SQL'), ('C++17', 'C++'),
                           ('C#Script', 'C#'), ('file.NET', '.NET'), ('GraphRAG', 'RAG')):
            with self.subTest(text=text, term=term):
                self.assertFalse(contains_term(text, term))
        self.assertTrue(contains_term('使用SQL查询', 'SQL'))

    def test_unspoken_hotword_cannot_count_as_expected_term(self):
        with self.assertRaises(ValueError):
            score_text('这是缓存。', '这是Redis缓存。', ['Redis'])
        self.assertIsNone(score_text('这是缓存。', '这是缓存。')['term_recall'])
        self.assertEqual(score_text('用Redis缓存', '用redis缓存', ['Redis', 'redis', 'r e d i s'])['terms_total'], 1)
        for invalid in ('Redis', [None], [{'term': 'Redis'}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                score_text('用Redis缓存', '用Redis缓存', invalid)

    def test_disfluencies_and_empty_output_are_scored(self):
        result = score_text('嗯我我用缓存', '')
        self.assertEqual(result['cer'], 1)
        self.assertEqual(result['mer'], 1)
        self.assertGreater(score_text('', '凭空出现')['mer'], 0)


if __name__ == '__main__':
    unittest.main()
