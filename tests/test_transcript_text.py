import unittest

from transcript_text import transcript_key


class TranscriptTextTests(unittest.TestCase):
    def test_case_width_and_sentence_punctuation_are_cosmetic(self):
        for text in ('什么是 C++？', '什么是c++?', '什么是Ｃ＋＋。', '什么是 c++！'):
            self.assertEqual(transcript_key(text), '什么是c++')
        self.assertEqual(transcript_key('Node.js.'), transcript_key('ＮＯＤＥ．ＪＳ！'))
        self.assertEqual(transcript_key('为什么？'), transcript_key('为 什么。'))
        self.assertEqual(transcript_key('. N E T'), transcript_key('.NET'))
        self.assertEqual(transcript_key('a ! = b'), transcript_key('a!=b'))

    def test_technical_names_and_operators_stay_distinct(self):
        groups = (
            ('C', 'C+', 'C++', 'C#'), ('.NET', 'NET'), ('Node.js', 'Nodejs'),
            ('GPT-4o', 'GPT4o'), ('I/O', 'IO'), ('a_b', 'ab'),
            ('a>=b', 'a<=b', 'a=b', 'a==b', 'a!=b'),
            ('a[0]', 'a(0)', 'a0'), ('10%', '10'), ('x*y', 'xy'),
            ('n²', 'n2'), ('n³', 'n3'), ('x₁', 'x1'),
        )
        for group in groups:
            with self.subTest(group=group):
                keys = [transcript_key(text) for text in group]
                self.assertEqual(len(set(keys)), len(keys))

    def test_blank_or_only_sentence_punctuation_has_no_key(self):
        self.assertEqual(transcript_key(' ，。！？;: \n'), '')
