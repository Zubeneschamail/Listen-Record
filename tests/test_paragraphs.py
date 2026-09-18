import unittest
from paragraphs import ParagraphAssembler


def row(start, end, text, source='system'):
    return dict(start=start, end=end, text=text, source=source, captured_at='original-time')


class ParagraphTests(unittest.TestCase):
    def test_continuous_chunks_update_one_bubble_then_silence_commits(self):
        p = ParagraphAssembler()
        self.assertEqual(p.consume(row(0, 2, '我们讨论')), [])
        self.assertEqual(p.preview('system', '一个问题'), '我们讨论一个问题')
        self.assertEqual(p.consume(row(2.4, 4, '一个问题。')), [])
        self.assertEqual(p.silence('system', 4.7, 4), [])
        result = p.silence('system', 4.81, 4)
        self.assertEqual(result[0]['text'], '我们讨论一个问题。')
        self.assertEqual((result[0]['start'], result[0]['end']), (0, 4))
        self.assertEqual(p.finish(), [])

    def test_active_speech_prevents_commit_while_decoding_lags(self):
        p = ParagraphAssembler()
        p.consume(row(0, 3, '前半句'))
        self.assertEqual(p.silence('system', 10, 9.5), [])
        self.assertEqual(p.preview('microphone', '我说'), '我说')

    def test_source_change_and_pause_split_paragraphs(self):
        p = ParagraphAssembler()
        p.consume(row(0, 2, '对方'))
        self.assertEqual(p.consume(row(2.2, 3, '我', 'microphone'))[0]['text'], '对方')
        self.assertEqual(p.consume(row(4.5, 6, '继续', 'microphone'))[0]['text'], '我')
        self.assertEqual(p.finish()[0]['text'], '继续')

    def test_long_sentence_and_stop_flush(self):
        p = ParagraphAssembler()
        p.consume(row(0, 14, '长' * 250))
        self.assertIsNotNone(p.pending)
        self.assertEqual(len(p.consume(row(14, 20, '句末。'))), 1)
        p.consume(row(20, 21, 'Hello'))
        p.consume(row(21, 22, 'world.'))
        self.assertEqual(p.finish()[0]['text'], 'Hello world.')
