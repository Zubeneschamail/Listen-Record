import unittest
from floating_caption import CaptionPages

class CaptionPagesTests(unittest.TestCase):
    def test_append_does_not_move_completed_line(self):
        p = CaptionPages(len, 5)
        self.assertEqual(p.update('abcdefgh', True), ['abcde', 'fgh'])
        self.assertEqual(p.update('abcdefghij', True), ['abcde', 'fghij'])
        self.assertEqual(p.update('abcdefghijk', True), ['k'])
        self.assertEqual(p.update('abcdefghijkl', True), ['kl'])

    def test_correction_does_not_replay_previous_page(self):
        p = CaptionPages(len, 5)
        p.update('abcdefghijkl', True)
        self.assertEqual(p.update('abXcdefghijkl', False), ['kl'])
        self.assertEqual(p.update('abXcdefghijkl', False), ['kl'])
        self.assertEqual(p.update('new words', True), ['new w', 'ords'])

    def test_repeated_refresh_and_clear(self):
        p = CaptionPages(len, 5)
        p.update('abcdefghijkl', True)
        self.assertEqual(p.update('abcdefghijkl', True), ['kl'])
        self.assertEqual(p.update('', False), [])
        self.assertEqual(p.update('hello', True), ['hello'])
