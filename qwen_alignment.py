"""Recover original text around punctuation-free forced-alignment tokens."""
import math
import unicodedata


def aligned_words(text, alignment, duration):
    """Return timestamped words without deleting code symbols or punctuation.

    Qwen's aligner drops punctuation from tokens (C++ -> C, .NET -> NET).
    Use those tokens only to locate boundaries in the unchanged ASR text.
    A mismatch is an error, never a reason to invent a partial transcript.
    """
    if not isinstance(text, str) or not math.isfinite(duration) or duration < 0:
        raise ValueError('Invalid alignment input')
    positions = [i for i, char in enumerate(text) if char == "'" or
                 unicodedata.category(char).startswith(('L', 'N'))]
    if ''.join(item['text'] for item in alignment) != ''.join(text[i] for i in positions):
        raise ValueError('Alignment tokens do not match the transcript')
    if not positions:
        return []
    starts, offset = [], 0
    for item in alignment:
        if not item['text']:
            raise ValueError('Empty alignment token')
        start = positions[offset]
        if start and text[start-1] == '.' and text[start].isascii() and text[start].isalnum():
            start -= 1
        starts.append(start)
        offset += len(item['text'])
    starts[0] = 0
    result = []
    previous_end = 0.
    for index, (start, item) in enumerate(zip(starts, alignment)):
        begin, end = float(item['start_time']), float(item['end_time'])
        if not (math.isfinite(begin) and math.isfinite(end) and 0 <= begin <= end):
            raise ValueError('Invalid word timestamp')
        if begin < previous_end - .001 or end > duration + .2:
            raise ValueError('Word timestamp lies outside the audio timeline')
        previous_end = end
        result.append(dict(word=text[start:starts[index+1] if index+1 < len(starts) else len(text)],
                           start=min(duration, begin), end=min(duration, end)))
    if ''.join(item['word'] for item in result) != text:
        raise ValueError('Alignment lost transcript characters')
    return result
