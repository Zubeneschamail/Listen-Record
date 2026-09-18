"""Join recognition chunks into readable speaker paragraphs."""
import re


def join_text(left, right):
    if not left or not right:
        return left or right
    separator = ' ' if left[-1].isascii() and right[0].isascii() and (left[-1].isalnum() or left[-1] in '.!?;,:') and right[0].isalnum() else ''
    return left + separator + right


class ParagraphAssembler:
    def __init__(self, pause=0.8):
        self.pause = pause
        self.pending = None

    def finish(self):
        row, self.pending = self.pending, None
        return [row] if row else []

    def consume(self, row):
        completed = []
        if self.pending and (row['source'] != self.pending['source'] or
                             row['start']-self.pending['end'] >= self.pause):
            completed.extend(self.finish())
        if self.pending is None:
            self.pending = dict(row)
        else:
            self.pending['text'] = join_text(self.pending['text'], row['text'])
            self.pending['end'] = max(self.pending['end'], row['end'])
        text = self.pending['text']
        duration = self.pending['end']-self.pending['start']
        if ((len(text) >= 240 or duration >= 45) and re.search(r'[。！？.!?][”’"\']?$', text)) or len(text) >= 600 or duration >= 90:
            completed.extend(self.finish())
        return completed

    def silence(self, source, audio_end, last_speech_end):
        if (self.pending and self.pending['source'] == source and
                audio_end-max(self.pending['end'], last_speech_end or 0) >= self.pause):
            return self.finish()
        return []

    def preview(self, source, draft):
        prefix = self.pending['text'] if self.pending and self.pending['source'] == source else ''
        return join_text(prefix, draft)
