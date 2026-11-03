"""Comparison keys for speech text; keep symbols that change technical meaning."""
import unicodedata


def transcript_key(text):
    """Ignore cosmetic differences while distinguishing names and operators.

    These keys are only for duplicate detection, not for changing displayed text.
    A period before an ASCII letter/digit belongs to .NET, Node.js or a version;
    a trailing sentence period does not. Keep brackets and operator punctuation
    too: a[0], a(0), a_0 and a0 can be different interview examples.
    """
    # Normalize fullwidth ASCII only. NFKC would also turn n² into n2, which
    # changes the meaning of math/code questions. Ignore spacing before testing
    # punctuation so . NET and a ! = b keep the same symbols as .NET and a!=b.
    text = ''.join(chr(ord(char)-0xfee0) if '\uff01' <= char <= '\uff5e' else char
                   for char in text if not char.isspace()).casefold()
    result = []
    for index, char in enumerate(text):
        if (char.isalnum() or unicodedata.category(char).startswith('S') or
                char in '#/_-=%&|^~@$\\()[]{}*'):
            result.append(char)
        elif (char == '.' and index+1 < len(text) and
              text[index+1].isascii() and text[index+1].isalnum()):
            result.append(char)
        elif char == '!' and text[index+1:index+2] == '=':
            result.append(char)
    return ''.join(result)
