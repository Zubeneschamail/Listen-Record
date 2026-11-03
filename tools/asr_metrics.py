"""Transparent ASR metrics; spelling checks do not judge semantic correctness."""
import re
import unicodedata


def normalized_characters(text):
    # Keep the original benchmark's CER definition for historical comparisons.
    return ''.join(char.casefold() for char in text if char.isalnum())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for index, expected in enumerate(reference, 1):
        current = [index]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[column]+1,
                               previous[column-1]+(expected != actual)))
        previous = current
    return previous[-1]


_MIXED_TOKEN = re.compile(
    r"\.?[a-z][a-z0-9]*(?:(?:[.'/_-][a-z0-9]+)|[+#]+)*"
    r'|\d+(?:\.\d+)?%?|[\u3400-\u9fff]|[+#]|[^\W_]', re.ASCII)


def mixed_tokens(text):
    """One token per Han character, English word, number or technical symbol.

    Case/fullwidth forms are normalized. C++/C#/.NET remain distinguishable;
    Chinese and English sentence punctuation is ignored. No filler deletion,
    synonym rewriting or conversion between Chinese and Arabic numbers.
    """
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'")
    return _MIXED_TOKEN.findall(text)


def contains_term(text, term):
    """Literal explicit spelling match with optional spaces, never fuzzy repair."""
    text = unicodedata.normalize('NFKC', text)
    compact = ''.join(unicodedata.normalize('NFKC', term).split())
    if not compact:
        return False
    pattern = r'\s*'.join(re.escape(char) for char in compact)
    if compact[0].isascii() and (compact[0].isalnum() or compact[0] in '.+#'):
        pattern = r'(?<![A-Za-z0-9.+#/_-])'+pattern
    if compact[-1].isascii() and (compact[-1].isalnum() or compact[-1] in '.+#'):
        pattern += r'(?![A-Za-z0-9.+#/_-])'
    return re.search(pattern, text, re.I | re.ASCII) is not None


def score_text(reference, hypothesis, critical_terms=()):
    ref_chars, hyp_chars = normalized_characters(reference), normalized_characters(hypothesis)
    ref_tokens, hyp_tokens = mixed_tokens(reference), mixed_tokens(hypothesis)
    edits, token_edits = edit_distance(ref_chars, hyp_chars), edit_distance(ref_tokens, hyp_tokens)
    if not isinstance(critical_terms, (tuple, list)):
        raise ValueError('critical_terms must be a list of explicit source spellings')
    terms, seen = [], set()
    for term in critical_terms:
        if not isinstance(term, str) or not term.strip() or not contains_term(reference, term):
            raise ValueError('Critical term is absent from the reference: '+str(term))
        key = ''.join(unicodedata.normalize('NFKC', term).split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(dict(term=term, matched=contains_term(hypothesis, term)))
    matched = sum(item['matched'] for item in terms)
    return dict(edits=edits, characters=len(ref_chars), cer=edits/max(1, len(ref_chars)),
                mixed_edits=token_edits, mixed_tokens=len(ref_tokens),
                mer=token_edits/max(1, len(ref_tokens)), critical_terms=terms,
                terms_matched=matched, terms_total=len(terms),
                term_recall=matched/len(terms) if terms else None)
