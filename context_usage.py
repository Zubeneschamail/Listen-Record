"""Per-request context occupancy, never cumulative account token consumption."""
import json
import math
from urllib.parse import urlsplit


def context_limit(provider, profile):
    # Official limits: https://api-docs.deepseek.com/quick_start/pricing
    # Third-party gateways may impose different limits even for the same model.
    if (provider == 'deepseek' and urlsplit(profile.get('base_url', '')).hostname == 'api.deepseek.com'
            and profile.get('model', '').strip() in ('deepseek-flash', 'deepseek-v4-pro')):
        return 1_000_000
    return None


def estimate_tokens(value):
    # Deliberately approximate; do not count base64 image bytes as text tokens.
    if isinstance(value, dict):
        value = {key: (None if key == 'image_url' else item) for key, item in value.items()}
        return sum(estimate_tokens(k) + estimate_tokens(v) for k, v in value.items()) + 4
    if isinstance(value, list):
        return sum(estimate_tokens(item) for item in value) + 2
    text = value if isinstance(value, str) else json.dumps(value)
    return math.ceil(sum(.25 if ord(char) < 128 else 1 for char in text))


def usage_report(provider, profile, messages, tools=None, usage=None, answer='', image=False):
    used = None
    if isinstance(usage, dict):
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if all(type(n) is int and n >= 0 for n in (prompt, completion)):
            used = prompt + completion
    estimated = used is None
    if estimated:
        used = estimate_tokens(messages) + estimate_tokens(answer)
        if tools:
            used += estimate_tokens(tools)
    return dict(model=profile.get('model', ''), used=used, limit=context_limit(provider, profile),
                estimated=estimated, image_unestimated=estimated and image)


def percentage(report):
    if not report or not report.get('limit'):
        return None
    return report['used'] / report['limit'] * 100


def usage_label(report):
    value = percentage(report)
    if value is None:
        return '—'
    number = '<0.1' if 0 < value < .1 else f'{value:.1f}' if value < 10 else f'{value:.0f}'
    return ('≈' if report.get('estimated') else '') + number + '%'
