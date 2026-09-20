"""Bounded, local extractive memory; full originals remain in the session."""
import json
import re


CONTEXT_BYTES = 36000


def clip(text, budget, tail=False):
    raw = text.encode('utf-8')
    if len(raw) <= budget:
        return text
    return (raw[-budget:] if tail else raw[:budget]).decode('utf-8', errors='ignore')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def recent(records, budget):
    selected, size = [], 0
    for record in reversed(records):
        cost = len(record.encode('utf-8')) + 1
        if size + cost > budget:
            if not selected:
                # A single oversized turn must not hide the newest context.
                return records, clip(record, budget//2) + '\n…\n' + clip(record, budget//2-10, tail=True)
            break
        selected.append(record)
        size += cost
    selected.reverse()
    return records[:len(records)-len(selected)], '\n'.join(selected)


def terms(text):
    words = set(re.findall(r'[a-zA-Z0-9_]{2,}', text.lower()))
    for phrase in re.findall(r'[\u4e00-\u9fff]+', text):
        words.update(phrase[i:i+2] for i in range(len(phrase)-1))
    return words


def build_context(rows, exchanges, question, budget=CONTEXT_BYTES):
    transcripts = [f'[转写 {i+1} / {source}] {text}' for i, (source, text) in enumerate(rows)]
    answers = [f'[问答 {i+1}] 用户问题：{q}\nAI 回复（非已确认事实）：{a}'
               for i, (q, a) in enumerate(exchanges)]
    full = {'模式': '完整会话', '全部转写': transcripts, '历史问答': answers}
    if len(encoded(full).encode('utf-8')) <= budget:
        return full
    old_text, recent_text = recent(transcripts, 12000)
    old_answers, recent_answers = recent(answers, 10000)
    older = old_text + old_answers
    # Deterministic excerpts avoid inventing facts or making another API call.
    # Spread the overview across the whole session, then retrieve relevant originals.
    overview = []
    for group in (old_text, old_answers):
        if group:
            indices = sorted({int(i * (len(group)-1)/19) for i in range(20)})
            for i in indices:
                record = group[i]
                overview.append(clip(record, 240) + (' … ' + clip(record, 120, tail=True)
                                                  if len(record.encode('utf-8')) > 240 else ''))
    query = terms(question)
    ranked = sorted(enumerate(older), key=lambda pair: (len(query & terms(pair[1])), pair[0]), reverse=True)
    relevant = []
    for index, record in ranked:
        if not query.intersection(terms(record)):
            break
        # Choose matching passages even when the source record is very large.
        pieces = re.split(r'(?<=[。！？.!?\n])', record)
        passages = sorted(enumerate(pieces), key=lambda p: len(query & terms(p[1])), reverse=True)[:4]
        relevant.append(clip(record.split(']', 1)[0]+']', 160) + ' ' +
                        clip(''.join(piece for _, piece in sorted(passages)), 1300))
        if len(relevant) == 5:
            break
    result = {'模式': '压缩会话', '说明': '早期内容为本地抽取摘要，存在省略；历史AI回复可能有误，不是用户确认的事实。',
              '早期摘要': clip('\n'.join(overview), 5000),
              '相关早期原文摘录': clip('\n'.join(relevant), 6000),
              '近期转写原文': recent_text, '近期问答原文': recent_answers}
    # Bound serialized bytes too, including escaped characters and JSON overhead.
    while len(encoded(result).encode('utf-8')) > budget:
        key = max(('早期摘要', '相关早期原文摘录', '近期转写原文', '近期问答原文'),
                  key=lambda k: len(result[k].encode('utf-8')))
        result[key] = clip(result[key], max(0, len(result[key].encode('utf-8'))-1000))
    return result


def build_conversation(rows, exchanges, question):
    """Keep actual user/assistant roles; summarize only older overflow turns."""
    pairs = [[{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
             for q, a in exchanges]
    background = {'模式': '完整会话', '全部转写':
                  [f'[转写 {i+1} / {source}] {text}' for i, (source, text) in enumerate(rows)]}
    messages = [message for pair in pairs for message in pair]
    if len(encoded([background, messages]).encode('utf-8')) <= CONTEXT_BYTES:
        return background, messages

    selected, size = [], 0
    for pair in reversed(pairs):
        cost = len(encoded(pair).encode('utf-8')) + 1
        if size + cost > 18000:
            if not selected:
                pair = [dict(message) for message in pair]
                while len(encoded(pair).encode('utf-8')) > 17000:
                    message = max(pair, key=lambda m: len(m['content'].encode('utf-8')))
                    text = message['content']
                    limit = max(20, len(text.encode('utf-8')) * 3 // 8)
                    message['content'] = clip(text, limit) + '\n[部分省略]\n' + clip(text, limit, tail=True)
                selected.append(pair)
            break
        selected.append(pair)
        size += cost
    selected.reverse()
    messages = [message for pair in selected for message in pair]
    older = exchanges[:len(exchanges)-len(selected)]
    background = build_context(rows, older, question,
        budget=CONTEXT_BYTES-len(encoded(messages).encode('utf-8'))-256)
    background['模式'] = '早期资料与转写摘要；近期问答见对话消息'
    return background, messages
