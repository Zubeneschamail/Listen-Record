"""Fetch pinned ASCEND test recordings for local evaluation, never training.

Select samples before running ASR. Preserve source annotations and SHA256s.
The public viewer assets must resolve to the declared dataset revision.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import httpx

DATASET = 'CAiRE/ASCEND'
REVISION = '737e9800ae31be9932ba8464c80366559bd28424'
API = 'https://datasets-server.huggingface.co/rows'
SOURCE = f'https://huggingface.co/datasets/{DATASET}'


def fetch_page(client, offset, split='test'):
    response = client.get(API, params=dict(dataset=DATASET, config='main', split=split,
                                          offset=offset, length=100))
    response.raise_for_status()
    page = response.json()
    if page.get('partial') or any(item.get('truncated_cells') for item in page['rows']):
        raise ValueError('Dataset annotations are incomplete')
    for item in page['rows']:
        sources = item['row']['audio']
        if len(sources) != 1 or f'/--/{REVISION}/--/' not in sources[0]['src']:
            raise ValueError('Public viewer revision differs from the pinned reference')
    return page


def select(rows, language, count):
    # Round-robin speakers, with their utterances evenly spaced across the
    # available source order. This avoids a single speaker/topic dominating.
    speakers = {}
    for item in rows:
        row = item['row']
        if (row['language'] == language and 2 <= row['duration'] <= 12 and
                '[UNK]' not in row['transcription'].upper()):
            speakers.setdefault(row['original_speaker_id'], []).append(item)
    if count < 0:
        raise ValueError('Sample count must not be negative')
    if sum(map(len, speakers.values())) < count:
        raise ValueError('Not enough eligible source samples')
    # Redistribute unused slots when a speaker has fewer eligible utterances.
    # A fixed ceiling per speaker can report failure despite enough total rows.
    quotas = dict.fromkeys(sorted(speakers), 0)
    remaining = count
    while remaining:
        for speaker in quotas:
            if remaining and quotas[speaker] < len(speakers[speaker]):
                quotas[speaker] += 1
                remaining -= 1
    candidates = []
    for speaker, items in sorted(speakers.items()):
        per_speaker = quotas[speaker]
        if not per_speaker:
            continue
        chosen = ([items[0]] if per_speaker == 1 else
                  [items[i*(len(items)-1)//(per_speaker-1)] for i in range(per_speaker)])
        candidates.append(chosen)
    result = []
    while len(result) < count and any(candidates):
        for items in candidates:
            if items and len(result) < count:
                result.append(items.pop(0))
    if len(result) != count:
        raise ValueError('Not enough eligible source samples')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        card_url = f'{SOURCE}/raw/{REVISION}/README.md'
        card = client.get(card_url).raise_for_status().content
        if b'cc-by-sa-4.0' not in card:
            raise ValueError('Expected source license declaration not found')
        (args.output/'SOURCE-README.md').write_bytes(card)
        first = fetch_page(client, 0, args.split)
        pages = [first]
        with ThreadPoolExecutor(max_workers=4) as pool:
            pages.extend(pool.map(lambda offset: fetch_page(client, offset, args.split),
                                  range(100, first['num_rows_total'], 100)))
        rows = [row for page in pages for row in page['rows']]
        if len(rows) != first['num_rows_total']:
            raise ValueError('Incomplete source index')
        selected = (select(rows, 'mixed', 24) + select(rows, 'zh', 8) + select(rows, 'en', 8))
        # Persist the selected index before downloading or invoking inference.
        annotations = [dict(row_index=item['row_idx'], **{
            key: value for key, value in item['row'].items() if key not in ('audio', 'path')})
            for item in selected]
        (args.output/'selected-annotations.json').write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2), encoding='utf-8')
        cases, files = [], []
        for item in selected:
            row = item['row']
            if not re.fullmatch(r'\d+', row['id']):
                raise ValueError('Invalid sample identifier')
            filename = f"ascend-{args.split}-{row['id']}.wav"
            url = row['audio'][0]['src']
            with client.stream('GET', url) as response:
                response.raise_for_status()
                audio = bytearray()
                for chunk in response.iter_bytes():
                    audio.extend(chunk)
                    if len(audio) > 4*1024*1024:
                        raise ValueError('Sample exceeds bounded download size')
            if audio[:4] != b'RIFF':
                raise ValueError('Public viewer did not return WAV audio')
            (args.output/filename).write_bytes(audio)
            cases.append(dict(name=filename[:-4], audio=filename, reference=row['transcription'],
                              language='zh', group=row['language'], hotwords='',
                              speaker=row['original_speaker_id'], topic=row['topic']))
            files.append(dict(name=filename, row_index=item['row_idx'],
                              source_url=url.partition('?')[0],
                              sha256=hashlib.sha256(audio).hexdigest(), bytes=len(audio)))
            print(json.dumps(dict(sample=filename, group=row['language'], bytes=len(audio))), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    provenance = dict(source=SOURCE, revision=REVISION, split=args.split, license='CC-BY-SA-4.0',
                      purpose='Local evaluation only; no training or application bundling',
                      selection='Before ASR: 24 mixed + 8 Chinese + 8 English, 2–12 seconds; no [UNK] annotations; round-robin speakers',
                      source_rows_with_unknown_annotations=sum('[UNK]' in item['row']['transcription'].upper() for item in rows),
                      limitation='Spontaneous general conversation, not a technical interview population estimate',
                      languages=dict(Counter(item['group'] for item in cases)),
                      speakers=dict(Counter(item['speaker'] for item in cases)),
                      topics=dict(Counter(item['topic'] for item in cases)), files=files)
    (args.output/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(samples=len(cases), groups=provenance['languages'], topics=provenance['topics'])), flush=True)


if __name__ == '__main__':
    main()
