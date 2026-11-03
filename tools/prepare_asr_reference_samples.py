"""Download a small pinned reference-speech set for local ASR evaluation only.

Uses authentic prompt recordings and their matching prompt_text (not the TTS
target text) from CV3-Eval. Does not train a model, capture audio or upload data.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import httpx

REVISION = '68b7682a410d048308d1a12f2fbc266770cd23d5'
SOURCE = 'https://github.com/QwenAudio/CV3-Eval'
BASE = f'https://raw.githubusercontent.com/QwenAudio/CV3-Eval/{REVISION}/'


def fetch(client, name, limit=8*1024*1024):
    with client.stream('GET', BASE+name) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError('Reference file exceeds download limit: '+name)
        return bytes(data)


def select_evenly(rows, count):
    count = min(len(rows), count)
    if count == 1:
        return rows[:1]
    return [rows[index*(len(rows)-1)//(count-1)] for index in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, provenance = [], []
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        license_data = fetch(client, 'LICENSE')
        (args.output/'SOURCE-LICENSE.txt').write_bytes(license_data)
        for group, directory, count in [('general', 'data/zero_shot/zh', 12),
                                        ('expressive', 'data/emotion_zeroshot/zh', 8)]:
            text_data = fetch(client, directory+'/prompt_text')
            wav_data = fetch(client, directory+'/prompt_wav.scp')
            (args.output/(group+'-prompt_text.txt')).write_bytes(text_data)
            (args.output/(group+'-prompt_wav.scp')).write_bytes(wav_data)
            texts = [line.split(maxsplit=1) for line in text_data.decode('utf-8').splitlines() if line.strip()]
            paths = dict(line.split(maxsplit=1) for line in wav_data.decode('utf-8').splitlines() if line.strip())
            for key, reference in select_evenly(texts, count):
                remote = PurePosixPath(paths[key])
                if remote.is_absolute() or '..' in remote.parts or remote.suffix != '.wav':
                    raise ValueError('Invalid reference audio path')
                audio = fetch(client, remote.as_posix())
                if audio[:4] != b'RIFF':
                    raise ValueError('Reference audio is not a WAV file: '+str(remote))
                filename = group+'-'+key+'.wav'
                (args.output/filename).write_bytes(audio)
                cases.append(dict(name=group+'-'+key, audio=filename, reference=reference,
                                  group=group, language='zh', hotwords=''))
                provenance.append(dict(name=cases[-1]['name'], url=BASE+remote.as_posix(),
                                       transcript_url=BASE+directory+'/prompt_text',
                                       sha256=hashlib.sha256(audio).hexdigest(), bytes=len(audio)))
                print(json.dumps({'sample': cases[-1]['name'], 'bytes': len(audio)}), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output/'provenance.json').write_text(json.dumps(dict(
        source=SOURCE, revision=REVISION, selection='Evenly spaced rows; selected before inference',
        purpose='Local ASR evaluation using source reference prompt recordings and matched transcripts',
        limitation='20 general/expressive Chinese utterances, not a representative technical interview corpus',
        license_sha256=hashlib.sha256(license_data).hexdigest(), files=provenance),
        ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
