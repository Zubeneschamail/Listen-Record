"""Offline recognition comparison using synthetic fixtures or a labeled manifest.

No playback, microphone capture, or audio upload. A manifest is a JSON list of
{audio, reference, hotwords?}; relative audio paths resolve beside the manifest.
Synthetic results are a regression aid, not an estimate of real-world accuracy.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from faster_whisper.audio import decode_audio
from app import clean_caption, mono_16k
from model_runtime import load_model
from recognition import RecognitionContext, INTERVIEW_HOTWORDS
from segmentation import AudioChunk, WordAssembler, decode_chunk, RATE
from tools.asr_metrics import score_text


def summarize(results):
    errors = sum(row['edits'] for row in results)
    characters = sum(row['characters'] for row in results)
    mixed_errors = sum(row['mixed_edits'] for row in results)
    tokens = sum(row['mixed_tokens'] for row in results)
    matched = sum(row['terms_matched'] for row in results)
    terms = sum(row['terms_total'] for row in results)
    return dict(cases=len(results), edits=errors, characters=characters, cer=errors/max(1, characters),
                mixed_edits=mixed_errors, mixed_tokens=tokens, mer=mixed_errors/max(1, tokens),
                terms_matched=matched, terms_total=terms, term_recall=matched/terms if terms else None,
                realtime_factor=sum(row['elapsed_seconds'] for row in results)/max(
                    .1, sum(row['audio_seconds'] for row in results)))


def synthesize(text, wav):
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"
    script = "$v=New-Object -ComObject SAPI.SpVoice; "
    script += "$vs=@($v.GetVoices() | Where-Object { $_.GetAttribute('Language') -match '^804' }); "
    script += "if(!$vs.Count){throw 'Chinese SAPI voice not installed'}; $v.Voice=$vs[0]; "
    script += "$s=New-Object -ComObject SAPI.SpFileStream; "
    script += "$s.Open(" + quote(wav) + ",3); $v.AudioOutputStream=$s; "
    script += "[void]$v.Speak(" + quote(text) + "); $s.Close()"
    subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['small'])
    parser.add_argument('--beams', nargs='+', type=int, default=[3, 5])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--languages', nargs='+', choices=['zh', 'en', 'auto'],
                        help='Compare explicit decoding languages; default uses each manifest language')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--without-hotwords', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if any(beam < 1 for beam in args.beams):
        parser.error('--beams must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        cases = json.loads(args.manifest.read_text(encoding='utf-8'))
        if not isinstance(cases, list) or not cases:
            parser.error('manifest must contain at least one labeled audio sample')
        for case in cases:
            case['audio'] = str(args.manifest.resolve().parent / case['audio'])
            score_text(case['reference'], case['reference'], case.get('critical_terms', []))
    else:
        fixtures = [
            ('general', '今天天气比较冷，出门记得带上外套。', ''),
            ('ai', '大语言模型通过检索增强生成减少幻觉，向量数据库保存文本的嵌入向量。',
             INTERVIEW_HOTWORDS['AI / 大模型']),
            ('systems', '我们采用微服务架构，通过消息队列异步处理任务，并使用幂等性避免重复写入。',
             INTERVIEW_HOTWORDS['后端开发']),
            ('mixed', 'LangChain 和 LangGraph 有什么区别？MCP 用来连接工具。',
             INTERVIEW_HOTWORDS['AI / 大模型']),
            ('names', 'PostgreSQL 负责存储数据，FastAPI 提供接口，Redis 用来缓存。',
             INTERVIEW_HOTWORDS['后端开发']),
            ('short', '对，可以。', ''),
        ]
        cases = []
        for name, text, terms in fixtures:
            wav = args.output / (name + '.wav')
            synthesize(text, wav)
            cases.append(dict(name=name, audio=str(wav), reference=text, hotwords=terms))
    if not isinstance(cases, list) or not cases:
        parser.error('manifest must contain at least one labeled audio sample')
    sounds = [decode_audio(case['audio'], sampling_rate=RATE) for case in cases]
    report = dict(source='labeled audio' if args.manifest else 'Chinese SAPI synthetic fixtures',
                  limitation=('A small labeled set does not establish overall interview accuracy.' if args.manifest else
                              'Synthetic speech does not measure real-world accents or microphone noise.'),
                  hotwords_enabled=not args.without_hotwords, results=[], summaries=[],
                  metrics=dict(cer='Legacy case-folded alphanumeric character error rate',
                               mer='Han characters + English words/numbers/technical symbols; NFKC and case-folded',
                               term_recall='Explicit source spellings only; not semantic fact verification'))
    for name in args.models:
        model = load_model(name, device=args.device)
        for policy in args.languages or ['manifest']:
            for beam in args.beams:
                results = []
                for case, sound in zip(cases, sounds):
                    guidance = RecognitionContext('' if args.without_hotwords else case.get('hotwords', ''))
                    options = guidance.options
                    guidance.options = lambda start, backlog=0, draft=False: dict(
                        options(start, backlog, draft), beam_size=beam)
                    started = time.monotonic()
                    language = case.get('language', 'zh') if policy == 'manifest' else None if policy == 'auto' else policy
                    rows = decode_chunk(model, WordAssembler(), AudioChunk(sound, 0, 0), language, guidance)
                    elapsed = time.monotonic() - started
                    text = clean_caption(''.join(row[2] for row in rows))
                    result = dict(model=name, backend=model.model.device, beam=beam,
                                  name=case.get('name', Path(case['audio']).stem),
                                  group=case.get('group'), topic=case.get('topic'), speaker=case.get('speaker'),
                                  language=language, language_policy=policy,
                                  hotwords=guidance.options(0)['hotwords'],
                                  reference=case['reference'], text=text,
                                  audio_seconds=len(sound)/RATE, elapsed_seconds=elapsed,
                                  **score_text(case['reference'], text, case.get('critical_terms', [])))
                    results.append(result)
                    report['results'].append(result)
                    print(json.dumps(result, ensure_ascii=True), flush=True)
                summary = dict(model=name, beam=beam, language_policy=policy, **summarize(results),
                               groups={group: summarize([row for row in results if row['group'] == group])
                                       for group in sorted({row['group'] for row in results if row['group'] is not None})})
                report['summaries'].append(summary)
                print(json.dumps(summary), flush=True)
                (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        del model
        gc.collect()
    # Capture regression: one inverted stereo channel used to erase all speech.
    signal = sounds[0]
    stereo = np.column_stack([signal, -signal])
    fixed = mono_16k(stereo, RATE)
    report['stereo_check'] = dict(before_rms=float(np.sqrt(np.mean(stereo.mean(axis=1)**2))),
                                after_rms=float(np.sqrt(np.mean(fixed**2))),
                                source_rms=float(np.sqrt(np.mean(signal**2))))
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
