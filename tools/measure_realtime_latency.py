"""Measure the real Transcriber pipeline using paced synthetic capture callbacks.

No speaker playback, microphone capture, AI requests, or desktop settings changes.
The audio-device transport is simulated; segmentation and model inference are real.
"""
import argparse
import json
import logging
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='small')
    parser.add_argument('--language', choices=['zh', 'en', 'auto'], default='zh')
    parser.add_argument('--interview-flow', action='store_true',
                        help='Verify a paused long question with local automatic Q&A (no API)')
    parser.add_argument('--dual-source', action='store_true',
                        help='Also capture a synthetic candidate interruption on the microphone track')
    args = parser.parse_args()
    if args.dual_source and not args.interview_flow:
        parser.error('--dual-source requires --interview-flow')
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output / 'benchmark.log', level=logging.INFO)
    import numpy as np
    from faster_whisper.audio import decode_audio
    import app
    from recognition import read_preferences
    from app_paths import DATA

    cases = [
        ('short', '你好，这是闻录语音转文字的延时测试。'),
        ('medium', '我们正在测试实时语音识别。电脑收到声音后，需要先找到句子的边界，再把内容转换成文字。'),
        ('continuous', '今天我们来测试一段连续讲话，看看电脑能不能及时把声音转换成文字，'
         '在说话的过程中字幕应该不断更新，我们还要观察一句话说完之后需要等待多长时间才能看到完整结果，'
         '如果处理速度跟不上说话速度，后面的文字就会越来越迟，这就是音频积压带来的额外等待。'),
    ]
    question_parts = [
        '请介绍一下你在项目里面处理消息队列的经验。',
        '这道题还有几个具体条件需要一起考虑，订单提交以后要异步扣减库存，'
        '消息可能会重复投递，消费者也有可能在处理过程中突然退出，'
        '如果服务重启以后重新收到之前的消息，应该怎样避免重复扣减库存，'
        '如果数据库写入成功但是消息确认失败，又应该怎样保证数据最终保持一致，'
        '请结合实际方案说明你的判断过程，最后再讲一下失败之后的恢复方案。',
    ]
    if args.interview_flow:
        cases = [('interview_question', ''.join(question_parts))]
    sounds = []
    for name, reference in cases:
        if args.interview_flow:
            from benchmark_transcription_accuracy import synthesize
            parts = []
            for index, text in enumerate(question_parts):
                wav = args.output / f'question-part-{index}.wav'
                synthesize(text, wav)
                audio = decode_audio(str(wav), sampling_rate=48000)
                voiced = np.flatnonzero(np.abs(audio) > .005)
                parts.append(audio[max(0, voiced[0]-4800):voiced[-1]+4800])
            sounds.append(np.concatenate([parts[0], np.zeros(48000, np.float32), parts[1]]))
            continue
        wav = args.output / (name + '.wav')
        script = "$v=New-Object -ComObject SAPI.SpVoice; "
        script += "$vs=@($v.GetVoices() | Where-Object { $_.GetAttribute('Language') -match '^804' }); "
        script += "if($vs.Count -eq 0){throw 'Chinese voice missing'}; $v.Voice=$vs[0]; "
        script += "$s=New-Object -ComObject SAPI.SpFileStream; "
        script += "$s.Open('" + str(wav).replace("'", "''") + "',3); $v.AudioOutputStream=$s; "
        script += "[void]$v.Speak('" + reference.replace("'", "''") + "'); $s.Close()"
        subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        sounds.append(decode_audio(str(wav), sampling_rate=48000))

    started = time.monotonic()
    model = app.load_model(args.model)
    load_seconds = time.monotonic() - started
    print(json.dumps({'model': args.model, 'backend': model.model.device,
                      'model_load_and_warmup_seconds': load_seconds}), flush=True)
    language = None if args.language == 'auto' else args.language
    report = dict(model=args.model, backend=model.model.device, language=args.language,
                  capture_rate=48000, capture_channels=2,
                  transport='synthetic real-time callbacks; real inference and event emission',
                  model_load_and_warmup_seconds=load_seconds, cases=[])
    hotwords = read_preferences(DATA / 'recognition-settings.json')

    for (name, reference), sound in zip(cases, sounds):
        events = queue.Queue()
        engine = app.Transcriber(events)
        engine.model_name, engine.model = args.model, model
        speech_samples = np.flatnonzero(np.abs(sound) > 0.005)
        speech_start = float(speech_samples[0] / 48000)
        speech_end = float(speech_samples[-1] / 48000)
        stop_feed = threading.Event()
        stages, observations = [], []
        clock = {}
        device_sounds = {0: sound}
        if args.dual_source:
            from benchmark_transcription_accuracy import synthesize
            wav = args.output/'candidate.wav'
            synthesize('我在项目里面主要负责后端接口，您刚才问的是怎么避免重复扣减库存吗？', wav)
            candidate = decode_audio(str(wav), sampling_rate=48000)
            device_sounds[1] = np.concatenate([np.zeros(12*48000, np.float32), candidate])
        from qa_worker import QAWorker
        qa_events = queue.Queue()
        qa = QAWorker(qa_events, lambda prompt, cancel: '本地测试回答，不调用模型服务。')
        qa.set_enabled(args.interview_flow)
        qa.set_question_source('system')
        auto_questions = []

        class Stream:
            def __init__(self, callback, index):
                self.callback = callback
                self.sound = device_sounds[index]

            def start_stream(self):
                clock.setdefault('start', time.monotonic())

                def feed():
                    offset = 0
                    while not stop_feed.is_set() and not engine.stop_event.is_set():
                        deadline = clock['start'] + (offset + 4800) / 48000
                        if stop_feed.wait(max(0, deadline - time.monotonic())):
                            break
                        block = self.sound[offset:offset + 4800]
                        if len(block) < 4800:
                            block = np.pad(block, (0, 4800 - len(block)))
                        stereo = np.repeat(block[:, None], 2, axis=1)
                        self.callback(stereo.tobytes(), 4800, {}, 0)
                        offset += 4800

                self.thread = threading.Thread(target=feed, daemon=True)
                self.thread.start()

            def is_active(self):
                return True

            def close(self):
                stop_feed.set()
                self.thread.join(timeout=2)

        class Capture:
            def get_device_info_by_index(self, index):
                return dict(defaultSampleRate=48000, maxInputChannels=2, isLoopbackDevice=index == 0)

            def open(self, **kw):
                return Stream(kw['stream_callback'], kw['input_device_index'])

            def terminate(self):
                pass

        real_decode, real_preview = app.decode_chunk, app.DraftPreview.render

        def decode(*a, **kw):
            begin = time.monotonic()
            result = real_decode(*a, **kw)
            stages.append(dict(kind='final_decode', audio_seconds=len(a[2].audio) / 16000,
                               compute_seconds=time.monotonic() - begin,
                               begin_seconds=begin - clock['start'],
                               backlog_seconds=a[5] if len(a) > 5 else 0))
            return result

        def preview(*a, **kw):
            begin = time.monotonic()
            result = real_preview(*a, **kw)
            elapsed = time.monotonic() - begin
            if elapsed > 0.05:
                stages.append(dict(kind='preview_decode', compute_seconds=elapsed,
                                   begin_seconds=begin - clock['start']))
            return result

        with patch.object(app.pa, 'PyAudio', Capture), patch.object(app, 'decode_chunk', decode), \
                patch.object(app.DraftPreview, 'render', preview):
            devices = [{'index': 0, 'source': 'system'}, {'index': 1, 'source': 'microphone'}] if args.dual_source else 0
            engine.start(devices, args.model, language, hotwords)
            timeout = time.monotonic() + len(sound) / 48000 + 90
            try:
                while time.monotonic() < timeout:
                    try:
                        kind, value = events.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    elapsed = time.monotonic() - clock.get('start', time.monotonic())
                    if kind == 'error':
                        raise RuntimeError(value)
                    if kind == 'voice_activity':
                        qa.observe_voice(value['source'], value['active'])
                        observations.append(dict(kind=kind, seconds=elapsed, value=value))
                    elif kind == 'segment':
                        qa.feed(value['text'], value.get('source', 'system'))
                    if kind == 'preview' and isinstance(value, dict) and value['text']:
                        observations.append(dict(kind=kind, seconds=elapsed, text=value['text']))
                    elif kind in ('segment', 'backlog'):
                        observations.append(dict(kind=kind, seconds=elapsed, value=value))
                    elif kind == 'done':
                        break
                    qa.tick()
                    while not qa_events.empty():
                        _, (generation, phase, payload) = qa_events.get_nowait()
                        if phase == 'thinking':
                            auto_questions.append(dict(seconds=elapsed, question=payload))
                        elif phase == 'done':
                            qa.active = False
                    # Let natural silence commit the last paragraph before stopping.
                    finals = [o for o in observations if o['kind'] == 'segment' and o['value'].get('source', 'system') == 'system']
                    if (finals and finals[-1]['value']['end'] >= speech_end - 0.6 and elapsed > speech_end + 1
                            and (not args.interview_flow or auto_questions)):
                        engine.stop()
                else:
                    raise TimeoutError('No final transcript before deadline')
            finally:
                engine.stop()
                stop_feed.set()
                engine.thread.join(timeout=60)
                if engine.thread.is_alive():
                    raise RuntimeError('Transcription worker did not finish')

        first = next(o for o in observations if o['kind'] in ('preview', 'segment'))
        finals = [o for o in observations if o['kind'] == 'segment' and o['value'].get('source', 'system') == 'system']
        result = dict(name=name, reference=reference, audio_seconds=len(sound) / 48000,
                      speech_start_seconds=speech_start, speech_end_seconds=speech_end,
                      first_text_after_speech_start_seconds=first['seconds'] - speech_start,
                      final_after_speech_end_seconds=finals[-1]['seconds'] - speech_end,
                      max_backlog_seconds=max(o['value'] for o in observations if o['kind'] == 'backlog'),
                      text=''.join(o['value']['text'] for o in finals),
                      stages=stages, events=observations)
        if args.interview_flow:
            result['auto_questions'] = auto_questions
            result['first_auto_question_after_speech_end_seconds'] = auto_questions[0]['seconds']-speech_end if auto_questions else None
        if args.dual_source:
            candidate_rows = [o['value']['text'] for o in observations if o['kind'] == 'segment' and o['value']['source'] == 'microphone']
            result['candidate_text'] = candidate_rows
            result['candidate_context_preserved'] = bool(candidate_rows) and all(text in qa.context for text in candidate_rows)
        report['cases'].append(result)
        (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'stages')}, ensure_ascii=False), flush=True)
        if args.interview_flow:
            assert len(finals) >= 2, 'The fixture must exercise a multi-part question'
            assert len(auto_questions) == 1, 'The entire spoken question must submit once'
            assert auto_questions[0]['seconds'] >= speech_end, 'Automatic Q&A started before speech ended'
            assert auto_questions[0]['question'] == '\n'.join(o['value']['text'] for o in finals), 'Submitted question missed a transcript part'
        if args.dual_source:
            assert result['candidate_context_preserved'], 'Candidate speech must remain background context'


if __name__ == '__main__':
    main()
