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
    args = parser.parse_args()
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
    sounds = []
    for name, reference in cases:
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
    report = dict(model=args.model, backend=model.model.device, language='zh',
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

        class Stream:
            def __init__(self, callback):
                self.callback = callback

            def start_stream(self):
                clock['start'] = time.monotonic()

                def feed():
                    offset = 0
                    while not stop_feed.is_set() and not engine.stop_event.is_set():
                        deadline = clock['start'] + (offset + 4800) / 48000
                        if stop_feed.wait(max(0, deadline - time.monotonic())):
                            break
                        block = sound[offset:offset + 4800]
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
                return dict(defaultSampleRate=48000, maxInputChannels=2, isLoopbackDevice=True)

            def open(self, **kw):
                return Stream(kw['stream_callback'])

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
            engine.start(0, args.model, 'zh', hotwords)
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
                    if kind == 'preview' and isinstance(value, dict) and value['text']:
                        observations.append(dict(kind=kind, seconds=elapsed, text=value['text']))
                    elif kind in ('segment', 'backlog'):
                        observations.append(dict(kind=kind, seconds=elapsed, value=value))
                    elif kind == 'done':
                        break
                    # Let natural silence commit the last paragraph before stopping.
                    finals = [o for o in observations if o['kind'] == 'segment']
                    if finals and finals[-1]['value']['end'] >= speech_end - 0.6 and elapsed > speech_end + 1:
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
        finals = [o for o in observations if o['kind'] == 'segment']
        result = dict(name=name, reference=reference, audio_seconds=len(sound) / 48000,
                      speech_start_seconds=speech_start, speech_end_seconds=speech_end,
                      first_text_after_speech_start_seconds=first['seconds'] - speech_start,
                      final_after_speech_end_seconds=finals[-1]['seconds'] - speech_end,
                      max_backlog_seconds=max(o['value'] for o in observations if o['kind'] == 'backlog'),
                      text=''.join(o['value']['text'] for o in finals),
                      stages=stages, events=observations)
        report['cases'].append(result)
        (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'stages')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
