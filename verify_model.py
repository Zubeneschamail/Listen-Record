"""Deterministic local speech recognition check; does not play or record audio."""
import subprocess
import argparse
import tempfile
import time
from pathlib import Path
from app import load_model, clean_caption

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="base")
parser.add_argument("--language", default="en", choices=["en", "zh"])
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--force-cuts", action="store_true", help="Stress-test overlap with forced 5s cuts")
parser.add_argument("--preview", action="store_true", help="Exercise provisional then final decoding")
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix="shengji-test-") as folder:
    wav = Path(folder) / "speech.wav"
    # This path is generated locally, never sourced from user text.
    script = """$voice = New-Object -ComObject SAPI.SpVoice
$english = @($voice.GetVoices() | Where-Object { $_.GetAttribute('Language') -match '^409' })
if ($english.Count -gt 0) { $voice.Voice = $english[0] }
$stream = New-Object -ComObject SAPI.SpFileStream
$stream.Open('__PATH__', 3)
$voice.AudioOutputStream = $stream
[void]$voice.Speak('Hello world. This is a test of system audio transcription. The computer can turn sound into text.')
$stream.Close()
""".replace("__PATH__", str(wav).replace("'", "''"))
    if args.language == "zh":
        script = script.replace("^409", "^804").replace(
            "Hello world. This is a test of system audio transcription. The computer can turn sound into text.",
            "你好，这是系统声音转文字的测试。电脑可以直接识别正在播放的声音。")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    model = load_model(args.model)
    started = time.monotonic()
    if args.streaming:
        import numpy as np
        from faster_whisper.audio import decode_audio
        from segmentation import PauseSegmenter, WordAssembler, decode_chunk, DraftPreview
        sound = decode_audio(str(wav), sampling_rate=16000)
        duration = len(sound) / 16000
        sound = np.r_[sound, np.zeros(16000, dtype=np.float32)]
        splitter, assembler = PauseSegmenter(), WordAssembler()
        if args.force_cuts:
            splitter = PauseSegmenter(lambda a: [{"start": 0, "end": len(a)}], maximum=5)
        chunks = []
        final_rows, drafts = [], []
        audio_clock = [0.0]
        preview = DraftPreview(clock=lambda: audio_clock[0])

        def confirm(batch):
            chunks.extend(batch)
            for chunk in batch:
                final_rows.extend(decode_chunk(model, assembler, chunk, args.language))
                preview.defer()

        for offset in range(0, len(sound), 1600):
            audio_clock[0] = offset / 16000
            confirm(splitter.push(sound[offset:offset + 1600], offset / 16000, 1000 + offset / 16000))
            if args.preview:
                draft = preview.render(model, assembler, splitter, args.language)
                if draft:
                    drafts.append(clean_caption(draft))
        confirm(splitter.finish())
        print("SPLITS", [(round(c.start, 2), round(len(c.audio) / 16000, 2), c.forced) for c in chunks])
        if args.preview:
            print("DRAFTS", drafts)
            assert drafts, "No draft appeared before final transcription"
        text = " ".join(clean_caption(row[2]) for row in final_rows)
    else:
        segments, info = model.transcribe(str(wav), language=args.language, beam_size=1, vad_filter=True,
                                         condition_on_previous_text=False, max_new_tokens=128,
                                         repetition_penalty=1.1, no_repeat_ngram_size=3)
        text = " ".join(clean_caption(s.text) for s in segments)
        duration = info.duration
    print(text)
    words = ["hello", "text"] if args.language == "en" else ["声音", "文字"]
    assert all(word in text.lower() for word in words), text
    print(f"PASS: {duration:.1f}s audio recognized in {time.monotonic()-started:.2f}s")
