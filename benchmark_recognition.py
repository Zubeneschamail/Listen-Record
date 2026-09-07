"""Local synthetic A/B check; no playback, recording, or network upload."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from faster_whisper.audio import decode_audio
from app import load_model, clean_caption
from recognition import DEFAULT_HOTWORDS, RecognitionContext
from segmentation import AudioChunk, WordAssembler, decode_chunk, RATE


def normalize(text):
    return "".join(c.casefold() for c in text if c.isalnum())


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--output", type=Path, default=Path("recordings/recognition-benchmark.json"))
    args = parser.parse_args()
    samples = ["我们正在讨论大模型，模型可以理解和生成文字。",
               "LangChain 和 LangGraph 有什么区别？MCP 用来连接工具。",
               "今天气温比较低，出门记得带上外套。"]
    model = load_model(args.model)
    results = []
    with tempfile.TemporaryDirectory(prefix="jianwen-benchmark-") as folder:
        audio = []
        for i, text in enumerate(samples):
            wav = Path(folder) / f"sample-{i}.wav"
            script = "$v=New-Object -ComObject SAPI.SpVoice; "
            script += "$voices=@($v.GetVoices() | Where-Object { $_.GetAttribute('Language') -match '^804' }); "
            script += "if($voices.Count -eq 0){throw 'Chinese SAPI voice not installed'}; $v.Voice=$voices[0]; "
            script += "$s=New-Object -ComObject SAPI.SpFileStream; "
            script += "$s.Open('" + str(wav).replace("'", "''") + "',3); $v.AudioOutputStream=$s; "
            script += "[void]$v.Speak('" + text.replace("'", "''") + "'); $s.Close()"
            subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            audio.append(decode_audio(str(wav), sampling_rate=RATE))
        for name in ("baseline", "guided"):
            guidance = RecognitionContext(DEFAULT_HOTWORDS) if name == "guided" else None
            offset = 0
            for reference, sound in zip(samples, audio):
                started = time.monotonic()
                rows = decode_chunk(model, WordAssembler(), AudioChunk(sound, offset, offset), "zh", guidance)
                elapsed = time.monotonic() - started
                text = clean_caption("".join(row[2] for row in rows))
                ref, hyp = normalize(reference), normalize(text)
                results.append(dict(mode=name, reference=reference, text=text,
                                    audio_seconds=len(sound)/RATE, elapsed_seconds=elapsed,
                                    normalized_cer=distance(ref, hyp)/max(1, len(ref))))
                offset += len(sound)/RATE+1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
