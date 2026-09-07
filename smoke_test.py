"""Play a known sentence through Windows and recognize actual loopback audio."""
import queue
import subprocess
import time
from app import Transcriber, pa


def main():
    events = queue.Queue()
    worker = Transcriber(events)
    with pa.PyAudio() as audio:
        device = audio.get_default_wasapi_loopback()
        print("DEVICE", device["name"], flush=True)
    worker.start(int(device["index"]), "base", "en")
    texts, ready, peak = [], False, 0
    started = time.monotonic()
    playback = None
    deadline = started + 180
    try:
        while time.monotonic() < deadline:
            try:
                kind, value = events.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "status":
                print("STATUS", value, flush=True)
                if "正在转写系统声音" in value:
                    ready = True
                    playback = subprocess.Popen([
                        "powershell", "-NoProfile", "-Command",
                        "$v = New-Object -ComObject SAPI.SpVoice; "
                        "$v.Rate = -1; [void]$v.Speak('Hello world. This is a test of system audio transcription. The computer can turn sound into text.')"],
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    started = time.monotonic()
            elif kind == "level":
                peak = max(peak, value)
            elif kind == "segment":
                texts.append(value["text"])
                print("TEXT", value, flush=True)
            elif kind == "error":
                raise RuntimeError(value)
            elif kind == "done":
                break
            if ready and time.monotonic() - started > 15:
                worker.stop()
        else:
            raise TimeoutError("Smoke test timed out")
        combined = " ".join(texts).lower()
        assert peak > 0, "No loopback audio captured"
        assert "hello" in combined and "text" in combined, combined
        print("PASS: real loopback audio recognized; stop drained remaining audio.", flush=True)
    finally:
        worker.stop()
        worker.thread.join(timeout=30)
        if playback and playback.poll() is None:
            playback.terminate()


if __name__ == "__main__":
    main()
