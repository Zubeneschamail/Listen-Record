from pathlib import Path
from faster_whisper import WhisperModel

if __name__ == "__main__":
    print("Preparing multilingual small model (first run needs internet)...", flush=True)
    WhisperModel("small", device="cpu", compute_type="int8",
                 download_root=str(Path(__file__).parent / "models"))
    print("Model ready for offline use.", flush=True)
