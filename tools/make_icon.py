"""Build app PNG/Windows ICO assets from the supplied full-resolution artwork."""
from pathlib import Path
from io import BytesIO
import struct
from PIL import Image

assets = Path(__file__).resolve().parents[1] / "assets"
assets.mkdir(exist_ok=True)
source = Image.open(assets / 'logo-source.png').convert('RGBA')


def render(size):
    # Always resample from the original; avoid successive resizing losses.
    return source.resize((size, size), Image.Resampling.LANCZOS)


for size in (24, 32, 256):
    render(size).save(assets / f"logo-{size}.png")
# Explicit ICO frames preserve the pixel-fitted versions; Pillow's default ICO
# writer otherwise scales all small frames from the largest image.
sizes = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
payloads = []
for size in sizes:
    buffer = BytesIO()
    render(size).save(buffer, format="PNG")
    payloads.append(buffer.getvalue())
offset = 6 + 16*len(sizes)
directory = []
for size, payload in zip(sizes, payloads):
    directory.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(payload), offset))
    offset += len(payload)
(assets / "wenlu.ico").write_bytes(struct.pack("<HHH", 0, 1, len(sizes)) + b"".join(directory) + b"".join(payloads))
print("Created PNGs and ten-size Windows icon from supplied artwork")
