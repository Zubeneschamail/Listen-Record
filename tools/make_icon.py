"""Build PNG/Windows ICO assets from the same geometric design as logo.svg.

Development-only dependency: Pillow. The app itself does not need Pillow.
"""
from pathlib import Path
from io import BytesIO
import struct
from PIL import Image, ImageDraw

assets = Path(__file__).resolve().parents[1] / "assets"
assets.mkdir(exist_ok=True)
accent = "#007ACC"


def render(size):
    # Render each target size independently. Integer-aligned straight edges and
    # thicker small-size bars avoid the blur from repeatedly downsampling a 256px icon.
    scale = 4
    canvas = Image.new("RGBA", (size * scale, size * scale))
    draw = ImageDraw.Draw(canvas)
    def rounded(box, radius, color):
        x0, y0, x1, y1 = (v * scale for v in box)
        draw.rounded_rectangle((x0, y0, x1-1, y1-1), radius=radius*scale, fill=color)
    rounded((0, 0, size, size), max(3, round(size*.22)), accent)
    left, top = round(size*.18), round(size*.18)
    right, bottom = round(size*.82), round(size*.72)
    draw.polygon([(left*scale, round(size*.59)*scale),
                  (left*scale, round(size*.84)*scale),
                  (round(size*.43)*scale, round(size*.64)*scale)], fill="white")
    rounded((left, top, right, bottom), max(2, round(size*.14)), "white")
    width = max(2, round(size*.065))
    for center, y0, y1 in [(.34, .37, .55), (.5, .28, .64), (.66, .34, .58)]:
        x = round(size*center)-width//2
        rounded((x, round(size*y0), x+width, round(size*y1)), width/2, accent)
    return canvas.resize((size, size), Image.Resampling.BOX)


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
(assets / "jianwen.ico").write_bytes(struct.pack("<HHH", 0, 1, len(sizes)) + b"".join(directory) + b"".join(payloads))
print("Created pixel-fitted PNGs and ten-size Windows icon")
