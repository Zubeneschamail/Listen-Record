"""In-memory clipboard image previews, separate from the model's text history."""
import io

from PIL import Image, ImageTk, UnidentifiedImageError


class ConversationImage:
    def __init__(self, source, size):
        self.source = source
        self.size = size
        self.photo = None
        self.width = None

    @classmethod
    def from_bytes(cls, data):
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > 25_000_000:
                    return None
                size = source.size
                source.thumbnail((960, 960))
                return cls(source.convert('RGB'), size)
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            return None

    @property
    def caption(self):
        return f'剪贴板图片 · {self.size[0]}×{self.size[1]}'

    def thumbnail(self, master, width):
        width = max(1, width)
        if self.photo is None or self.width != width:
            preview = self.source.copy()
            preview.thumbnail((width, 360), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(preview, master=master)
            self.width = width
        return self.photo
