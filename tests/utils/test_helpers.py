from pathlib import Path

from PIL import Image

# Shared test image defaults
IMAGE_SIZE = (8, 8)
IMAGE_COLOR = (255, 0, 0)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def make_image(path: Path, size: tuple[int, int] | None = None, color: tuple | str | None = None) -> None:
    """Create and save a simple RGB image to `path`. Creates parent dirs as needed."""
    size = size or IMAGE_SIZE
    color = color or IMAGE_COLOR
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
