"""Regenerate the README demo GIF and synthetic smoke images."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "examples" / "public_sample"
ASSET_ROOT = ROOT / "assets"
GREEN = "#8bd450"
DARK = "#0d1117"
MUTED = "#8b949e"
WHITE = "#f0f6fc"
BLUE = "#58a6ff"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def plant_smoke_image(index: int) -> Image.Image:
    image = Image.new("RGB", (512, 384), "#edf7e8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 300, 512, 384), fill="#d7c4a3")
    draw.line((256, 330, 256, 105), fill="#507c3a", width=12)
    for offset, side in ((0, -1), (55, 1), (110, -1)):
        center_y = 245 - offset
        center_x = 256 + side * 78
        points = []
        for step in range(40):
            angle = 2 * math.pi * step / 40
            radius_x = 62 * (1 + 0.12 * math.sin(index + angle * 3))
            radius_y = 34
            points.append((center_x + radius_x * math.cos(angle), center_y + radius_y * math.sin(angle)))
        draw.polygon(points, fill=("#75b84f", "#67a846", "#82bf59", "#6caf47")[index], outline="#386b2b")
        draw.line((256, center_y, center_x, center_y), fill="#507c3a", width=5)
    draw.ellipse(
        (218, 65, 294, 141),
        fill=("#d64f4f", "#7048b8", "#efb93f", "#df5b78")[index],
        outline="#703030",
        width=4,
    )
    draw.text((18, 18), f"synthetic smoke {index + 1}", fill="#35512b", font=font(22, bold=True))
    return image


def demo_frame(progress: int) -> Image.Image:
    image = Image.new("RGB", (1120, 560), DARK)
    draw = ImageDraw.Draw(image)
    draw.text((50, 36), "AADS", fill=GREEN, font=font(42, bold=True))
    draw.text((176, 46), "GPU-free evidence replay", fill=WHITE, font=font(28, bold=True))
    draw.text((50, 108), "Controlled demo · not production-ready", fill=MUTED, font=font(21))
    rows = [
        ("Controlled acceptance", "48 / 48", GREEN),
        ("Correct disease answers", "36 / 36", WHITE),
        ("Safe review / abstain", "12 / 12", BLUE),
        ("Negative false accepts", "0", GREEN),
        ("Wrong-part disease labels", "0", GREEN),
    ]
    visible = min(len(rows), progress)
    for row_index, (label, value, color) in enumerate(rows[:visible]):
        y = 182 + row_index * 62
        draw.rounded_rectangle((50, y - 8, 1070, y + 45), radius=10, fill="#161b22", outline="#30363d")
        draw.text((76, y), label, fill=MUTED, font=font(22))
        draw.text((850, y), value, fill=color, font=font(24, bold=True))
    if progress >= len(rows) + 1:
        draw.rounded_rectangle((50, 500, 1070, 542), radius=10, fill="#16351f")
        draw.text((355, 507), "PASS · fail-closed safety gates preserved", fill=GREEN, font=font(21, bold=True))
    return image


def main() -> None:
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for index in range(4):
        sample_path = SAMPLE_ROOT / f"synthetic_plant_{index + 1}.png"
        plant_smoke_image(index).save(sample_path, optimize=True)
        sample_rows.append(
            {
                "path": sample_path.name,
                "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                "width": 512,
                "height": 384,
                "source_kind": "programmatic_synthetic",
                "license": "MIT",
                "evidence_role": "io_smoke_only",
            }
        )
    (SAMPLE_ROOT / "manifest.json").write_text(
        json.dumps({"schema_version": "aads.public_sample.v1", "samples": sample_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    frames = [demo_frame(progress) for progress in range(1, 7)]
    frames.extend([frames[-1]] * 4)
    frames[0].save(
        ASSET_ROOT / "demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[350, 350, 350, 350, 350, 350, 600, 600, 600, 1200],
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
