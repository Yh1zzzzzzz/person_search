from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class LetterboxResize:
    """Resize with aspect ratio preserved, then pad to a square canvas.

    - Scales the image so that the longer side == target_size
    - Pads the shorter side with black pixels to reach target_size

    Input/Output: PIL.Image.Image
    """

    target_size: int
    fill: tuple[int, int, int] = (0, 0, 0)
    interpolation: int = Image.BICUBIC

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"LetterboxResize expects a PIL.Image.Image, got {type(img)}")

        target = int(self.target_size)
        if target <= 0:
            raise ValueError(f"target_size must be positive, got {target}")

        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size: {w}x{h}")

        scale = target / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = img.resize((new_w, new_h), resample=self.interpolation)
        canvas = Image.new("RGB", (target, target), self.fill)
        left = (target - new_w) // 2
        top = (target - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas
