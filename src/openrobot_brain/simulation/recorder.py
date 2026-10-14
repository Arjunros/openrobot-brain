"""Render a captioned video of a run straight from the simulator (no screen recording).

* ``.mp4`` needs the optional extra: ``pip install -e ".[video]"`` (imageio + ffmpeg).
* ``.gif`` works with Pillow alone (smaller resolution and frame rate).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RecorderError(RuntimeError):
    pass


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class Recorder:
    def __init__(self, model: mujoco.MjModel, path: str | Path, camera: str = "front", fps: int | None = None,
                 width: int | None = None, height: int | None = None):
        self.path = Path(path)
        self.kind = self.path.suffix.lower()
        if self.kind not in (".mp4", ".gif"):
            raise RecorderError("recording path must end in .mp4 or .gif")
        self.fps = fps or (25 if self.kind == ".mp4" else 12)
        self.width = width or (960 if self.kind == ".mp4" else 560)
        self.height = height or (540 if self.kind == ".mp4" else 315)
        self.camera = camera
        self.title = ""
        self.subtitle = ""
        self.badge: tuple[str, tuple[int, int, int]] | None = None
        self.frames = 0
        try:
            self._renderer = mujoco.Renderer(model, self.height, self.width)
        except Exception as exc:
            raise RecorderError(f"offscreen rendering unavailable for recording: {exc}") from exc
        self._gif_frames: list[Image.Image] = []
        self._writer = None
        if self.kind == ".mp4":
            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise RecorderError('MP4 recording needs: pip install -e ".[video]"  (or record a .gif)') from exc
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=8,
                                              macro_block_size=8)
        self._f_big, self._f_small = _font(max(14, self.height // 22)), _font(max(11, self.height // 32))

    def annotate(self, title: str | None = None, subtitle: str | None = None,
                 badge: tuple[str, tuple[int, int, int]] | None | bool = False) -> None:
        if title is not None:
            self.title = title
        if subtitle is not None:
            self.subtitle = subtitle
        if badge is not False:
            self.badge = badge  # type: ignore[assignment]

    def _overlay(self, img: np.ndarray) -> Image.Image:
        im = Image.fromarray(img).convert("RGBA")
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pad = self.height // 40
        bar_h = int(self.height * 0.15)
        d.rectangle([0, 0, self.width, bar_h], fill=(10, 12, 16, 170))
        d.text((pad, pad), self.title, font=self._f_big, fill=(255, 255, 255, 255))
        d.text((pad, pad + int(self.height * 0.065)), self.subtitle[:110], font=self._f_small,
               fill=(200, 210, 220, 255))
        if self.badge:
            text, rgb = self.badge
            bw = d.textlength(text, font=self._f_small) + 2 * pad
            x0, y0 = self.width - bw - pad, self.height - int(self.height * 0.09)
            d.rounded_rectangle([x0, y0, x0 + bw, y0 + int(self.height * 0.065)], radius=6, fill=(*rgb, 230))
            d.text((x0 + pad, y0 + int(self.height * 0.012)), text, font=self._f_small, fill=(255, 255, 255, 255))
        return Image.alpha_composite(im, layer).convert("RGB")

    def capture(self, data: mujoco.MjData) -> None:
        self._renderer.update_scene(data, camera=self.camera)
        frame = self._overlay(self._renderer.render())
        if self._writer is not None:
            self._writer.append_data(np.asarray(frame))
        else:
            self._gif_frames.append(frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        self.frames += 1

    def still(self, data: mujoco.MjData, seconds: float) -> None:
        for _ in range(max(1, int(seconds * self.fps))):
            self.capture(data)

    def close(self) -> Path | None:
        try:
            if self._writer is not None:
                self._writer.close()
            elif self._gif_frames:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._gif_frames[0].save(self.path, save_all=True, append_images=self._gif_frames[1:],
                                         duration=int(1000 / self.fps), loop=0, optimize=True)
        finally:
            self._renderer.close()
        return self.path if self.frames else None
