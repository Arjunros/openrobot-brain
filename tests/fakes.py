"""Test doubles for the OpenAI client.

``FakeVisionClient`` mimics a vision model honestly: it decodes the PNG it is
sent, finds coloured blobs, and reports their pixel centres. It never reads
simulator state, so tests that use it exercise the real image -> pixels ->
metres -> grasp path.
"""

from __future__ import annotations

import base64
import io
import json
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image

THRESH = {
    "red": lambda r, g, b: (r > 170) & (g < 130) & (b < 130),
    "green": lambda r, g, b: (g > 170) & (r < 130) & (b < 160),
    "blue": lambda r, g, b: (b > 190) & (r < 130) & (g < 200),
}


def find_blobs(png: bytes) -> dict[str, tuple[float, float]]:
    return {c: (u, v) for c, (u, v, _) in find_blobs_with_yaw(png).items()}


def square_yaw_deg(xs: np.ndarray, ys: np.ndarray) -> float:
    """Edge angle of a square blob, counter-clockwise as seen in the image, in [-45, 45).

    A square has no long axis, so a covariance/PCA angle is meaningless. Instead use the
    4-fold moment sum(r^4 e^{4i phi}): its angle points at the corners (45 degrees off the edges)."""
    du, up = xs - xs.mean(), -(ys - ys.mean())         # image v grows downward
    r2, phi = du * du + up * up, np.arctan2(up, du)
    yaw = np.angle(np.sum(r2 * r2 * np.exp(4j * phi))) / 4 - np.pi / 4
    return float(np.degrees((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4))


def find_blobs_with_yaw(png: bytes) -> dict[str, tuple[float, float, float]]:
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    out = {}
    for color, f in THRESH.items():
        ys, xs = np.nonzero(f(r, g, b))
        if xs.size > 30:
            out[color] = (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, square_yaw_deg(xs, ys))
    return out


def image_from_request(kwargs) -> bytes | None:
    for part in kwargs["input"][0]["content"]:
        if part["type"] == "input_image":
            return base64.b64decode(part["image_url"].split(",", 1)[1])
    return None


class FakeVisionClient:
    """Policy for 'stack the <a> cube on the <b> cube', seen only through the camera.

    ``bias_px`` shifts reported u on the first pick attempt, to simulate a perception error.
    """

    def __init__(self, bias_px: float = 0.0, report_yaw: bool = True):
        self.calls: list[dict] = []
        self.bias_px = bias_px
        self.report_yaw = report_yaw
        self.responses = SimpleNamespace(create=self.create)
        self._base_px: tuple[float, float] | None = None
        self._base_yaw = 0.0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        png = image_from_request(kwargs)
        assert png is not None, "vision request without an image"
        prompt = kwargs["input"][0]["content"][0]["text"]
        a, b = re.findall(r"\b(red|green|blue)\b", prompt.split("\n")[0])[:2]
        blobs3 = find_blobs_with_yaw(png)
        blobs = {c: (u, v) for c, (u, v, _) in blobs3.items()}
        obs = []
        for color, (u, v, yaw) in blobs3.items():
            if color == a and self.bias_px and len(self.calls) == 1:
                u += self.bias_px
            if not self.report_yaw:
                yaw = 0.0
            obs.append({"id": f"{color}_cube", "label": f"{color} cube", "u": u, "v": v, "yaw_deg": yaw,
                        "on_top_of": ""})

        if b in blobs and self._base_px is None:
            self._base_px = blobs[b]          # remember where the base cube sits before stacking
            self._base_yaw = blobs3[b][2] if self.report_yaw else 0.0
        # After stacking, the base cube's side faces can still show at the edges, so judge by whether
        # the top cube now sits where the base cube was.
        stacked = (a in blobs and self._base_px is not None
                   and np.hypot(blobs[a][0] - self._base_px[0], blobs[a][1] - self._base_px[1]) < 12)
        if stacked:
            obs = [o for o in obs if o["id"] != f"{b}_cube"]
            obs.append({"id": f"{b}_cube", "label": f"{b} cube", "u": self._base_px[0], "v": self._base_px[1],
                        "yaw_deg": self._base_yaw, "on_top_of": ""})
            for o in obs:
                if o["id"] == f"{a}_cube":
                    o["on_top_of"] = f"{b}_cube"
            body = {"status": "done", "message": f"The {a} cube is on the {b} cube.", "observations": obs,
                    "actions": []}
        elif "holding" in prompt.split("GRIPPER:")[1].split("\n")[0]:
            body = {"status": "act", "message": "Placing.", "observations": obs,
                    "actions": [{"action": "place_on", "object": f"{b}_cube"}]}
        else:
            body = {"status": "act", "message": f"Stacking {a} on {b}.", "observations": obs,
                    "actions": [{"action": "pick", "object": f"{a}_cube"},
                                {"action": "place_on", "object": f"{b}_cube"}]}
        return SimpleNamespace(output_text=json.dumps(body), status="completed", incomplete_details=None)


class ScriptedClient:
    """Returns canned texts in order; records requests."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.texts.pop(0), status="completed", incomplete_details=None)
