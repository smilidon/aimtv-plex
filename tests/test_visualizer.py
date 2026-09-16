from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from aimtv.audio_ops import Features
from aimtv.models import VisionBundle
from aimtv.visualizer import FrameState, step_frame


class _Pipe:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        grid = np.indices((32, 32)).sum(axis=0) % 2 * 255
        rgb = np.stack([grid, 255 - grid, grid], axis=-1).astype(np.uint8)
        return SimpleNamespace(images=[Image.fromarray(rgb)])


class VisualizerPromptTests(unittest.TestCase):
    def test_context_prompt_is_used_verbatim_without_legacy_rewrite(self) -> None:
        pipe = _Pipe()
        state = FrameState(
            image=Image.new("RGB", (32, 32), "gray"),
            prompt="old random channel",
            channel_t=99.0,
            subject="old subject",
        )
        prompt = "neon hedgehog, jakarta rain, vivid analog television"
        updated = step_frame(
            VisionBundle(pipe=pipe, device="cpu"),
            state,
            Features(rms=0.2, bass=0.1, mid=0.3, treble=0.2, beat=False),
            rng=random.Random(3),
            dt=0.125,
            prompt_override=prompt,
            prompt_changed=True,
            hard_reset=True,
        )
        self.assertEqual(updated.prompt, prompt)
        self.assertEqual(updated.subject, "")
        self.assertEqual(pipe.prompts, [prompt])


if __name__ == "__main__":
    unittest.main()
