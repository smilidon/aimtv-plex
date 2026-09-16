from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from aimtv import models


class RequiredModelSetupTests(unittest.TestCase):
    def test_declining_model_consent_does_not_download(self) -> None:
        with (
            patch.object(models, "models_present", return_value=False),
            patch("aimtv.alignment.alignment_model_present", return_value=False),
            patch("builtins.input", return_value="no"),
            patch("aimtv.alignment.fetch_alignment_model") as fetch_voice,
            patch.object(models, "fetch_models") as fetch_vision,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                models.fetch_required_models()
        fetch_voice.assert_not_called()
        fetch_vision.assert_not_called()

    def test_one_consent_installs_both_missing_local_models(self) -> None:
        voice_path = Path("/models/voice")
        vision_path = Path("/models/video")
        with (
            patch.object(models, "models_present", return_value=False),
            patch("aimtv.alignment.alignment_model_present", return_value=False),
            patch("builtins.input", return_value="yes"),
            patch(
                "aimtv.alignment.fetch_alignment_model",
                return_value=voice_path,
            ) as fetch_voice,
            patch.object(models, "fetch_models", return_value=vision_path) as fetch_vision,
            redirect_stdout(io.StringIO()),
        ):
            result = models.fetch_required_models()
        self.assertEqual(result, (voice_path, vision_path))
        fetch_voice.assert_called_once_with(assume_yes=True)
        fetch_vision.assert_called_once_with(assume_yes=True)


if __name__ == "__main__":
    unittest.main()
