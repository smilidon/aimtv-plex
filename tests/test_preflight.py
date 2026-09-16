from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from aimtv import cli, preflight


class AiradioReadinessTests(unittest.TestCase):
    def test_missing_airadio_prints_pip_and_pipx_install_choices(self) -> None:
        output = io.StringIO()
        with patch.object(preflight, "airadio_installed", return_value=False):
            with redirect_stdout(output):
                code = preflight.check_airadio_ready()
        self.assertEqual(code, 1)
        self.assertIn("pip install airadio", output.getvalue())
        self.assertIn("pipx install airadio", output.getvalue())

    def test_empty_library_tells_user_to_run_airadio(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = io.StringIO()
            with (
                patch.object(preflight, "airadio_installed", return_value=True),
                patch.object(preflight, "airadio_home", return_value=Path(raw)),
                redirect_stdout(output),
            ):
                code = preflight.check_airadio_ready()
        self.assertEqual(code, 1)
        self.assertIn("not populated enough", output.getvalue())
        self.assertIn("Run `airadio` for a bit", output.getvalue())

    def test_run_stops_before_renderer_import_when_airadio_is_not_ready(self) -> None:
        with patch.object(cli, "check_airadio_ready", return_value=1):
            self.assertEqual(cli.main(["run"]), 1)


if __name__ == "__main__":
    unittest.main()
