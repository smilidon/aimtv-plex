"""Distribution identity must not drift from the CLI or install upstream."""

from __future__ import annotations

import io
import json
import re
import unittest
from contextlib import redirect_stdout
from importlib import metadata, resources

import aimtv
from aimtv.cli import main


class PackagingTests(unittest.TestCase):
    def test_distribution_and_runtime_versions_match(self) -> None:
        self.assertEqual(metadata.version("aimtv-plex"), aimtv.__version__)
        self.assertEqual(metadata.metadata("aimtv-plex")["Name"], "aimtv-plex")

    def test_cli_reports_the_distribution_version(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"aimtv {aimtv.__version__}")

    def test_no_recursive_or_upstream_distribution_dependency(self) -> None:
        for requirement in metadata.requires("aimtv-plex") or []:
            name = re.split(r"[\s\[<>=!~;(]", requirement, maxsplit=1)[0]
            name = re.sub(r"[-_.]+", "-", name).lower()
            self.assertNotIn(name, {"aimtv", "aimtv-plex"})

    def test_packaged_prompt_bank_is_present(self) -> None:
        path = resources.files("aimtv").joinpath("data/prompts.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(data["prompts"])
        self.assertTrue(data["subjects"])


if __name__ == "__main__":
    unittest.main()
