"""The v2 age copy for version 2's evaluation script, and the double-click detection of the launchers."""
from __future__ import annotations

import contextlib
import csv
import io
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from fakes_vlm import make_photo, photo_answer
from test_main import T0, CliTestCase

from trailcam_vlm import __main__ as cli
from trailcam_vlm import v2_ages


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class TeensInUnknownTests(unittest.TestCase):
    def test_row(self):
        self.assertEqual(v2_ages.teens_in_unknown({"teens": "2", "age_unknown": "1", "adults": "3"}),
                         {"teens": "2", "age_unknown": "3", "adults": "3"})
        for row in ({"teens": "", "age_unknown": "1"}, {"teens": "2", "age_unknown": ""}, {"teens": "x", "age_unknown": "1"}):
            with self.subTest(row=row):
                self.assertEqual(v2_ages.teens_in_unknown(row), row)


class ConvertExportTests(CliTestCase):
    def test_export_folder_is_converted(self):
        make_photo(self.input / "cam" / "a.jpg", people=3, when=T0)
        make_photo(self.input / "cam" / "b.jpg", people=2, when=T0 + timedelta(seconds=10))
        self.fake.photo_reply = lambda people, bicycles, body: photo_answer(
            people, adults=1, teens=people - 1, children=0, age_unclear=0, notes="=SUM(A1)")
        result = self.run_cli()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(v2_ages.main([str(result.folder)]), 0)
        images = result.path(".csv").with_name(result.folder.name + "_v2ages.csv")
        events = result.path("_events.csv").with_name(result.folder.name + "_events_v2ages.csv")
        self.assertIn(str(images), out.getvalue())
        original, copy = result.images, read(images)
        self.assertEqual([r["relative_path"] for r in copy], [r["relative_path"] for r in original])
        for before, after in zip(original, copy):
            self.assertEqual(int(after["age_unknown"]), int(before["teens"]) + int(before["age_unknown"]))
            self.assertEqual({k: v for k, v in after.items() if k != "age_unknown"},
                             {k: v for k, v in before.items() if k != "age_unknown"})  # notes stay escaped once
        self.assertTrue(events.is_file())
        self.assertEqual(len(read(events)), len(result.events))
        self.assertEqual(list(read(images)[0]), list(original[0]))  # same columns, same order

    def test_bad_inputs(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(v2_ages.main([str(self.tmp / "missing")]), 2)
        other = self.tmp / "other.csv"
        other.write_text("a,b\n1,2\n", encoding="utf-8")
        with contextlib.redirect_stderr(err):
            self.assertEqual(v2_ages.main([str(other)]), 2)
        self.assertIn("no teens and age_unknown columns", err.getvalue())


class DoubleClickTests(unittest.TestCase):
    def test_only_the_launching_cmd(self):
        own, launcher, cmd, shell = 30, 20, 10, 5
        # Double-clicked: cmd.exe + venv launcher + this Python.
        self.assertTrue(cli.only_the_launching_cmd([cmd, launcher, own], own, launcher, in_venv=True))
        # Without a venv launcher: cmd.exe + this Python.
        self.assertTrue(cli.only_the_launching_cmd([cmd, own], own, cmd, in_venv=False))
        # From PowerShell (or an open cmd window): one more process on the console.
        self.assertFalse(cli.only_the_launching_cmd([shell, cmd, launcher, own], own, launcher, in_venv=True))
        self.assertFalse(cli.only_the_launching_cmd([shell, cmd, own], own, cmd, in_venv=False))


if __name__ == "__main__":
    unittest.main()
