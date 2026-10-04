"""Command line and end-to-end runs against a fake Ollama server (no network, no GPU).

Every test patches ``trailcam_vlm.__main__.ROOT`` to a temporary folder, so the
repository's own settings.json, images/, exports/ and .cache/ are never read
or written.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import fakes_vlm
import requests
from fakes_vlm import FakeOllama, FakeResponse, make_photo, photo_answer, set_mtime
from PIL import Image

from trailcam_vlm import __main__ as cli
from trailcam_vlm import __version__
from trailcam_vlm.export import EVENT_FIELDS, IMAGE_FIELDS, SUMMARY_FIELDS
from trailcam_vlm.prompts import EVENT_PROMPT, EVENT_SCHEMA, PHOTO_PROMPT, PHOTO_SCHEMA, PROMPT_VERSION

T0 = datetime(2024, 5, 3, 7, 30, 0)
BROKEN_TIME = datetime(2024, 6, 1, 12, 0, 0)
MULTI_EVENT = "North_gate__2024-05-03T07-30-00"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Export:
    """One finished run: exit code, console output and the export folder."""

    def __init__(self, code, stdout, stderr, folder):
        self.code, self.stdout, self.stderr, self.folder = code, stdout, stderr, folder

    def path(self, suffix):
        return self.folder / (self.folder.name + suffix)

    def rows(self, suffix):
        with open(self.path(suffix), newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    @property
    def images(self):
        return self.rows(".csv")

    @property
    def image(self):
        return {row["relative_path"]: row for row in self.images}

    @property
    def events(self):
        return self.rows("_events.csv")

    @property
    def event(self):
        return {row["event_id"]: row for row in self.events}

    @property
    def camera_days(self):
        return self.rows("_camera_days.csv")

    @property
    def run(self):
        return json.loads(self.path("_run.json").read_text(encoding="utf-8"))


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trailcam_vlm_cli_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        patcher = mock.patch.object(cli, "ROOT", self.repo)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.input = self.tmp / "photos"
        self.input.mkdir()
        self.output = self.tmp / "exports"
        self.cache = self.tmp / "cache"
        self.fake = FakeOllama()

    def run_cli(self, *extra, paths=True):
        argv = ["--input", str(self.input), "--output", str(self.output), "--cache", str(self.cache)] if paths else []
        argv += [str(a) for a in extra]
        before = set(self.output.glob("export_*")) if self.output.exists() else set()
        out, err = io.StringIO(), io.StringIO()
        with self.fake.installed(), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        new = sorted(set(self.output.glob("export_*")) - before) if self.output.exists() else []
        self.assertLessEqual(len(new), 1)
        return Export(code, out.getvalue(), err.getvalue(), new[0] if new else None)

    def standard_photos(self):
        """Two cameras: a 3-photo visit + a lone photo at North gate; one photo and a corrupt file at Spring trail."""
        gate, spring = self.input / "North gate", self.input / "Spring trail"
        make_photo(gate / "IMG_0001.jpg", people=1, when=T0)
        make_photo(gate / "IMG_0002.jpg", people=2, when=T0 + timedelta(seconds=20))
        make_photo(gate / "IMG_0003.jpg", people=1, bicycles=1, when=T0 + timedelta(seconds=40))
        make_photo(gate / "IMG_0010.jpg", people=0, when=datetime(2024, 5, 3, 9, 0, 0))
        make_photo(spring / "IMG_0100.jpg", people=3, when=datetime(2024, 5, 4, 10, 0, 0))
        broken = spring / "broken.jpg"
        broken.write_bytes(b"this is not a photograph")
        set_mtime(broken, BROKEN_TIME)

    def cache_files(self, kind):
        return sorted((self.cache / kind).glob("*/*.json"))


class FirstRunTests(CliTestCase):
    def setUp(self):
        super().setUp()
        self.standard_photos()
        self.result = self.run_cli()

    def test_exit_code_and_export_files(self):
        result = self.result
        self.assertEqual(result.code, 1, result.stderr)  # one photo could not be analysed
        self.assertTrue(result.folder.name.startswith("export_"))
        self.assertEqual(sorted(p.name for p in result.folder.iterdir()),
                         sorted(result.folder.name + s for s in (".csv", "_events.csv", "_camera_days.csv",
                                                                  "_run.json")))
        for suffix, fields in ((".csv", IMAGE_FIELDS), ("_events.csv", EVENT_FIELDS),
                               ("_camera_days.csv", SUMMARY_FIELDS)):
            raw = result.path(suffix).read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), suffix)
            self.assertEqual(raw[3:].split(b"\r\n", 1)[0].decode("utf-8").split(","), fields)

    def test_every_photo_is_asked_once_and_the_visit_once(self):
        self.assertEqual(len(self.fake.tag_requests), 1)
        self.assertEqual(len(self.fake.photo_requests), 5)
        self.assertEqual(len(self.fake.event_requests), 1)  # only the 3-photo event; single photos need none
        self.assertEqual([(p, b) for p, b, *_ in self.fake.event_requests[0]["images"]], [(1, 0), (2, 0), (1, 1)])

    def test_image_rows(self):
        rows = self.result.images
        self.assertEqual([r["relative_path"] for r in rows],
                         ["North gate/IMG_0001.jpg", "North gate/IMG_0002.jpg", "North gate/IMG_0003.jpg",
                          "North gate/IMG_0010.jpg", "Spring trail/broken.jpg", "Spring trail/IMG_0100.jpg"])
        by_path = self.result.image
        expected = {"North gate/IMG_0001.jpg": "1", "North gate/IMG_0002.jpg": "2", "North gate/IMG_0003.jpg": "1",
                    "North gate/IMG_0010.jpg": "0", "Spring trail/IMG_0100.jpg": "3"}
        for relative, people in expected.items():
            with self.subTest(photo=relative):
                row = by_path[relative]
                digest = sha256(self.input / relative)
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["error"], "")
                self.assertEqual(row["people_total"], people)
                self.assertEqual(row["adults"], people)
                self.assertEqual(row["dir_left"], people)
                self.assertEqual(row["sha256"], digest)
                self.assertEqual(row["image_id"], "img_" + digest[:16])
                self.assertEqual(row["cache_hit"], "False")
                self.assertEqual((row["model"], row["prompt_version"]), ("gemma4:26b", PROMPT_VERSION))
                self.assertEqual((row["requests"], row["repaired"], row["consistency_problems"]), ("1", "False", ""))
                self.assertEqual(row["time_source"], "exif_original")
                self.assertEqual(row["people_count_assessable"], "True")
        self.assertEqual(by_path["Spring trail/IMG_0100.jpg"]["people_class"], "3–4")
        self.assertEqual(by_path["North gate/IMG_0010.jpg"]["people_class"], "0")
        self.assertEqual(by_path["North gate/IMG_0003.jpg"]["bicycles"], "1")
        self.assertEqual(by_path["North gate/IMG_0001.jpg"]["capture_time"], "2024-05-03T07:30:00")
        self.assertEqual(by_path["North gate/IMG_0001.jpg"]["camera_id"], "North gate")
        self.assertEqual(by_path["North gate/IMG_0001.jpg"]["run_id"], self.result.run["run_id"])

    def test_corrupt_photo_becomes_an_error_row(self):
        row = self.result.image["Spring trail/broken.jpg"]
        self.assertEqual(row["status"], "error")
        self.assertTrue(row["error"].startswith("UnidentifiedImageError: "), row["error"])
        self.assertEqual((row["people_total"], row["people_class"], row["requests"]), ("", "", ""))
        self.assertEqual((row["capture_time"], row["time_source"]), ("2024-06-01T12:00:00", "file_mtime"))
        self.assertEqual(row["sha256"], sha256(self.input / "Spring trail" / "broken.jpg"))
        self.assertIn("ERROR Spring trail/broken.jpg: UnidentifiedImageError", self.result.stderr)
        # The photo after the corrupt one was still analysed.
        self.assertEqual(self.result.image["Spring trail/IMG_0100.jpg"]["status"], "ok")

    def test_event_columns_in_the_image_rows(self):
        by_path = self.result.image
        for frame, name in enumerate(("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"), 1):
            row = by_path["North gate/" + name]
            self.assertEqual((row["event_id"], row["event_frame"], row["event_image_count"]), (MULTI_EVENT, str(frame), "3"))
        row = by_path["North gate/IMG_0010.jpg"]
        self.assertEqual((row["event_id"], row["event_frame"], row["event_image_count"]),
                         ("North_gate__2024-05-03T09-00-00", "1", "1"))
        self.assertEqual(by_path["Spring trail/broken.jpg"]["event_id"], "Spring_trail__2024-06-01T12-00-00")

    def test_event_rows(self):
        rows = self.result.events
        self.assertEqual([r["event_id"] for r in rows],
                         [MULTI_EVENT, "North_gate__2024-05-03T09-00-00", "Spring_trail__2024-05-04T10-00-00",
                          "Spring_trail__2024-06-01T12-00-00"])
        visit = rows[0]
        self.assertEqual(visit["method"], "multi_photo_model")
        self.assertEqual(visit["status"], "ok")
        self.assertEqual((visit["people_unique"], visit["people_max_frame"], visit["people_unique_class"]),
                         ("3", "2", "3–4"))
        self.assertEqual((visit["start"], visit["end"], visit["duration_seconds"], visit["image_count"]),
                         ("2024-05-03T07:30:00", "2024-05-03T07:30:40", "40.0", "3"))
        self.assertEqual(visit["bicycles"], "1")
        self.assertEqual((visit["needs_review"], visit["review_reasons"]), ("False", ""))
        self.assertEqual(json.loads(visit["images"]),
                         ["North gate/IMG_0001.jpg", "North gate/IMG_0002.jpg", "North gate/IMG_0003.jpg"])
        self.assertEqual(visit["requests"], "1")
        lone = rows[1]
        self.assertEqual((lone["method"], lone["people_unique"], lone["people_unique_class"]), ("single_photo", "0", "0"))
        self.assertEqual((rows[2]["method"], rows[2]["people_unique"]), ("single_photo", "3"))
        failed = rows[3]
        self.assertEqual((failed["method"], failed["status"], failed["people_unique"]), ("per_photo_max", "error", ""))
        self.assertEqual((failed["needs_review"], failed["review_reasons"]),
                         ("True", "unique_counts_not_available; photo_analysis_failed"))

    def test_camera_days(self):
        rows = self.result.camera_days
        self.assertEqual([(r["camera_id"], r["date"]) for r in rows],
                         [("North gate", "2024-05-03"), ("Spring trail", "2024-05-04"), ("Spring trail", "2024-06-01")])
        self.assertEqual((rows[0]["events"], rows[0]["images"], rows[0]["people_unique"], rows[0]["people_max_frame"],
                          rows[0]["events_needing_review"]), ("2", "4", "3", "2", "0"))
        self.assertEqual((rows[1]["people_unique"], rows[1]["adults"]), ("3", "3"))
        self.assertEqual((rows[2]["images"], rows[2]["people_unique"], rows[2]["events_needing_review"]), ("1", "", "1"))

    def test_run_summary(self):
        run = self.result.run
        self.assertEqual(run["version"], __version__)
        self.assertEqual(run["input_dir"], str(self.input.resolve()))
        self.assertEqual((run["model"], run["ollama_host"], run["prompt_version"]),
                         ("gemma4:26b", "http://localhost:11434", PROMPT_VERSION))
        self.assertEqual((run["photos"], run["cached_photos"], run["photos_with_errors"]), (6, 0, 1))
        self.assertEqual((run["events"], run["multi_photo_event_requests"]), (4, 1))
        self.assertEqual((run["max_side"], run["event_max_side"], run["max_event_frames"], run["event_gap_seconds"]),
                         (2048, 1280, 6, 60))
        self.assertIs(run["events_enabled"], True)
        self.assertEqual((run["camera_id_mode"], run["camera_id_pattern"]), ("folder", None))
        self.assertIsInstance(run["total_seconds"], float)
        self.assertTrue(run["finished_at_utc"].endswith("+00:00"))
        self.assertTrue(self.result.folder.name.endswith(run["run_id"]))

    def test_requests_sent_to_the_model(self):
        for request in self.fake.chat_requests:
            body = request["body"]
            self.assertEqual(request["url"], "http://localhost:11434/api/chat")
            self.assertEqual(request["timeout"], 600)
            self.assertEqual(body["model"], "gemma4:26b")
            self.assertIs(body["stream"], False)
            self.assertIs(body["think"], False)
            self.assertEqual(body["options"]["temperature"], 0)
            self.assertEqual(body["options"]["num_predict"], 700)
            self.assertEqual(len(body["messages"]), 1)
            for *_, size, fmt in request["images"]:
                self.assertEqual(fmt, "JPEG")
        for request in self.fake.photo_requests:
            self.assertEqual(request["body"]["format"], PHOTO_SCHEMA)
            self.assertEqual(request["body"]["messages"][0]["content"], PHOTO_PROMPT)
            self.assertEqual(len(request["images"]), 1)
        event = self.fake.event_requests[0]["body"]
        self.assertEqual(event["format"], EVENT_SCHEMA)
        self.assertEqual(event["messages"][0]["content"], EVENT_PROMPT.format(n=3))
        self.assertEqual(self.fake.tag_requests, [{"url": "http://localhost:11434/api/tags", "timeout": 5}])

    def test_answers_are_cached_per_photo_and_per_visit(self):
        photo_files = self.cache_files("photos")
        self.assertEqual(len(photo_files), 5)
        self.assertEqual(len({p.parent for p in photo_files}), 1)
        expected = {sha256(p) + ".json" for p in self.input.rglob("IMG_*.jpg")}
        self.assertEqual({p.name for p in photo_files}, expected)
        cached = json.loads(photo_files[0].read_text(encoding="utf-8"))
        self.assertEqual(set(cached), {"answer", "info", "model", "prompt_version"})
        self.assertEqual((cached["model"], cached["prompt_version"]), ("gemma4:26b", PROMPT_VERSION))
        event_files = self.cache_files("events")
        self.assertEqual(len(event_files), 1)
        cached = json.loads(event_files[0].read_text(encoding="utf-8"))
        self.assertEqual(cached["photos"], ["North gate/IMG_0001.jpg", "North gate/IMG_0002.jpg",
                                            "North gate/IMG_0003.jpg"])
        self.assertEqual(cached["answer"]["unique_all"], 3)
        self.assertEqual(list(self.cache.rglob("*.tmp")), [])

    def test_progress_output(self):
        out = self.result.stdout
        self.assertIn("Found 6 photos in", out)
        lines = out.splitlines()
        first = next(line for line in lines if line.startswith("[1/6] "))
        self.assertTrue(first.startswith("[1/6] ok: North gate/IMG_0001.jpg"), first)
        self.assertIn("people=1", first)
        self.assertIn("children=0", first)
        self.assertTrue(any(line.startswith("[5/6] error: Spring trail/broken.jpg") for line in lines), out)
        visit = next(line for line in lines if line.startswith(f"event {MULTI_EVENT}: "))
        self.assertIn("3 photos", visit)
        self.assertIn("people=3", visit)
        self.assertIn("Photos: 6; cached: 0; errors: 1", out)
        self.assertIn("Events: ", out)


class CacheTests(CliTestCase):
    def setUp(self):
        super().setUp()
        self.standard_photos()
        self.first = self.run_cli()
        self.fake.reset()

    def test_second_run_is_served_from_the_cache(self):
        second = self.run_cli()
        self.assertEqual(self.fake.chat_requests, [])
        self.assertEqual(self.fake.tag_requests, [])  # the server is only checked when a request is needed
        self.assertNotEqual(second.folder, self.first.folder)
        self.assertEqual(second.run["cached_photos"], 5)
        self.assertEqual(second.run["multi_photo_event_requests"], 1)  # answered from the event cache
        for row in second.images:
            self.assertEqual(row["cache_hit"], "True" if row["status"] == "ok" else "", row["relative_path"])
        ignore = {"run_id", "cache_hit"}
        for before, after in zip(self.first.images, second.images):
            self.assertEqual({k: v for k, v in before.items() if k not in ignore},
                             {k: v for k, v in after.items() if k not in ignore})
        for before, after in zip(self.first.events, second.events):
            self.assertEqual({k: v for k, v in before.items() if k != "run_id"},
                             {k: v for k, v in after.items() if k != "run_id"})
        self.assertIn("cached: North gate/IMG_0001.jpg", second.stdout)
        self.assertIn("Photos: 6; cached: 5; errors: 1", second.stdout)

    def test_force_asks_the_model_again(self):
        again = self.run_cli("--force")
        self.assertEqual(len(self.fake.photo_requests), 5)
        self.assertEqual(len(self.fake.event_requests), 1)
        self.assertEqual(again.run["cached_photos"], 0)
        self.assertTrue(all(r["cache_hit"] in ("False", "") for r in again.images))
        self.assertEqual(len(self.cache_files("photos")), 5)  # overwritten, not duplicated

    def test_photo_cache_is_keyed_by_photo_size(self):
        self.run_cli("--max-side", 1024)
        self.assertEqual(len(self.fake.photo_requests), 5)
        self.assertEqual(len(self.fake.event_requests), 0)  # event answers depend on --event-max-side only
        self.fake.reset()
        self.run_cli("--event-max-side", 640)
        self.assertEqual(len(self.fake.photo_requests), 0)
        self.assertEqual(len(self.fake.event_requests), 1)

    def test_cache_is_keyed_by_model(self):
        self.fake.models.append("qwen3-vl:8b")
        result = self.run_cli("--model", "qwen3-vl:8b")
        self.assertEqual(len(self.fake.photo_requests), 5)
        self.assertEqual(len(self.fake.event_requests), 1)
        self.assertEqual({r["body"]["model"] for r in self.fake.chat_requests}, {"qwen3-vl:8b"})
        self.assertEqual(result.run["model"], "qwen3-vl:8b")
        self.assertEqual({r["model"] for r in result.images}, {"qwen3-vl:8b"})

    def test_new_photo_is_the_only_request(self):
        make_photo(self.input / "North gate" / "IMG_0011.jpg", people=2, when=datetime(2024, 5, 3, 9, 0, 30))
        result = self.run_cli()
        self.assertEqual([p for p, *_ in (r["images"][0] for r in self.fake.photo_requests)], [2])
        # The lone 09:00 photo now has company: one new multi-photo request for that visit.
        self.assertEqual(len(self.fake.event_requests), 1)
        self.assertEqual(result.event["North_gate__2024-05-03T09-00-00"]["method"], "multi_photo_model")
        self.assertEqual(result.event[MULTI_EVENT]["method"], "multi_photo_model")

    def test_corrupt_cache_file_is_asked_again(self):
        damaged = next(p for p in self.cache_files("photos")
                       if p.name == sha256(self.input / "North gate" / "IMG_0002.jpg") + ".json")
        damaged.write_text('{"answer": ', encoding="utf-8")
        result = self.run_cli()
        self.assertEqual([p for p, *_ in (r["images"][0] for r in self.fake.photo_requests)], [2])
        self.assertEqual(result.image["North gate/IMG_0002.jpg"]["cache_hit"], "False")
        self.assertTrue(json.loads(damaged.read_text(encoding="utf-8"))["answer"])

    def test_cache_without_answer_key_is_asked_again(self):
        damaged = self.cache_files("photos")[0]
        damaged.write_text('{"info": {}}', encoding="utf-8")
        self.run_cli()
        self.assertEqual(len(self.fake.photo_requests), 1)

    # A cache file that is valid JSON but holds no answer object is treated as missing (asked again).
    def test_cache_file_with_null_answer_does_not_crash_the_run(self):
        damaged = next(p for p in self.cache_files("photos")
                       if p.name == sha256(self.input / "North gate" / "IMG_0010.jpg") + ".json")
        damaged.write_text('{"answer": null, "info": {}}', encoding="utf-8")
        result = self.run_cli()
        self.assertIn(result.code, (0, 1))
        self.assertIsNotNone(result.folder)


class EventRequestTests(CliTestCase):
    def test_no_events_flag_skips_multi_photo_requests(self):
        self.standard_photos()
        result = self.run_cli("--no-events")
        self.assertEqual(len(self.fake.event_requests), 0)
        self.assertEqual(len(self.fake.photo_requests), 5)
        visit = result.event[MULTI_EVENT]
        self.assertEqual((visit["method"], visit["people_unique"], visit["people_max_frame"]), ("per_photo_max", "", "2"))
        self.assertEqual(visit["review_reasons"], "unique_counts_not_available")
        self.assertEqual(visit["bicycles"], "1")
        self.assertIs(result.run["events_enabled"], False)
        self.assertEqual(result.run["multi_photo_event_requests"], 0)
        self.assertEqual(result.image["North gate/IMG_0002.jpg"]["event_id"], MULTI_EVENT)

    def test_event_with_one_analysable_photo_gets_no_multi_photo_request(self):
        make_photo(self.input / "cam" / "IMG_0001.jpg", people=2, when=T0)
        broken = self.input / "cam" / "IMG_0002.jpg"
        broken.write_bytes(b"\xff\xd8 not really")
        set_mtime(broken, T0 + timedelta(seconds=10))
        result = self.run_cli()
        self.assertEqual(len(self.fake.event_requests), 0)
        row = result.event["cam__2024-05-03T07-30-00"]
        self.assertEqual((row["method"], row["status"], row["image_count"]), ("per_photo_max", "ok", "2"))
        self.assertEqual(row["review_reasons"],
                         "unique_counts_not_available; photo_analysis_failed; times_from_file_dates")

    def test_long_event_is_sampled_and_flagged(self):
        for i, people in enumerate((1, 1, 4)):
            make_photo(self.input / "Long" / f"IMG_{i + 1:04d}.jpg", people=people, when=T0 + timedelta(seconds=10 * i))
        result = self.run_cli("--max-event-frames", 2)
        self.assertEqual(len(self.fake.event_requests), 1)
        self.assertEqual(len(self.fake.event_requests[0]["images"]), 2)
        self.assertEqual(self.fake.event_requests[0]["body"]["messages"][0]["content"], EVENT_PROMPT.format(n=2))
        row = result.event["Long__2024-05-03T07-30-00"]
        self.assertEqual(row["image_count"], "3")
        self.assertEqual(row["needs_review"], "True")
        self.assertIn("event_sampled_for_model", row["review_reasons"].split("; "))
        self.assertEqual(result.run["max_event_frames"], 2)

    # Long events send a selection spread over the visit that always includes the busiest photo.
    def test_split_event_counts_people_from_every_photo(self):
        for i, people in enumerate((1, 1, 4)):
            make_photo(self.input / "Long" / f"IMG_{i + 1:04d}.jpg", people=people, when=T0 + timedelta(seconds=10 * i))
        result = self.run_cli("--max-event-frames", 2)
        row = result.event["Long__2024-05-03T07-30-00"]
        self.assertGreaterEqual(int(row["people_unique"] or 0), int(row["people_max_frame"]))

    def test_failed_event_request_falls_back_to_per_photo_maxima(self):
        self.standard_photos()
        self.fake.event_reply = lambda images, body: FakeResponse({"error": "out of memory"}, status_code=500)
        result = self.run_cli()
        visit = result.event[MULTI_EVENT]
        self.assertEqual((visit["method"], visit["status"]), ("per_photo_max", "error"))
        self.assertTrue(visit["error"].startswith("event request failed: HTTPError"), visit["error"])
        self.assertEqual((visit["people_unique"], visit["people_max_frame"]), ("", "2"))
        self.assertEqual(visit["needs_review"], "True")
        self.assertEqual(result.run["multi_photo_event_requests"], 0)
        self.assertEqual(self.cache_files("events"), [])
        self.fake.event_reply = None
        self.fake.reset()
        retry = self.run_cli()
        self.assertEqual(len(self.fake.event_requests), 1)  # failures are not cached
        self.assertEqual(retry.event[MULTI_EVENT]["method"], "multi_photo_model")

    def test_unexpected_event_failure_is_reported_on_the_event(self):
        self.standard_photos()
        self.fake.event_reply = lambda images, body: RuntimeError("driver crashed")
        result = self.run_cli()
        visit = result.event[MULTI_EVENT]
        self.assertEqual(visit["status"], "error")
        self.assertEqual(visit["error"], "RuntimeError: driver crashed")


class ModelAnswerTests(CliTestCase):
    def setUp(self):
        super().setUp()
        make_photo(self.input / "cam" / "a.jpg", people=1, when=T0)
        make_photo(self.input / "cam" / "b.jpg", people=2, when=T0 + timedelta(hours=1))
        make_photo(self.input / "cam" / "c.jpg", people=3, when=T0 + timedelta(hours=2))

    def test_clean_run_exits_zero(self):
        result = self.run_cli()
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(self.fake.event_requests), 0)  # three separate visits

    def test_inconsistent_answer_is_repaired(self):
        def reply(people, bicycles, body):
            if people == 2 and len(body["messages"]) == 1:
                return photo_answer(2, adults=1)  # ages sum to 1, not 2
            return None

        self.fake.photo_reply = reply
        result = self.run_cli()
        row = result.image["cam/b.jpg"]
        self.assertEqual((row["requests"], row["repaired"], row["consistency_problems"]), ("2", "True", ""))
        self.assertEqual(row["adults"], "2")
        repair = [r for r in self.fake.photo_requests if len(r["body"]["messages"]) == 3]
        self.assertEqual(len(repair), 1)
        self.assertIn("age groups sum to 1, not people_total 2", repair[0]["body"]["messages"][2]["content"])
        self.assertEqual(result.code, 0)

    def test_answer_still_inconsistent_after_repair_is_kept_and_reported(self):
        self.fake.photo_reply = lambda people, bicycles, body: photo_answer(2, adults=1) if people == 2 else None
        result = self.run_cli()
        row = result.image["cam/b.jpg"]
        self.assertEqual(row["status"], "ok")
        self.assertEqual((row["requests"], row["repaired"]), ("2", "True"))
        self.assertEqual(row["consistency_problems"], "age groups sum to 1, not people_total 2")
        self.assertEqual(row["people_total"], "2")

    def test_unparseable_answers_become_an_error_row(self):
        self.fake.photo_reply = lambda people, bicycles, body: "Sorry, I cannot" if people == 2 else None
        result = self.run_cli()
        self.assertEqual(result.code, 1)
        row = result.image["cam/b.jpg"]
        self.assertEqual(row["status"], "error")
        self.assertTrue(row["error"].startswith("OllamaError: unparseable answer"), row["error"])
        self.assertEqual(result.image["cam/c.jpg"]["status"], "ok")
        attempts = [r for r in self.fake.photo_requests if r["images"][0][0] == 2]
        self.assertEqual([r["body"]["options"]["num_predict"] for r in attempts], [700, 1050])
        self.assertEqual(len(self.cache_files("photos")), 2)  # failures are not cached
        self.fake.photo_reply = None
        self.fake.reset()
        retry = self.run_cli()
        self.assertEqual(len(self.fake.photo_requests), 1)
        self.assertEqual(retry.code, 0)

    def test_unassessable_photo(self):
        dark = photo_answer(None, people_count_assessable=False, people_near=None, adults=None, teens=None,
                            children=None, age_unclear=None, direction_counts=None, scene_uncertain=True,
                            notes="too dark")
        self.fake.photo_reply = lambda people, bicycles, body: dark if people == 2 else None
        result = self.run_cli()
        row = result.image["cam/b.jpg"]
        self.assertEqual((row["status"], row["people_count_assessable"], row["people_total"], row["people_class"]),
                         ("ok", "False", "", ""))
        self.assertEqual((row["bicycles"], row["notes"]), ("0", "too dark"))
        self.assertEqual(result.event["cam__2024-05-03T08-30-00"]["people_unique"], "")

    def test_server_error_for_one_photo(self):
        self.fake.photo_reply = lambda people, bicycles, body: (FakeResponse({"error": "boom"}, status_code=500)
                                                                if people == 1 else None)
        result = self.run_cli()
        self.assertEqual(result.code, 1)
        self.assertTrue(result.image["cam/a.jpg"]["error"].startswith("OllamaError: HTTPError"))
        self.assertEqual(result.image["cam/b.jpg"]["status"], "ok")

    def test_interrupted_run_keeps_finished_answers(self):
        calls = []

        def reply(people, bicycles, body):
            calls.append(people)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return None

        self.fake.photo_reply = reply
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli()
        self.assertEqual(len(self.cache_files("photos")), 1)  # saved right after it was answered
        self.fake.photo_reply = None
        self.fake.reset()
        self.run_cli()
        self.assertEqual(sorted(r["images"][0][0] for r in self.fake.photo_requests), [2, 3])

    def test_limit_analyses_the_first_photos_only(self):
        result = self.run_cli("--limit", 2)
        self.assertEqual([r["relative_path"] for r in result.images], ["cam/a.jpg", "cam/b.jpg"])
        self.assertEqual(len(self.fake.photo_requests), 2)
        self.assertEqual(result.run["photos"], 2)
        self.assertIn("Found 2 photos", result.stdout)

    def test_limit_zero_means_all(self):
        result = self.run_cli("--limit", 0)
        self.assertEqual(len(result.images), 3)

    def test_host_timeout_and_model_options_reach_the_client(self):
        self.fake.models = ["gemma4:latest"]
        result = self.run_cli("--model", "gemma4", "--ollama-host", "http://gpu-box:1234/", "--timeout", 30)
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual(self.fake.tag_requests[0]["url"], "http://gpu-box:1234/api/tags")
        for request in self.fake.chat_requests:
            self.assertEqual(request["url"], "http://gpu-box:1234/api/chat")
            self.assertEqual(request["timeout"], 30.0)
            self.assertEqual(request["body"]["model"], "gemma4")
        self.assertEqual(result.run["ollama_host"], "http://gpu-box:1234/")


class ServerAvailabilityTests(CliTestCase):
    def setUp(self):
        super().setUp()
        make_photo(self.input / "cam" / "a.jpg", people=1, when=T0)

    def test_unavailable_model_exits_2_with_a_clear_message(self):
        self.fake.models = ["llava:7b", "gemma4:e4b"]
        result = self.run_cli()
        self.assertEqual(result.code, 2)
        self.assertIn("Error: Ollama is running at http://localhost:11434 but model gemma4:26b is not installed",
                      result.stderr)
        self.assertIn("ollama pull gemma4:26b", result.stderr)
        self.assertEqual(self.fake.chat_requests, [])
        self.assertIsNone(result.folder)
        self.assertEqual(list(self.output.iterdir()), [])  # the output folder is checked first, but stays empty

    def test_server_not_running_exits_2(self):
        self.fake.tags_error = requests.ConnectionError("connection refused")
        result = self.run_cli()
        self.assertEqual(result.code, 2)
        self.assertIn("no Ollama server answers at http://localhost:11434", result.stderr)
        self.assertIsNone(result.folder)

    def test_bare_model_name_matches_the_latest_tag(self):
        self.fake.models = ["gemma4:latest"]
        self.assertEqual(self.run_cli("--model", "gemma4").code, 0)

    def test_empty_folder_needs_no_server(self):
        (self.input / "cam" / "a.jpg").unlink()
        self.fake.tags_error = requests.ConnectionError("connection refused")
        result = self.run_cli()
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual(self.fake.tag_requests, [])
        self.assertEqual(result.images, [])
        self.assertEqual(result.events, [])
        self.assertEqual(result.run["photos"], 0)

    def test_missing_input_folder_exits_2(self):
        result = self.run_cli("--input", self.tmp / "nowhere", paths=False)
        self.assertEqual(result.code, 2)
        self.assertIn("Input folder does not exist", result.stderr)
        self.assertEqual(self.fake.tag_requests, [])


class FolderTests(CliTestCase):
    def test_positional_input_folder(self):
        make_photo(self.input / "a.jpg", people=1, when=T0)
        result = self.run_cli(self.input, "--output", self.output, "--cache", self.cache, paths=False)
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual([r["relative_path"] for r in result.images], ["a.jpg"])
        self.assertEqual(result.images[0]["camera_id"], ".")
        self.assertEqual(result.images[0]["event_id"], "root__2024-05-03T07-30-00")

    def test_output_and_cache_inside_the_input_are_not_analysed(self):
        make_photo(self.input / "cam" / "a.jpg", people=1, when=T0)
        make_photo(self.input / "exports" / "old" / "contact_sheet.jpg", people=5)
        make_photo(self.input / "answers" / "x.jpg", people=5)
        make_photo(self.input / ".hidden" / "y.jpg", people=5)
        self.output, self.cache = self.input / "exports", self.input / "answers"
        result = self.run_cli()
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual([r["relative_path"] for r in result.images], ["cam/a.jpg"])

    def test_non_recursive(self):
        make_photo(self.input / "top.jpg", people=1, when=T0)
        make_photo(self.input / "cam" / "a.jpg", people=1, when=T0)
        result = self.run_cli("--no-recursive")
        self.assertEqual([r["relative_path"] for r in result.images], ["top.jpg"])

    def test_identical_files_share_one_answer(self):
        make_photo(self.input / "north" / "IMG_0001.jpg", people=2, when=T0)
        (self.input / "south").mkdir()
        shutil.copyfile(self.input / "north" / "IMG_0001.jpg", self.input / "south" / "IMG_0001.jpg")
        result = self.run_cli()
        self.assertEqual(len(self.fake.photo_requests), 1)
        north, south = result.image["north/IMG_0001.jpg"], result.image["south/IMG_0001.jpg"]
        self.assertEqual(north["image_id"], south["image_id"])
        self.assertEqual((north["cache_hit"], south["cache_hit"]), ("False", "True"))
        self.assertEqual((north["camera_id"], south["camera_id"]), ("north", "south"))
        self.assertEqual(len(result.events), 2)  # different cameras are never one visit

    def test_regex_camera_ids(self):
        folder = self.input / "People_100"
        make_photo(folder / "001_Cam-A_IMG1.jpg", people=1, when=T0)
        make_photo(folder / "002_Cam-D_IMG1.jpg", people=2, when=T0)
        pattern = r"^People_100/\d{3}_(?P<camera>Cam-A|Cam_B|Cam-C|Cam-D|CamE)_"
        result = self.run_cli("--camera-id", "regex", "--camera-pattern", pattern)
        self.assertEqual([r["camera_id"] for r in result.images], ["Cam-A", "Cam-D"])
        self.assertEqual([r["event_id"] for r in result.events],
                         ["Cam-A__2024-05-03T07-30-00", "Cam-D__2024-05-03T07-30-00"])
        self.assertEqual(len(self.fake.event_requests), 0)
        self.assertEqual((result.run["camera_id_mode"], result.run["camera_id_pattern"]), ("regex", pattern))
        self.fake.reset()
        folder_mode = self.run_cli()
        self.assertEqual([r["event_id"] for r in folder_mode.events], ["People_100__2024-05-03T07-30-00"])
        self.assertEqual(len(self.fake.event_requests), 1)

    def test_single_camera_mode(self):
        make_photo(self.input / "a" / "1.jpg", people=1, when=T0)
        make_photo(self.input / "b" / "2.jpg", people=1, when=T0 + timedelta(seconds=5))
        result = self.run_cli("--camera-id", "single")
        self.assertEqual({r["camera_id"] for r in result.images}, {"all"})
        self.assertEqual([r["event_id"] for r in result.events], ["all__2024-05-03T07-30-00"])

    def test_event_gap_option(self):
        make_photo(self.input / "cam" / "1.jpg", people=1, when=T0)
        make_photo(self.input / "cam" / "2.jpg", people=1, when=T0 + timedelta(seconds=90))
        self.assertEqual(len(self.run_cli().events), 2)
        self.assertEqual(len(self.run_cli("--event-gap", 120).events), 1)

    def test_unicode_paths(self):
        make_photo(self.input / "מצלמה צפון" / "תמונה 1.jpg", people=1, when=T0)
        make_photo(self.input / "מצלמה צפון" / "תמונה 2.jpg", people=2, when=T0 + timedelta(seconds=10))
        result = self.run_cli()
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual([r["relative_path"] for r in result.images],
                         ["מצלמה צפון/תמונה 1.jpg", "מצלמה צפון/תמונה 2.jpg"])
        event = result.events[0]
        self.assertEqual(event["event_id"], "מצלמה_צפון__2024-05-03T07-30-00")
        self.assertEqual(json.loads(event["images"]), ["מצלמה צפון/תמונה 1.jpg", "מצלמה צפון/תמונה 2.jpg"])
        self.assertIn("מצלמה צפון/תמונה 1.jpg", result.stdout)

    def test_formula_like_file_names_are_neutralised(self):
        make_photo(self.input / "=cmd" / "@IMG.jpg", people=1, when=T0)
        result = self.run_cli()
        self.assertEqual(result.images[0]["relative_path"], "'=cmd/@IMG.jpg")
        self.assertEqual(result.images[0]["camera_id"], "'=cmd")

    def test_rotated_photo_is_sent_upright(self):
        make_photo(self.input / "cam" / "a.jpg", people=60, bicycles=0, when=T0, orientation=6)
        self.fake.photo_reply = lambda people, bicycles, body: photo_answer(0)
        self.run_cli()
        size = self.fake.photo_requests[0]["images"][0][2]
        self.assertEqual(size, (200, 360))

    def test_large_photo_is_downscaled_for_the_model(self):
        path = self.input / "big.jpg"
        Image.new("RGB", (5000, 2500), (1, 2, 3)).save(path)
        self.fake.photo_reply = lambda people, bicycles, body: photo_answer(0)
        self.run_cli("--max-side", 1000)
        self.assertEqual(self.fake.photo_requests[0]["images"][0][2], (1000, 500))


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trailcam_vlm_args_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        patcher = mock.patch.object(cli, "ROOT", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def parse(self, *argv):
        return cli.arguments([str(a) for a in argv])

    def rejected(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            self.parse(*argv)
        self.assertEqual(caught.exception.code, 2)
        return err.getvalue()

    def write_settings(self, text):
        (self.tmp / "settings.json").write_text(text, encoding="utf-8-sig")

    def test_defaults_without_a_settings_file(self):
        args = self.parse()
        self.assertEqual(args.input, (self.tmp / "images").resolve())
        self.assertEqual(args.output, (self.tmp / "exports").resolve())
        self.assertEqual(args.cache, (self.tmp / ".cache").resolve())
        self.assertEqual((args.model, args.ollama_host), ("gemma4:26b", "http://localhost:11434"))
        self.assertEqual((args.max_side, args.event_max_side, args.max_event_frames, args.event_gap),
                         (2048, 1280, 6, 60))
        self.assertEqual((args.camera_id_mode, args.camera_pattern), ("folder", None))
        self.assertEqual((args.events, args.recursive, args.force, args.limit, args.timeout),
                         (True, True, False, 0, 600))

    def test_settings_file_overrides_defaults_and_flags_override_settings(self):
        self.write_settings(json.dumps({"model": "qwen3-vl:8b", "max_side": 1024, "events": False,
                                        "input_dir": "pics", "event_gap_seconds": 30, "timeout": 90}))
        args = self.parse()
        self.assertEqual((args.model, args.max_side, args.events, args.event_gap, args.timeout),
                         ("qwen3-vl:8b", 1024, False, 30, 90))
        self.assertEqual(args.input, (self.tmp / "pics").resolve())
        args = self.parse("--model", "gemma4:26b", "--events", "--max-side", 512)
        self.assertEqual((args.model, args.events, args.max_side), ("gemma4:26b", True, 512))

    def test_settings_must_be_a_json_object(self):
        self.write_settings("[1, 2]")
        with self.assertRaises(ValueError):
            self.parse()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main([]), 2)
        self.assertIn("must contain a JSON object", err.getvalue())

    def test_invalid_settings_json_exits_2(self):
        self.write_settings("{ not json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main([]), 2)
        self.assertTrue(err.getvalue().startswith("Error: "))

    def test_repository_settings_file_matches_the_defaults(self):
        settings = json.loads((fakes_vlm.REPO / "settings.json").read_text(encoding="utf-8-sig"))
        self.assertIsInstance(settings, dict)
        self.assertEqual(set(settings) - set(cli.DEFAULTS), set())  # unknown keys would be ignored silently

    def test_numeric_settings_are_checked_like_flags(self):
        self.write_settings(json.dumps({"max_side": 100}))
        self.assertIn("--max-side must be between 256 and 4096", self.rejected())
        self.write_settings(json.dumps({"max_event_frames": "6"}))  # argparse converts string defaults
        self.assertEqual(self.parse().max_event_frames, 6)
        self.write_settings(json.dumps({"camera_id_mode": "regex", "camera_id_pattern": None}))
        self.assertIn("needs a named group", self.rejected())

    def _parse_or_reject(self):
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                return self.parse()
        except (SystemExit, ValueError):
            return None  # rejecting the setting is an acceptable fix

    # argparse does not check defaults, so settings.json switches and choices are validated separately
    # ("events": "false" must not switch events on; an unknown camera_id_mode must not act like "folder").
    def test_switch_and_choice_settings_are_validated(self):
        self.write_settings(json.dumps({"events": "false"}))
        args = self._parse_or_reject()
        if args is not None:
            self.assertIs(args.events, False)
        self.write_settings(json.dumps({"camera_id_mode": "Regex"}))
        args = self._parse_or_reject()
        if args is not None:
            self.assertIn(args.camera_id_mode, ("folder", "regex", "single"))

    def test_relative_paths_resolve_against_the_repository(self):
        args = self.parse("--input", "photos", "--output", "out", "--cache", "c")
        self.assertEqual((args.input, args.output, args.cache),
                         ((self.tmp / "photos").resolve(), (self.tmp / "out").resolve(), (self.tmp / "c").resolve()))
        elsewhere = self.tmp / "abs"
        self.assertEqual(self.parse("--input", elsewhere).input, elsewhere.resolve())

    def test_positional_folder(self):
        self.assertEqual(self.parse(self.tmp / "x").input, (self.tmp / "x").resolve())

    def test_positional_folder_and_input_conflict(self):
        message = self.rejected(self.tmp / "x", "--input", self.tmp / "y")
        self.assertIn("Use either a positional input folder or --input, not both.", message)

    def test_photo_size_limits(self):
        for flag in ("--max-side", "--event-max-side"):
            for value in (255, 4097, 0, -1):
                with self.subTest(flag=flag, value=value):
                    self.assertIn(f"{flag} must be between 256 and 4096", self.rejected(flag, value))
            for value in (256, 4096):
                self.assertEqual(getattr(self.parse(flag, value), flag[2:].replace("-", "_")), value)
        self.assertIn("invalid int value", self.rejected("--max-side", "big"))

    def test_event_frame_limits(self):
        for value in (1, 13, 0):
            self.assertIn("--max-event-frames must be between 2 and 12", self.rejected("--max-event-frames", value))
        for value in (2, 12):
            self.assertEqual(self.parse("--max-event-frames", value).max_event_frames, value)

    def test_event_gap_must_be_finite_and_positive(self):
        for value in ("0", "-5", "nan", "inf"):
            with self.subTest(value=value):
                self.assertIn("--event-gap must be a finite positive number", self.rejected("--event-gap", value))
        self.assertEqual(self.parse("--event-gap", "0.5").event_gap, 0.5)

    def test_limit_and_timeout(self):
        self.assertIn("--limit must be >= 0", self.rejected("--limit", -1))
        for value in (0, -3, "inf", "nan"):
            self.assertIn("--timeout must be a finite positive number", self.rejected("--timeout", value))
        args = self.parse("--limit", 5, "--timeout", 2.5)
        self.assertEqual((args.limit, args.timeout), (5, 2.5))

    def test_camera_modes(self):
        self.assertIn("invalid choice", self.rejected("--camera-id", "gps"))
        self.assertEqual(self.parse("--camera-id", "single").camera_id_mode, "single")

    def test_regex_mode_needs_a_named_camera_group(self):
        self.assertIn("needs a named group (?P<camera>...)", self.rejected("--camera-id", "regex"))
        self.assertIn("needs a named group", self.rejected("--camera-id", "regex", "--camera-pattern", r"CAM\d+"))
        self.assertIn("Invalid --camera-pattern", self.rejected("--camera-id", "regex", "--camera-pattern", "(?P<camera>"))
        args = self.parse("--camera-id", "regex", "--camera-pattern", r"(?P<camera>CAM\d+)")
        self.assertEqual((args.camera_id_mode, args.camera_pattern), ("regex", r"(?P<camera>CAM\d+)"))

    def test_pattern_is_only_checked_in_regex_mode(self):
        self.assertEqual(self.parse("--camera-pattern", "(").camera_pattern, "(")

    def test_switches(self):
        args = self.parse("--no-events", "--no-recursive", "--force")
        self.assertEqual((args.events, args.recursive, args.force), (False, False, True))

    def test_version(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            self.parse("--version")
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), __version__)
        self.assertEqual(__version__, "0.1.0")

    def test_main_returns_2_for_argument_errors_via_system_exit(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            cli.main(["--limit", "-1"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
