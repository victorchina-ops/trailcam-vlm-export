"""Long events, offline re-export, a server that stops answering, malformed answers and settings checks."""
from __future__ import annotations

import json
import unittest
from datetime import timedelta

import requests
from fakes_vlm import FakeResponse, event_answer, make_photo, photo_answer
from test_export import RUN, event, item, three_photos
from test_main import MULTI_EVENT, T0, CliTestCase

from trailcam_vlm import __main__ as cli
from trailcam_vlm.export import event_row, image_row, summary_rows
from trailcam_vlm.prompts import PHOTO_SCHEMA, clean_answer


def photos_with(people):
    return [{"answer": {"people_total": p}, "n": k} for k, p in enumerate(people)]


class SelectFramesTests(unittest.TestCase):
    def test_short_events_are_sent_whole(self):
        frames = photos_with([1, 2, 3])
        self.assertEqual(cli.select_frames(frames, 6), frames)

    def test_spread_includes_first_last_and_busiest_in_time_order(self):
        frames = photos_with([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1])
        chosen = [f["n"] for f in cli.select_frames(frames, 4)]
        self.assertEqual(len(chosen), 4)
        self.assertEqual(chosen, sorted(chosen))
        self.assertIn(0, chosen)
        self.assertIn(11, chosen)
        self.assertIn(10, chosen)

    def test_busiest_first_or_last_photo_needs_no_swap(self):
        frames = photos_with([7, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([f["n"] for f in cli.select_frames(frames, 3)], [0, 4, 7])

    def test_two_frames_keep_first_and_last_unless_the_busiest_is_inside(self):
        frames = photos_with([1, 1, 4])
        self.assertEqual([f["n"] for f in cli.select_frames(frames, 2)], [0, 2])
        frames = photos_with([1, 4, 1])
        chosen = [f["n"] for f in cli.select_frames(frames, 2)]
        self.assertIn(1, chosen)
        self.assertEqual(len(chosen), 2)


class LongEventTests(CliTestCase):
    def test_long_event_sends_a_spread_with_the_busiest_photo(self):
        counts = (1, 1, 1, 5, 1, 1, 1, 1)
        for i, people in enumerate(counts):
            make_photo(self.input / "Long" / f"IMG_{i + 1:04d}.jpg", people=people, when=T0 + timedelta(seconds=5 * i))
        result = self.run_cli("--max-event-frames", 3)
        sent = [people for people, *_ in self.fake.event_requests[0]["images"]]
        self.assertEqual(len(sent), 3)
        self.assertIn(5, sent)
        row = result.event["Long__2024-05-03T07-30-00"]
        self.assertGreaterEqual(int(row["people_unique"]), int(row["people_max_frame"]))
        self.assertIn("event_sampled_for_model", row["review_reasons"])


class OfflineAndFailureTests(CliTestCase):
    def test_fully_cached_rerun_needs_no_server(self):
        self.standard_photos()
        self.run_cli()
        self.fake.reset()
        self.fake.tags_error = requests.ConnectionError("Ollama is not running")
        result = self.run_cli()
        self.assertEqual(result.code, 1)  # the corrupt photo is still an error
        self.assertEqual((self.fake.tag_requests, self.fake.chat_requests), ([], []))
        self.assertEqual(result.event[MULTI_EVENT]["method"], "multi_photo_model")

    def test_failed_event_request_sets_exit_code_and_run_json(self):
        make_photo(self.input / "cam" / "a.jpg", people=1, when=T0)
        make_photo(self.input / "cam" / "b.jpg", people=2, when=T0 + timedelta(seconds=10))
        self.fake.event_reply = lambda images, body: FakeResponse({"error": "out of memory"}, status_code=400)
        result = self.run_cli()
        self.assertEqual(result.code, 1)
        self.assertEqual(result.run["event_requests_failed"], 1)
        self.assertIn("failed event requests: 1", result.stdout)

    def test_server_that_stops_answering_ends_model_requests(self):
        for i in range(6):
            make_photo(self.input / "cam" / f"IMG_{i:04d}.jpg", people=1, when=T0 + timedelta(minutes=5 * i))
        calls = []

        def dead(people, bicycles, body):
            calls.append(1)
            self.fake.tags_error = requests.ConnectionError("refused")
            return requests.ConnectionError("refused")
        self.fake.photo_reply = dead
        result = self.run_cli()
        self.assertEqual(result.code, 1)
        self.assertEqual(len(calls), 2 * cli.CONSECUTIVE_FAILURES)  # each failure is retried once, then it stops
        errors = [row["error"] for row in result.images]
        self.assertTrue(all(e for e in errors))
        self.assertIn("Ollama stopped answering", errors[-1])
        self.assertTrue(result.run["stopped_early"])

    def test_cache_file_that_is_not_an_answer_does_not_crash_the_run(self):
        make_photo(self.input / "cam" / "a.jpg", people=2, when=T0)
        self.run_cli()
        cache = self.cache_files("photos")[0]
        for damaged in ('{"answer": [1, 2], "info": {}}', '{"answer": {"people_total": "many"}, "info": null}'):
            with self.subTest(damaged=damaged):
                cache.write_text(damaged, encoding="utf-8")
                self.fake.reset()
                result = self.run_cli()
                self.assertIn(result.code, (0, 1))
                self.assertIsNotNone(result.folder)

    def test_output_folder_equal_to_the_input_folder_still_finds_photos(self):
        make_photo(self.input / "cam" / "a.jpg", people=1, when=T0)
        result = self.run_cli("--output", self.input)
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual(len(self.fake.photo_requests), 1)


class AnswerCleaningTests(unittest.TestCase):
    def test_types_are_checked(self):
        raw = {"people_total": "3", "people_near": -1, "adults": 2.0, "teens": True, "children": None,
               "direction_counts": {"left": 1, "right": "x"}, "scene_uncertain": "yes", "notes": None,
               "bicycles": 1.5, "unexpected": 7}
        clean = clean_answer(raw, PHOTO_SCHEMA)
        self.assertEqual((clean["people_total"], clean["people_near"], clean["adults"], clean["teens"]), (3, None, 2, None))
        self.assertIsNone(clean["direction_counts"])  # "right": "x" is invalid, so no direction is exported
        partial = clean_answer({"direction_counts": {"left": 1}}, PHOTO_SCHEMA)["direction_counts"]
        self.assertEqual(partial, {"left": 1, "right": 0, "toward": 0, "away": 0, "stationary": 0, "unclear": 0})
        self.assertIsNone(clean["scene_uncertain"])
        self.assertEqual(clean["notes"], "")
        self.assertIsNone(clean["bicycles"])
        self.assertNotIn("unexpected", clean)
        self.assertIsNone(clean_answer(None, PHOTO_SCHEMA))
        self.assertIsNone(clean_answer([1], PHOTO_SCHEMA))

    def test_not_assessable_photo_exports_blank_people_columns(self):
        answer = photo_answer(2, people_count_assessable=False)
        row = image_row(item(answer=answer), RUN)
        self.assertIs(row["people_count_assessable"], False)
        for key in ("people_total", "people_near", "adults", "children", "people_class", "dir_left"):
            self.assertIsNone(row[key], key)
        self.assertEqual(row["bicycles"], 0)


class EventReviewTests(unittest.TestCase):
    def test_single_photo_event_keeps_unresolved_problems(self):
        row = event_row(event([item(info={"problems": ["age groups sum to 3, not people_total 2"]})]), RUN)
        self.assertEqual(row["method"], "single_photo")
        self.assertEqual(row["review_reasons"], "inconsistent_model_answer")
        self.assertIn("age groups", row["consistency_problems"])

    def test_unique_far_above_the_busiest_photo_needs_review(self):
        row = event_row(event(three_photos(1, 1), event_answer(5, 1)), RUN)
        self.assertIn("unique_far_above_busiest_photo", row["review_reasons"])
        row = event_row(event(three_photos(1, 1), event_answer(2, 1)), RUN)
        self.assertIsNone(row["review_reasons"])

    def test_busiest_photo_is_a_floor_for_the_visit(self):
        row = event_row(event(three_photos(1, 3), event_answer(2, 3)), RUN)
        self.assertEqual(row["people_unique"], 3)
        self.assertIn("unique_below_busiest_photo", row["review_reasons"])
        self.assertEqual(sum(row[k] for k in ("adults", "teens", "children", "age_unknown")), 3)
        self.assertEqual(sum(row[f"dir_{d}"] for d in ("left", "right", "toward", "away", "stationary", "unclear")), 3)
        self.assertLessEqual(row["people_near_unique"], row["people_unique"])

    def test_camera_days_count_events_without_unique_counts(self):
        rows = [event_row(event(three_photos(1, 2)), RUN), event_row(event([item()], event_id="cam__x"), RUN)]
        day = summary_rows(rows, RUN)[0]
        self.assertEqual(day["events_without_unique_count"], 1)

    def test_long_image_lists_are_truncated_for_excel(self):
        photos = [item(f"cam/{'x' * 200}_{k:04d}.jpg") for k in range(400)]
        cell = event_row(event(photos), RUN)["images"]
        self.assertLessEqual(len(cell), 32000)
        self.assertIn("see the images CSV", json.loads(cell)[-1])


class SettingsTests(unittest.TestCase):
    def test_bad_settings_are_rejected_with_the_key_name(self):
        cases = {"events": "false", "recursive": 1, "camera_id_mode": "Regex", "timeout": None, "max_side": True,
                 "model": 26, "colour": "red"}
        for key, value in cases.items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    cli.check_settings({key: value}, "settings.json")
                self.assertIn(key, str(caught.exception))

    def test_good_settings_and_comments_pass(self):
        cli.check_settings({**cli.DEFAULTS, "camera_id_pattern": "(?P<camera>x)", "event_gap_seconds": 30.5,
                            "max_side": "1024", "_comment": "anything"}, "settings.json")


if __name__ == "__main__":
    unittest.main()
