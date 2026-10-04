"""CSV rows (images, events, camera-days), count classes and safe CSV writing."""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import fakes_vlm  # noqa: F401  (puts the repository on sys.path)
from fakes_vlm import event_answer, photo_answer

from trailcam_vlm import export
from trailcam_vlm.export import (COUNT_CLASSES, EVENT_FIELDS, IMAGE_FIELDS, OBJECT_FIELDS, SUMMARY_FIELDS,
                                 count_class, event_row, image_row, spreadsheet_safe, summary_rows, write_csv)
from trailcam_vlm.prompts import DIRECTIONS, OBJECTS, PROMPT_VERSION

RUN = "2026-10-04_10-00-00_000000"
DIR_FIELDS = [f"dir_{d}" for d in DIRECTIONS]

# Column names shared with trailcam-expert-export v2 (its images / events / camera-days CSVs).
SIBLING_IMAGE_COLUMNS = ["run_id", "image_id", "relative_path", "sha256", "camera_id", "capture_time", "time_source",
                         "event_id", "event_frame", "event_image_count", "status", "error", "cache_hit",
                         "people_total", "people_near", "people_class", "people_near_class", "adults", "children",
                         "age_unknown", *DIR_FIELDS, "bicycles", "strollers", "motorcycles", "atv_utv",
                         "other_vehicles", "dogs", "backpacks", "large_bags", "large_bags_uncertain", "notes"]
SIBLING_EVENT_COLUMNS = ["run_id", "event_id", "camera_id", "start", "end", "duration_seconds", "image_count",
                         "people_unique", "people_max_frame", "people_near_max_frame", "people_near_unique",
                         "people_unique_class", "people_near_unique_class", "adults", "children", "age_unknown",
                         *DIR_FIELDS, "bicycles", "strollers", "motorcycles", "atv_utv", "other_vehicles", "dogs",
                         "backpacks", "large_bags", "large_bags_uncertain", "needs_review", "review_reasons",
                         "images", "notes"]
SIBLING_SUMMARY_COLUMNS = ["run_id", "camera_id", "date", "events", "images", "people_max_frame", "people_unique",
                           "people_near_max_frame", "people_near_unique", "adults", "children", "age_unknown",
                           *DIR_FIELDS, "bicycles", "strollers", "motorcycles", "atv_utv", "other_vehicles", "dogs",
                           "backpacks", "large_bags", "events_needing_review"]


def item(relative="cam/IMG_0001.jpg", status="ok", answer=None, info=None, **extra):
    """An analysed photo as ``__main__.run`` builds it."""
    base = {"path": Path("/photos") / relative, "relative_path": relative, "sha256": "ab" * 32,
            "image_id": "img_" + "ab" * 8, "camera_id": "cam", "capture_time": "2024-05-03T07:30:00",
            "time_source": "exif_original", "sequence_number": 1, "model": "gemma4:26b",
            "prompt_version": PROMPT_VERSION, "status": status, "event_id": "cam__2024-05-03T07-30-00",
            "event_frame": 1, "event_image_count": 1, "cache_hit": False}
    if status == "ok":
        base["answer"] = photo_answer(1)
        base["info"] = {"requests": 1, "seconds": 2.5, "repaired": False, "problems": []}
    if answer is not None:
        base["answer"] = answer
    if info is not None:
        base["info"] = info
    base.update(extra)
    return base


def failed(relative="cam/broken.jpg", error="UnidentifiedImageError: cannot identify image file", **extra):
    return item(relative, status="error", error=error, **extra)


class FieldListTests(unittest.TestCase):
    def test_no_duplicate_columns(self):
        for fields in (IMAGE_FIELDS, EVENT_FIELDS, SUMMARY_FIELDS):
            self.assertEqual(len(fields), len(set(fields)))

    def test_columns_shared_with_trailcam_expert_export(self):
        for ours, sibling in ((IMAGE_FIELDS, SIBLING_IMAGE_COLUMNS), (EVENT_FIELDS, SIBLING_EVENT_COLUMNS),
                              (SUMMARY_FIELDS, SIBLING_SUMMARY_COLUMNS)):
            self.assertEqual([c for c in sibling if c not in ours], [])
            # Shared columns keep the sibling's relative order.
            self.assertEqual([c for c in ours if c in sibling], sibling)

    def test_summary_sums_event_columns(self):
        for key in SUMMARY_FIELDS[5:-2]:
            self.assertIn(key, EVENT_FIELDS)
        self.assertEqual(SUMMARY_FIELDS[:5], ["run_id", "camera_id", "date", "events", "images"])
        self.assertEqual(SUMMARY_FIELDS[-2:], ["events_without_unique_count", "events_needing_review"])

    def test_object_columns(self):
        self.assertEqual(tuple(OBJECT_FIELDS), tuple(OBJECTS))
        for fields in (IMAGE_FIELDS, EVENT_FIELDS, SUMMARY_FIELDS):
            for key in (*OBJECT_FIELDS, *DIR_FIELDS):
                self.assertIn(key, fields)


class CountClassTests(unittest.TestCase):
    def test_classes(self):
        expected = {0: "0", 1: "1", 2: "2", 3: "3–4", 4: "3–4", 5: "5–10", 7: "5–10",
                    10: "5–10", 11: ">10", 250: ">10"}
        for value, label in expected.items():
            with self.subTest(value=value):
                self.assertEqual(count_class(value), label)

    def test_range_labels_use_an_en_dash(self):
        self.assertEqual(count_class(3), "3–4")
        self.assertEqual(count_class(5), "5–10")
        for _, _, label in COUNT_CLASSES:
            self.assertNotIn("-", label)  # "3-4" would become a date in Excel

    def test_unknown_count_has_no_class(self):
        self.assertIsNone(count_class(None))

    def test_negative_count_has_no_class(self):
        self.assertIsNone(count_class(-1))

    def test_classes_cover_every_count_without_gaps(self):
        previous_high = -1
        for low, high, _ in COUNT_CLASSES:
            self.assertEqual(low, previous_high + 1)
            previous_high = high if high is not None else float("inf")
        self.assertEqual(previous_high, float("inf"))


class ImageRowTests(unittest.TestCase):
    def test_columns_are_exactly_the_image_fields(self):
        self.assertEqual(list(image_row(item(), RUN)), IMAGE_FIELDS)
        self.assertEqual(list(image_row(failed(), RUN)), IMAGE_FIELDS)

    def test_analysed_photo(self):
        answer = photo_answer(3, people_near=2, adults=1, teens=1, children=1, age_unclear=0,
                              direction_counts={"left": 1, "right": 0, "toward": 2, "away": 0, "stationary": 0,
                                                "unclear": 0},
                              bicycles=1, dogs=2, large_bags_uncertain=1, scene_uncertain=True,
                              notes="one person partly hidden")
        info = {"requests": 2, "seconds": 7.25, "repaired": True, "problems": ["a", "b"]}
        row = image_row(item(answer=answer, info=info, cache_hit=True), RUN)
        self.assertEqual(row["run_id"], RUN)
        self.assertEqual(row["image_id"], "img_" + "ab" * 8)
        self.assertEqual(row["relative_path"], "cam/IMG_0001.jpg")
        self.assertEqual(row["sha256"], "ab" * 32)
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["error"])
        self.assertIs(row["cache_hit"], True)
        self.assertEqual((row["model"], row["prompt_version"]), ("gemma4:26b", PROMPT_VERSION))
        self.assertEqual((row["seconds"], row["requests"], row["repaired"]), (7.25, 2, True))
        self.assertEqual(row["consistency_problems"], "a; b")
        self.assertIs(row["people_count_assessable"], True)
        self.assertEqual((row["people_total"], row["people_near"]), (3, 2))
        self.assertEqual((row["people_class"], row["people_near_class"]), ("3–4", "2"))
        self.assertEqual((row["adults"], row["teens"], row["children"], row["age_unknown"]), (1, 1, 1, 0))
        self.assertNotIn("age_unclear", row)  # the model's age_unclear is exported as age_unknown
        self.assertEqual([row[f] for f in DIR_FIELDS], [1, 0, 2, 0, 0, 0])
        self.assertEqual((row["bicycles"], row["dogs"], row["large_bags_uncertain"], row["strollers"]), (1, 2, 1, 0))
        self.assertIs(row["scene_uncertain"], True)
        self.assertEqual(row["notes"], "one person partly hidden")

    def test_event_and_capture_columns_are_copied(self):
        row = image_row(item(camera_id="North gate", capture_time="2024-05-03T07:30:00", time_source="file_mtime",
                             event_id="North_gate__x", event_frame=2, event_image_count=3), RUN)
        self.assertEqual((row["camera_id"], row["capture_time"], row["time_source"]),
                         ("North gate", "2024-05-03T07:30:00", "file_mtime"))
        self.assertEqual((row["event_id"], row["event_frame"], row["event_image_count"]), ("North_gate__x", 2, 3))

    def test_no_problems_is_blank(self):
        self.assertIsNone(image_row(item(), RUN)["consistency_problems"])
        self.assertIsNone(image_row(item(info={"problems": None}), RUN)["consistency_problems"])

    def test_teens_are_a_category_of_their_own(self):
        row = image_row(item(answer=photo_answer(5, adults=1, teens=2, children=0, age_unclear=2)), RUN)
        self.assertEqual((row["adults"], row["teens"], row["children"], row["age_unknown"]), (1, 2, 0, 2))
        self.assertEqual(sum(row[k] for k in ("adults", "teens", "children", "age_unknown")), row["people_total"])
        row = image_row(item(answer=photo_answer(1, teens=None)), RUN)
        self.assertIsNone(row["teens"])
        self.assertEqual(row["age_unknown"], 0)  # a missing teens answer no longer blanks age_unknown
        row = image_row(item(answer=photo_answer(1, age_unclear=None)), RUN)
        self.assertIsNone(row["age_unknown"])

    def test_zero_is_kept_and_differs_from_blank(self):
        row = image_row(item(answer=photo_answer(0)), RUN)
        self.assertEqual((row["people_total"], row["people_class"], row["age_unknown"], row["dir_left"]),
                         (0, "0", 0, 0))
        self.assertEqual(row["bicycles"], 0)

    def test_photo_that_could_not_be_assessed(self):
        answer = photo_answer(None, people_count_assessable=False, people_near=None, adults=None, teens=None,
                              children=None, age_unclear=None, direction_counts=None, scene_uncertain=True,
                              notes="too dark")
        row = image_row(item(answer=answer), RUN)
        self.assertIs(row["people_count_assessable"], False)
        for key in ("people_total", "people_near", "people_class", "people_near_class", "adults", "teens",
                    "children", "age_unknown", *DIR_FIELDS):
            with self.subTest(key=key):
                self.assertIsNone(row[key])
        self.assertEqual(row["bicycles"], 0)  # objects were still checked

    def test_missing_answer_keys_are_blank(self):
        row = image_row(item(answer={"people_total": 2}), RUN)
        self.assertEqual((row["people_total"], row["people_class"]), (2, "2"))
        for key in ("people_near", "adults", "age_unknown", "bicycles", "notes", *DIR_FIELDS):
            self.assertIsNone(row[key])

    def test_failed_photo(self):
        row = image_row(failed(), RUN)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error"], "UnidentifiedImageError: cannot identify image file")
        for key in ("people_count_assessable", "people_total", "people_near", "people_class", "adults", "children",
                    "age_unknown", "seconds", "requests", "repaired", "consistency_problems", "scene_uncertain",
                    "notes", *DIR_FIELDS, *OBJECT_FIELDS):
            with self.subTest(key=key):
                self.assertIsNone(row[key])
        self.assertEqual(row["relative_path"], "cam/broken.jpg")
        self.assertEqual(row["event_id"], "cam__2024-05-03T07-30-00")

    def test_failed_photo_ignores_a_partial_answer(self):
        row = image_row(failed(answer=photo_answer(4), info={"requests": 1, "seconds": 1.0}), RUN)
        self.assertIsNone(row["people_total"])
        self.assertEqual(row["requests"], 1)  # the request info is still reported

    def test_failed_photo_without_hash(self):
        row = image_row(failed(sha256="", image_id="img_000007"), RUN)
        self.assertEqual((row["sha256"], row["image_id"]), ("", "img_000007"))


def event(items, answer=None, info=None, event_id="cam__2024-05-03T07-30-00", **extra):
    entry = {"event_id": event_id, "items": items}
    if answer is not None:
        entry["answer"] = answer
        entry["info"] = {"requests": 1, "seconds": 20.0, "repaired": False, "problems": []} if info is None else info
    entry.update(extra)
    return entry


def three_photos(*counts, **extra):
    counts = counts or (1, 2, 1)
    times = ["2024-05-03T07:30:00", "2024-05-03T07:30:20", "2024-05-03T07:30:45", "2024-05-03T07:31:30"]
    return [item(f"cam/IMG_{i + 1:04d}.jpg", answer=photo_answer(n, people_near=max(n - 1, 0)),
                 capture_time=times[i], **extra)
            for i, n in enumerate(counts)]


class EventRowTests(unittest.TestCase):
    def test_columns_are_exactly_the_event_fields(self):
        self.assertEqual(list(event_row(event(three_photos(), event_answer(3, 2)), RUN)), EVENT_FIELDS)
        self.assertEqual(list(event_row(event([item()]), RUN)), EVENT_FIELDS)
        self.assertEqual(list(event_row(event([failed()]), RUN)), EVENT_FIELDS)

    def test_multi_photo_model_answer(self):
        answer = event_answer(4, 2, unique_near=3, adults=2, teens=1, children=1, age_unclear=0,
                              direction_counts={"left": 3, "right": 1, "toward": 0, "away": 0, "stationary": 0,
                                                "unclear": 0},
                              bicycles=2, notes="a family")
        info = {"requests": 2, "seconds": 31.5, "repaired": True, "problems": []}
        row = event_row(event(three_photos(1, 2, 1), answer, info), RUN)
        self.assertEqual(row["method"], "multi_photo_model")
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["error"])
        self.assertEqual((row["run_id"], row["event_id"], row["camera_id"]), (RUN, "cam__2024-05-03T07-30-00", "cam"))
        self.assertEqual((row["start"], row["end"], row["duration_seconds"], row["image_count"]),
                         ("2024-05-03T07:30:00", "2024-05-03T07:30:45", 45.0, 3))
        self.assertEqual((row["people_unique"], row["people_near_unique"]), (4, 3))
        self.assertEqual((row["people_max_frame"], row["people_near_max_frame"]), (2, 1))  # from the photos
        self.assertEqual((row["people_unique_class"], row["people_near_unique_class"]), ("3–4", "3–4"))
        self.assertEqual((row["adults"], row["teens"], row["children"], row["age_unknown"]), (2, 1, 1, 0))
        self.assertEqual([row[f] for f in DIR_FIELDS], [3, 1, 0, 0, 0, 0])
        self.assertEqual(row["bicycles"], 2)  # different objects across the visit, from the model
        self.assertEqual((row["seconds"], row["requests"], row["repaired"]), (31.5, 2, True))
        self.assertIsNone(row["consistency_problems"])
        self.assertEqual(row["notes"], "a family")
        self.assertIs(row["needs_review"], False)
        self.assertIsNone(row["review_reasons"])
        self.assertEqual(json.loads(row["images"]), ["cam/IMG_0001.jpg", "cam/IMG_0002.jpg", "cam/IMG_0003.jpg"])

    def test_inconsistent_model_answer_needs_review(self):
        info = {"requests": 2, "seconds": 3.0, "repaired": True, "problems": ["age groups sum to 2, not unique_all 3"]}
        row = event_row(event(three_photos(), event_answer(3, 2), info), RUN)
        self.assertEqual(row["consistency_problems"], "age groups sum to 2, not unique_all 3")
        self.assertIs(row["needs_review"], True)
        self.assertEqual(row["review_reasons"], "inconsistent_model_answer")

    def test_unique_below_the_busiest_photo_needs_review(self):
        row = event_row(event(three_photos(1, 3, 1), event_answer(2, 3)), RUN)
        self.assertEqual(row["review_reasons"], "unique_below_busiest_photo")
        row = event_row(event(three_photos(1, 3, 1), event_answer(3, 3)), RUN)
        self.assertIs(row["needs_review"], False)

    def test_unknown_unique_count(self):
        row = event_row(event(three_photos(), event_answer(None, 2)), RUN)
        self.assertIsNone(row["people_unique"])
        self.assertIsNone(row["people_unique_class"])
        self.assertEqual(row["review_reasons"], "unique_counts_not_available")

    def test_unknown_unique_count_with_a_near_count_does_not_crash(self):
        row = event_row(event(three_photos(1, 3, 1), event_answer(None, 3, unique_near=1)), RUN)
        self.assertIsNone(row["people_unique"])
        self.assertEqual(row["people_near_unique"], 2)  # raised to the busiest photo's near count, no upper bound

    def test_multi_answer_without_direction_counts(self):
        row = event_row(event(three_photos(), event_answer(2, 2, direction_counts=None)), RUN)
        self.assertEqual([row[f] for f in DIR_FIELDS], [None] * 6)

    def test_single_photo_event(self):
        answer = photo_answer(2, people_near=1, adults=1, children=1, dogs=1,
                              direction_counts={"left": 0, "right": 0, "toward": 1, "away": 1, "stationary": 0,
                                                "unclear": 0}, notes="dog on lead")
        row = event_row(event([item(answer=answer)]), RUN)
        self.assertEqual(row["method"], "single_photo")
        self.assertEqual(row["status"], "ok")
        self.assertEqual((row["people_unique"], row["people_near_unique"]), (2, 1))
        self.assertEqual((row["people_max_frame"], row["people_near_max_frame"]), (2, 1))
        self.assertEqual((row["people_unique_class"], row["people_near_unique_class"]), ("2", "1"))
        self.assertEqual((row["adults"], row["children"], row["age_unknown"], row["dogs"]), (1, 1, 0, 1))
        self.assertEqual([row[f] for f in DIR_FIELDS], [0, 0, 1, 1, 0, 0])
        self.assertEqual(row["notes"], "dog on lead")
        self.assertEqual((row["start"], row["end"], row["duration_seconds"], row["image_count"]),
                         ("2024-05-03T07:30:00", "2024-05-03T07:30:00", 0.0, 1))
        self.assertIs(row["needs_review"], False)

    def test_single_photo_that_saw_nobody(self):
        row = event_row(event([item(answer=photo_answer(0))]), RUN)
        self.assertEqual((row["people_unique"], row["people_unique_class"], row["age_unknown"]), (0, "0", 0))

    def test_failed_single_photo(self):
        row = event_row(event([failed()]), RUN)
        self.assertEqual(row["method"], "per_photo_max")
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["people_unique"])
        self.assertIsNone(row["people_max_frame"])
        for key in ("adults", "children", "age_unknown", *OBJECT_FIELDS):
            self.assertIsNone(row[key])
        self.assertEqual(row["review_reasons"], "unique_counts_not_available; photo_analysis_failed")
        self.assertIs(row["needs_review"], True)

    def test_per_photo_maxima_without_a_multi_photo_answer(self):
        photos = [item("cam/a.jpg", answer=photo_answer(2, adults=2, bicycles=1)),
                  item("cam/b.jpg", answer=photo_answer(2, adults=1, children=1, dogs=1),
                       capture_time="2024-05-03T07:30:30")]
        row = event_row(event(photos), RUN)
        self.assertEqual(row["method"], "per_photo_max")
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["error"])
        self.assertIsNone(row["people_unique"])
        self.assertIsNone(row["people_near_unique"])
        self.assertIsNone(row["people_unique_class"])
        self.assertEqual(row["people_max_frame"], 2)
        # Ages come from one (the first busiest) photo so they add up; objects are per-object maxima.
        self.assertEqual((row["adults"], row["children"], row["bicycles"], row["dogs"]), (2, 0, 1, 1))
        self.assertEqual(row["age_unknown"], 0)
        self.assertEqual(sum(row[f] for f in DIR_FIELDS), 2)
        self.assertEqual(row["review_reasons"], "unique_counts_not_available")
        self.assertIsNone(row["seconds"])

    def test_failed_multi_photo_request(self):
        row = event_row(event(three_photos(), status="error", error="event request failed: OllamaError: x"), RUN)
        self.assertEqual(row["method"], "per_photo_max")
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error"], "event request failed: OllamaError: x")
        self.assertEqual(row["people_max_frame"], 2)
        self.assertIs(row["needs_review"], True)

    def test_failed_photo_inside_an_event(self):
        photos = three_photos(1, 2) + [failed("cam/IMG_0003.jpg", capture_time="2024-05-03T07:31:00")]
        row = event_row(event(photos, event_answer(2, 2)), RUN)
        self.assertEqual(row["method"], "multi_photo_model")
        self.assertEqual(row["image_count"], 3)
        self.assertEqual(row["end"], "2024-05-03T07:31:00")  # failed photos still have a time
        self.assertEqual(row["review_reasons"], "photo_analysis_failed")
        row = event_row(event(photos), RUN)
        self.assertEqual(row["method"], "per_photo_max")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["review_reasons"], "unique_counts_not_available; photo_analysis_failed")

    def test_all_photos_failed(self):
        row = event_row(event([failed("cam/a.jpg"), failed("cam/b.jpg")]), RUN)
        self.assertEqual((row["method"], row["status"]), ("per_photo_max", "error"))
        self.assertIsNone(row["people_max_frame"])

    def test_split_event_needs_review(self):
        row = event_row(event(three_photos(), event_answer(3, 2), sampled=True), RUN)
        self.assertEqual(row["review_reasons"], "event_sampled_for_model")

    def test_file_dates_need_review_only_with_several_photos(self):
        photos = three_photos(1, 1)
        photos[1]["time_source"] = "file_mtime"
        row = event_row(event(photos, event_answer(1, 1)), RUN)
        self.assertEqual(row["review_reasons"], "times_from_file_dates")
        photos[1]["time_source"] = None
        self.assertEqual(event_row(event(photos, event_answer(1, 1)), RUN)["review_reasons"],
                         "times_from_file_dates")
        single = event_row(event([item(time_source="file_mtime")]), RUN)
        self.assertIs(single["needs_review"], False)

    def test_reasons_are_joined_in_order(self):
        photos = three_photos(1, 3) + [failed("cam/c.jpg", time_source="file_mtime")]
        info = {"problems": ["x"]}
        row = event_row(event(photos, event_answer(2, 3), info, sampled=True), RUN)
        self.assertEqual(row["review_reasons"], "inconsistent_model_answer; unique_below_busiest_photo; "
                                                "photo_analysis_failed; event_sampled_for_model; times_from_file_dates")

    def test_blank_and_zero_are_kept_apart(self):
        unassessable = photo_answer(None, people_count_assessable=False, people_near=None, adults=None, teens=None,
                                    children=None, age_unclear=None, direction_counts=None)
        row = event_row(event([item("cam/a.jpg", answer=unassessable), item("cam/b.jpg", answer=unassessable)]), RUN)
        self.assertIsNone(row["people_max_frame"])
        self.assertIsNone(row["adults"])
        self.assertIsNone(row["age_unknown"])
        self.assertEqual(row["bicycles"], 0)
        row = event_row(event([item("cam/a.jpg", answer=unassessable), item("cam/b.jpg", answer=photo_answer(0))]),
                        RUN)
        self.assertEqual((row["people_max_frame"], row["adults"], row["age_unknown"]), (0, 0, 0))

    def test_ages_from_the_multi_photo_answer(self):
        row = event_row(event(three_photos(), event_answer(3, 2, adults=0, teens=2, age_unclear=1)), RUN)
        self.assertEqual((row["adults"], row["teens"], row["children"], row["age_unknown"]), (0, 2, 0, 1))
        row = event_row(event(three_photos(), event_answer(3, 2, teens=None)), RUN)
        self.assertIsNone(row["teens"])

    def test_events_without_times(self):
        photos = [item("cam/a.jpg", capture_time=None, time_source=None),
                  item("cam/b.jpg", capture_time=None, time_source=None)]
        row = event_row(event(photos), RUN)
        self.assertEqual((row["start"], row["end"], row["duration_seconds"]), (None, None, None))

    def test_unsorted_times(self):
        photos = [item("cam/a.jpg", capture_time="2024-05-03T07:31:00"),
                  item("cam/b.jpg", capture_time="2024-05-03T07:30:00")]
        row = event_row(event(photos), RUN)
        self.assertEqual((row["start"], row["end"], row["duration_seconds"]),
                         ("2024-05-03T07:30:00", "2024-05-03T07:31:00", 60.0))

    def test_image_list_keeps_unicode(self):
        row = event_row(event([item("מצלמה צפון/תמונה 1.jpg")]), RUN)
        self.assertIn("מצלמה צפון/תמונה 1.jpg", row["images"])
        self.assertEqual(json.loads(row["images"]), ["מצלמה צפון/תמונה 1.jpg"])


def ev_row(camera="cam", start="2024-05-03T07:30:00", images=1, needs_review=False, **values):
    row = dict.fromkeys(EVENT_FIELDS)
    row.update(camera_id=camera, start=start, image_count=images, needs_review=needs_review)
    row.update(values)
    return row


class SummaryRowsTests(unittest.TestCase):
    def test_events_are_added_up_per_camera_and_day(self):
        rows = [ev_row("north", "2024-05-03T07:30:00", 3, False, people_unique=4, people_max_frame=2, adults=3,
                       children=1, dir_left=4, bicycles=1),
                ev_row("north", "2024-05-03T18:00:00", 1, True, people_unique=1, people_max_frame=1, adults=1,
                       children=0, dir_left=0, bicycles=0),
                ev_row("north", "2024-05-04T07:00:00", 2, False, people_unique=2, people_max_frame=2),
                ev_row("south", "2024-05-03T09:00:00", 1, False, people_unique=0, people_max_frame=0)]
        summary = summary_rows(rows, RUN)
        self.assertEqual([(r["camera_id"], r["date"]) for r in summary],
                         [("north", "2024-05-03"), ("north", "2024-05-04"), ("south", "2024-05-03")])
        first = summary[0]
        self.assertEqual(first["run_id"], RUN)
        self.assertEqual((first["events"], first["images"], first["events_needing_review"]), (2, 4, 1))
        self.assertEqual((first["people_unique"], first["people_max_frame"]), (5, 3))
        self.assertEqual((first["adults"], first["children"], first["dir_left"], first["bicycles"]), (4, 1, 4, 1))
        self.assertEqual(summary[2]["people_unique"], 0)

    def test_all_blank_stays_blank_and_partial_blanks_are_skipped(self):
        rows = [ev_row(people_unique=None, people_max_frame=2), ev_row(people_unique=3, people_max_frame=None),
                ev_row(people_unique=None, people_max_frame=None)]
        out = summary_rows(rows, RUN)[0]
        self.assertEqual((out["people_unique"], out["people_max_frame"]), (3, 2))
        self.assertIsNone(out["adults"])
        self.assertIsNone(out["dir_left"])

    def test_events_without_a_start_are_grouped_as_unknown(self):
        out = summary_rows([ev_row(start=None, people_unique=1), ev_row(start=None, people_unique=2)], RUN)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["date"], out[0]["people_unique"]), ("unknown", 3))

    def test_columns(self):
        out = summary_rows([ev_row(people_unique=1)], RUN)[0]
        self.assertEqual(set(out), set(SUMMARY_FIELDS))

    def test_no_events(self):
        self.assertEqual(summary_rows([], RUN), [])

    def test_rows_from_event_row(self):
        rows = [event_row(event(three_photos(1, 2, 1), event_answer(3, 2)), RUN),
                event_row(event([item("cam/IMG_0009.jpg", answer=photo_answer(1), capture_time="2024-05-03T12:00:00")],
                                event_id="cam__2"), RUN),
                event_row(event([failed(capture_time="2024-05-03T13:00:00")], event_id="cam__3"), RUN)]
        out = summary_rows(rows, RUN)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["events"], out[0]["images"], out[0]["events_needing_review"]), (3, 5, 1))
        self.assertEqual((out[0]["people_unique"], out[0]["people_max_frame"]), (4, 3))


class SpreadsheetSafeTests(unittest.TestCase):
    def test_formula_like_text_is_quoted(self):
        for value in ("=SUM(A1:A3)", "+1", "-1", "@cmd", "  =1+1", " -x", "=HYPERLINK(\"http://x\")"):
            with self.subTest(value=value):
                self.assertEqual(spreadsheet_safe(value), "'" + value)

    def test_leading_control_characters_are_quoted(self):
        for value in ("\tx", "\rx", "\nx", "\t=1"):
            with self.subTest(value=repr(value)):
                self.assertEqual(spreadsheet_safe(value), "'" + value)

    def test_ordinary_text_is_unchanged(self):
        for value in ("cam/IMG_0001.jpg", "a=b", "3–4", ">10", "", "2024-05-03T07:30:00", "x-y", '["a.jpg"]',
                      "מצלמה"):
            with self.subTest(value=value):
                self.assertEqual(spreadsheet_safe(value), value)

    def test_non_text_is_unchanged(self):
        for value in (-1, 0, 2.5, -0.5, None, True, False):
            with self.subTest(value=value):
                self.assertIs(spreadsheet_safe(value), value)


class WriteCsvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trailcam_vlm_csv_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def read(self, path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return list(csv.reader(handle))

    def test_utf8_bom_header_and_values(self):
        path = self.tmp / "out.csv"
        write_csv(path, [{"a": 1, "b": None, "c": True}, {"a": "x", "b": 2.5, "c": False}], ["a", "b", "c"])
        raw = path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbfa,b,c\r\n"))
        self.assertEqual(self.read(path), [["a", "b", "c"], ["1", "", "True"], ["x", "2.5", "False"]])

    def test_column_order_follows_the_field_list(self):
        path = self.tmp / "out.csv"
        write_csv(path, [{"b": 2, "a": 1}], ["a", "b"])
        self.assertEqual(self.read(path), [["a", "b"], ["1", "2"]])

    def test_extra_keys_are_dropped_and_missing_keys_blank(self):
        path = self.tmp / "out.csv"
        write_csv(path, [{"a": 1, "path": Path("x"), "item": object()}, {}], ["a", "b"])
        self.assertEqual(self.read(path), [["a", "b"], ["1", ""], ["", ""]])

    def test_formulas_are_neutralised(self):
        path = self.tmp / "out.csv"
        write_csv(path, [{"relative_path": "=cmd|' /C calc'!A0.jpg", "notes": "@SUM(1)", "n": -3}],
                  ["relative_path", "notes", "n"])
        self.assertEqual(self.read(path)[1], ["'=cmd|' /C calc'!A0.jpg", "'@SUM(1)", "-3"])

    def test_unicode_and_class_labels_round_trip(self):
        path = self.tmp / "out.csv"
        write_csv(path, [{"relative_path": "מצלמה צפון/תמונה, 1.jpg", "people_class": count_class(4),
                          "notes": 'two "quoted"\nlines'}], ["relative_path", "people_class", "notes"])
        self.assertEqual(self.read(path)[1], ["מצלמה צפון/תמונה, 1.jpg", "3–4", 'two "quoted"\nlines'])
        self.assertIn("3–4".encode("utf-8"), path.read_bytes())

    def test_header_only_without_rows(self):
        path = self.tmp / "empty.csv"
        write_csv(path, [], IMAGE_FIELDS)
        self.assertEqual(self.read(path), [IMAGE_FIELDS])

    def test_existing_file_is_replaced_and_no_temporary_files_remain(self):
        path = self.tmp / "out.csv"
        path.write_text("old", encoding="utf-8")
        write_csv(str(path), [{"a": 1}], ["a"])
        self.assertEqual(self.read(path), [["a"], ["1"]])
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["out.csv"])

    def test_failure_leaves_the_previous_file_and_no_temporary_files(self):
        path = self.tmp / "out.csv"
        path.write_text("previous", encoding="utf-8")

        def rows():
            yield {"a": 1}
            raise RuntimeError("disk full")

        with self.assertRaises(RuntimeError):
            write_csv(path, rows(), ["a"])
        self.assertEqual(path.read_text(encoding="utf-8"), "previous")
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["out.csv"])

    def test_missing_folder_raises(self):
        with self.assertRaises(OSError):
            write_csv(self.tmp / "missing" / "out.csv", [{"a": 1}], ["a"])
        self.assertFalse((self.tmp / "missing").exists())

    def test_real_rows_round_trip(self):
        path = self.tmp / "images.csv"
        rows = [image_row(item(answer=photo_answer(4)), RUN), image_row(failed(), RUN)]
        write_csv(path, rows, IMAGE_FIELDS)
        with open(path, newline="", encoding="utf-8-sig") as handle:
            read = list(csv.DictReader(handle))
        self.assertEqual(read[0]["people_total"], "4")
        self.assertEqual(read[0]["people_class"], "3–4")
        self.assertEqual(read[0]["consistency_problems"], "")
        self.assertEqual(read[1]["people_total"], "")
        self.assertEqual(read[1]["status"], "error")
        self.assertEqual(list(read[0]), IMAGE_FIELDS)


class ModuleTests(unittest.TestCase):
    def test_public_names(self):
        for name in ("IMAGE_FIELDS", "EVENT_FIELDS", "SUMMARY_FIELDS", "image_row", "event_row", "summary_rows",
                     "write_csv", "count_class", "spreadsheet_safe"):
            self.assertTrue(hasattr(export, name), name)


if __name__ == "__main__":
    unittest.main()
