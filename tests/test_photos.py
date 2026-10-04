"""Photo discovery, capture times, sequence numbers, camera ids, events and image loading."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import fakes_vlm  # noqa: F401  (puts the repository on sys.path)
import numpy as np
from fakes_vlm import make_photo, set_mtime
from PIL import Image, UnidentifiedImageError

from trailcam_vlm import photos
from trailcam_vlm.photos import (EXTENSIONS, camera_id_for, capture_time, discover, file_hash, group_events,
                                 open_rgb, sequence_number)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trailcam_vlm_photos_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def touch(self, relative, data=b"x"):
        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def relative(self, paths, root=None):
        root = root or self.tmp
        return [p.relative_to(root).as_posix() for p in paths]


class DiscoverTests(TempDirTestCase):
    def test_supported_extensions_in_any_case(self):
        for name in ("a.jpg", "b.JPG", "c.jpeg", "d.png", "e.bmp", "f.tif", "g.TIFF", "h.webp", "i.Jpeg"):
            self.touch(name)
        for name in ("notes.txt", "clip.mp4", "raw.heic", "noext", "a.jpg.bak", "thumbs.db", "x.gif"):
            self.touch(name)
        self.assertEqual(self.relative(discover(self.tmp)),
                         ["a.jpg", "b.JPG", "c.jpeg", "d.png", "e.bmp", "f.tif", "g.TIFF", "h.webp", "i.Jpeg"])
        self.assertEqual(EXTENSIONS, {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

    def test_sorted_case_insensitively_by_relative_path(self):
        for name in ("b.jpg", "A.jpg", "a/c.jpg", "B/a.jpg", "a.JPG2.jpg", "Z/z.jpg", "z.jpg"):
            self.touch(name)
        self.assertEqual(self.relative(discover(self.tmp)),
                         ["A.jpg", "a.JPG2.jpg", "a/c.jpg", "b.jpg", "B/a.jpg", "z.jpg", "Z/z.jpg"])

    def test_recursive_by_default(self):
        self.touch("top.jpg")
        self.touch("cam1/IMG_1.jpg")
        self.touch("cam1/deeper/IMG_2.jpg")
        self.assertEqual(self.relative(discover(self.tmp)), ["cam1/deeper/IMG_2.jpg", "cam1/IMG_1.jpg", "top.jpg"])

    def test_non_recursive_lists_only_the_top_folder(self):
        self.touch("top.jpg")
        self.touch("cam1/IMG_1.jpg")
        self.assertEqual(self.relative(discover(self.tmp, recursive=False)), ["top.jpg"])

    def test_hidden_folders_are_skipped(self):
        self.touch(".thumbnails/a.jpg")
        self.touch("cam/.cache/b.jpg")
        self.touch("cam/c.jpg")
        self.assertEqual(self.relative(discover(self.tmp)), ["cam/c.jpg"])

    def test_excluded_folders_are_skipped(self):
        self.touch("cam/a.jpg")
        self.touch("exports/old/contact.jpg")
        self.touch("cache/x.png")
        self.touch("cam/exports/kept.jpg")  # only the excluded path itself, not a same-named folder elsewhere
        found = discover(self.tmp, excluded=(self.tmp / "exports", self.tmp / "cache"))
        self.assertEqual(self.relative(found), ["cam/a.jpg", "cam/exports/kept.jpg"])

    def test_excluded_paths_are_resolved(self):
        self.touch("cam/a.jpg")
        self.touch("exports/b.jpg")
        found = discover(self.tmp, excluded=[str(self.tmp / "cam" / ".." / "exports")])
        self.assertEqual(self.relative(found), ["cam/a.jpg"])

    def test_nested_excluded_folder(self):
        self.touch("cam/a.jpg")
        self.touch("cam/out/b.jpg")
        self.touch("cam/out/deep/c.jpg")
        self.assertEqual(self.relative(discover(self.tmp, excluded=[self.tmp / "cam" / "out"])), ["cam/a.jpg"])

    def test_excluded_folder_outside_the_input_is_harmless(self):
        self.touch("in/a.jpg")
        self.assertEqual(self.relative(discover(self.tmp / "in", excluded=[self.tmp / "elsewhere"]),
                                       self.tmp / "in"), ["a.jpg"])

    def test_paths_are_returned_under_the_given_folder(self):
        self.touch("cam/a.jpg")
        found = discover(str(self.tmp))
        self.assertEqual(found, [self.tmp / "cam" / "a.jpg"])
        self.assertIsInstance(found[0], Path)

    def test_empty_folder(self):
        self.assertEqual(discover(self.tmp), [])

    def test_unicode_folder_and_file_names(self):
        self.touch("מצלמה צפון/תמונה 1.jpg")
        self.touch("Café/IMG_0001.JPG")
        self.assertEqual(self.relative(discover(self.tmp)), ["Café/IMG_0001.JPG", "מצלמה צפון/תמונה 1.jpg"])

    def test_symlinked_files_and_folders_are_skipped(self):
        target_dir = self.tmp / "outside"
        target_file = self.touch("outside/real.jpg")
        root = self.tmp / "in"
        self.touch("in/cam/own.jpg")
        try:
            os.symlink(target_file, root / "cam" / "link.jpg")
            os.symlink(target_dir, root / "linked_dir", target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create symlinks here: {exc}")
        self.assertEqual(self.relative(discover(root), root), ["cam/own.jpg"])

    def test_symlinks_are_skipped_simulated(self):
        # Creating symlinks needs a privilege on Windows; simulate them so the rule is always exercised.
        self.touch("in/cam/own.jpg")
        self.touch("in/cam/link.jpg")
        self.touch("in/linked_dir/x.jpg")
        root = self.tmp / "in"
        real = Path.is_symlink

        def is_symlink(path):
            return path.name in ("link.jpg", "linked_dir") or real(path)

        with mock.patch.object(Path, "is_symlink", is_symlink):
            found = discover(root)
        self.assertEqual(self.relative(found, root), ["cam/own.jpg"])


class CaptureTimeTests(TempDirTestCase):
    def photo_with_exif(self, name, original=None, digitized=None, ifd0=None, fmt="jpg"):
        path = self.tmp / f"{name}.{fmt}"
        exif = Image.Exif()
        exif[0x010F] = "TestCam"  # an IFD0 tag: Pillow drops PNG EXIF that holds only the Exif sub-IFD
        if original is not None:
            exif.get_ifd(0x8769)[0x9003] = original
        if digitized is not None:
            exif.get_ifd(0x8769)[0x9004] = digitized
        if ifd0 is not None:
            exif[0x0132] = ifd0
        Image.new("RGB", (32, 24), (1, 2, 3)).save(path, exif=exif)
        set_mtime(path, datetime(2020, 1, 2, 3, 4, 5))
        return path

    def test_date_time_original_is_preferred(self):
        path = self.photo_with_exif("a", "2024:05:03 07:30:15", "2023:01:01 00:00:00", "2022:01:01 00:00:00")
        self.assertEqual(capture_time(path), ("2024-05-03T07:30:15", "exif_original"))

    def test_falls_back_to_date_time_digitized(self):
        path = self.photo_with_exif("b", None, "2023:06:07 08:09:10", "2022:01:01 00:00:00")
        self.assertEqual(capture_time(path), ("2023-06-07T08:09:10", "exif_digitized"))

    def test_falls_back_to_ifd0_date_time(self):
        path = self.photo_with_exif("c", ifd0="2022:02:02 02:02:02")
        self.assertEqual(capture_time(path), ("2022-02-02T02:02:02", "exif_datetime"))

    def test_png_exif_is_read(self):
        path = self.photo_with_exif("d", "2024:05:03 07:30:15", fmt="png")
        self.assertEqual(capture_time(path), ("2024-05-03T07:30:15", "exif_original"))

    def test_clock_reset_years_are_skipped(self):
        path = self.photo_with_exif("e", "1980:01:01 00:00:00", "1970:01:01 00:00:00", "2021:03:04 05:06:07")
        self.assertEqual(capture_time(path), ("2021-03-04T05:06:07", "exif_datetime"))
        path = self.photo_with_exif("f", "1981:01:01 00:00:00")
        self.assertEqual(capture_time(path), ("1981-01-01T00:00:00", "exif_original"))

    def test_malformed_exif_times_fall_through(self):
        path = self.photo_with_exif("g", "0000:00:00 00:00:00", "2024:13:45 99:99:99", "    :  :     :  :  ")
        self.assertEqual(capture_time(path), ("2020-01-02T03:04:05", "file_mtime"))

    def test_without_exif_the_file_time_is_used(self):
        path = self.tmp / "plain.jpg"
        Image.new("RGB", (8, 8)).save(path)
        stamp = datetime(2024, 7, 1, 6, 5, 4, 700000).timestamp()
        os.utime(path, (stamp, stamp))
        self.assertEqual(capture_time(path), ("2024-07-01T06:05:04", "file_mtime"))

    def test_unreadable_image_uses_the_file_time(self):
        path = self.touch("broken.jpg", b"this is not a jpeg")
        set_mtime(path, datetime(2023, 3, 3, 3, 3, 3))
        self.assertEqual(capture_time(path), ("2023-03-03T03:03:03", "file_mtime"))

    def test_missing_file(self):
        self.assertEqual(capture_time(self.tmp / "missing.jpg"), (None, None))

    def _file_time_raising(self, error):
        path = self.tmp / "plain.jpg"
        Image.new("RGB", (8, 8)).save(path)

        class Clock(datetime):
            @classmethod
            def fromtimestamp(cls, *args, **kwargs):
                raise error

        with mock.patch.object(photos, "datetime", Clock):
            return capture_time(path)

    def test_file_time_the_platform_cannot_convert(self):
        # Windows raises OSError for NTFS times beyond year 9999.
        self.assertEqual(self._file_time_raising(OSError(22, "Invalid argument")), (None, None))

    # Other platforms raise ValueError ("year is out of range") or OverflowError for such file times;
    # capture_time runs outside the per-photo try in __main__.run, so it must not raise.
    def test_out_of_range_file_time_is_not_fatal(self):
        for error in (OverflowError("timestamp out of range for platform time_t"),
                      ValueError("year 60000 is out of range")):
            self.assertEqual(self._file_time_raising(error), (None, None))

    def test_exif_text_parsing(self):
        parse = photos._parse_exif_time
        self.assertEqual(parse("2024:05:03 07:30:15"), datetime(2024, 5, 3, 7, 30, 15))
        self.assertEqual(parse("2024:05:03 07:30:15\x00"), datetime(2024, 5, 3, 7, 30, 15))
        self.assertEqual(parse("  2024:05:03 07:30:15  "), datetime(2024, 5, 3, 7, 30, 15))
        self.assertEqual(parse("2024:05:03 07:30:15.123"), datetime(2024, 5, 3, 7, 30, 15))
        for bad in (None, "", "2024-05-03 07:30:15", "0000:00:00 00:00:00", "2024:02:30 10:00:00", "garbage"):
            with self.subTest(value=bad):
                self.assertIsNone(parse(bad))

    def test_exif_tag_order(self):
        self.assertEqual([source for *_, source in photos.EXIF_TIME_TAGS],
                         ["exif_original", "exif_digitized", "exif_datetime"])


class SequenceNumberTests(unittest.TestCase):
    def test_last_digit_run_of_the_file_name(self):
        cases = {"IMAG0571.JPG": 571, "cam/IMG_0003.jpg": 3, "2024-05-03_0012.jpg": 12, "DSCF0001 (2).jpg": 2,
                 "RCNX0100.JPG": 100, "folder123/photo.jpg": None, "photo.jpg": None, "0.jpg": 0}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(sequence_number(Path(name)), expected)

    def test_accepts_strings(self):
        self.assertEqual(sequence_number("cam/IMAG0042.jpg"), 42)

    def test_very_long_digit_runs_are_not_counters(self):
        self.assertEqual(sequence_number("12345678901.jpg"), 12345678901)
        self.assertIsNone(sequence_number("123456789012.jpg"))
        self.assertIsNone(sequence_number("IMG_20240503073015123.jpg"))


class CameraIdTests(unittest.TestCase):
    def test_folder_mode_uses_the_parent_folder(self):
        self.assertEqual(camera_id_for("North gate/IMG_0001.JPG"), "North gate")
        self.assertEqual(camera_id_for("site/cam 2/2024-05/IMG.jpg"), "site/cam 2/2024-05")

    def test_windows_separators_are_normalised(self):
        self.assertEqual(camera_id_for("site\\cam 2\\IMG.jpg"), "site/cam 2")

    def test_root_photos_get_dot(self):
        self.assertEqual(camera_id_for("IMG_0001.jpg"), ".")
        self.assertEqual(camera_id_for(Path("IMG_0001.jpg")), ".")

    def test_single_mode(self):
        self.assertEqual(camera_id_for("a/b/c.jpg", "single"), "all")
        self.assertEqual(camera_id_for("c.jpg", "single", r"(?P<camera>x)"), "all")

    def test_regex_mode(self):
        pattern = r"^People_100/\d{3}_(?P<camera>Cam-A|Cam_B|Cam-D)_"
        self.assertEqual(camera_id_for("People_100/001_Cam-A_IMG1.jpg", "regex", pattern), "Cam-A")
        self.assertEqual(camera_id_for("People_100\\002_Cam-D_x.jpg", "regex", pattern), "Cam-D")

    def test_regex_falls_back_to_the_folder(self):
        pattern = r"(?P<camera>CAM\d+)"
        self.assertEqual(camera_id_for("North/IMG_1.jpg", "regex", pattern), "North")
        self.assertEqual(camera_id_for("IMG_1.jpg", "regex", pattern), ".")
        self.assertEqual(camera_id_for("North/IMG_1.jpg", "regex", r"(?P<camera>x?)IMG"), "North")  # empty group
        self.assertEqual(camera_id_for("North/IMG_1.jpg", "regex", r"(IMG)"), "North")  # no named group
        self.assertEqual(camera_id_for("North/IMG_1.jpg", "regex", None), "North")

    def test_regex_searches_anywhere_in_the_path(self):
        self.assertEqual(camera_id_for("2024/site_CAM7/IMG_1.jpg", "regex", r"(?P<camera>CAM\d+)"), "CAM7")


def rec(name, camera="cam", when=None, seq=None):
    return {"relative_path": name, "camera_id": camera, "capture_time": when, "sequence_number": seq}


def ids_and_names(events):
    return [(e["event_id"], [r["relative_path"] for r in e["records"]]) for e in events]


class GroupEventsTests(unittest.TestCase):
    def test_gap_boundary(self):
        records = [rec("a", when="2024-05-03T07:30:00"), rec("b", when="2024-05-03T07:31:00"),
                   rec("c", when="2024-05-03T07:32:01")]
        events = group_events(records, gap_seconds=60)
        self.assertEqual([[r["relative_path"] for r in e["records"]] for e in events], [["a", "b"], ["c"]])

    def test_default_gap_is_60_seconds(self):
        records = [rec("a", when="2024-05-03T07:30:00"), rec("b", when="2024-05-03T07:31:00"),
                   rec("c", when="2024-05-03T07:32:01")]
        self.assertEqual(len(group_events(records)), 2)

    def test_fractional_gap(self):
        records = [rec("a", when="2024-05-03T07:30:00"), rec("b", when="2024-05-03T07:30:02")]
        self.assertEqual(len(group_events(records, 1.5)), 2)
        self.assertEqual(len(group_events(records, 2.0)), 1)

    def test_chain_of_short_gaps_stays_one_event(self):
        records = [rec(f"p{i}", when=f"2024-05-03T07:{i:02d}:00") for i in range(10)]
        self.assertEqual(len(group_events(records, 60)), 1)

    def test_cameras_are_never_combined(self):
        records = [rec("a", "north", "2024-05-03T07:30:00"), rec("b", "south", "2024-05-03T07:30:00"),
                   rec("c", "north", "2024-05-03T07:30:10")]
        self.assertEqual(ids_and_names(group_events(records)),
                         [("north__2024-05-03T07-30-00", ["a", "c"]), ("south__2024-05-03T07-30-00", ["b"])])

    def test_photos_are_ordered_by_time_whatever_the_input_order(self):
        records = [rec("late", when="2024-05-03T07:30:40"), rec("early", when="2024-05-03T07:30:00"),
                   rec("middle", when="2024-05-03T07:30:20")]
        self.assertEqual(ids_and_names(group_events(records)),
                         [("cam__2024-05-03T07-30-00", ["early", "middle", "late"])])

    def test_equal_times_are_ordered_by_sequence_then_path(self):
        t = "2024-05-03T07:30:00"
        records = [rec("b", when=t, seq=2), rec("z", when=t, seq=1), rec("a", when=t, seq=2), rec("n", when=t)]
        self.assertEqual(ids_and_names(group_events(records))[0][1], ["n", "z", "a", "b"])

    def test_untimed_photos_join_their_predecessor_when_numbers_are_close(self):
        records = [rec("IMG_0010.jpg", when="2024-05-03T07:30:00", seq=10), rec("IMG_0013.jpg", seq=13),
                   rec("IMG_0014.jpg", seq=14), rec("IMG_0018.jpg", seq=18)]
        self.assertEqual([names for _, names in ids_and_names(group_events(records))],
                         [["IMG_0010.jpg", "IMG_0013.jpg", "IMG_0014.jpg"], ["IMG_0018.jpg"]])

    def test_untimed_photos_without_numbers_are_separate_events(self):
        records = [rec("a.jpg"), rec("b.jpg"), rec("c.jpg", when="2024-05-03T07:30:00")]
        self.assertEqual(ids_and_names(group_events(records)),
                         [("cam__2024-05-03T07-30-00", ["c.jpg"]), ("cam__a.jpg", ["a.jpg"]),
                          ("cam__b.jpg", ["b.jpg"])])

    def test_untimed_photos_come_after_timed_ones(self):
        records = [rec("IMG_0001.jpg", seq=1), rec("IMG_0002.jpg", when="2024-05-03T07:30:00", seq=2)]
        self.assertEqual(ids_and_names(group_events(records)),
                         [("cam__2024-05-03T07-30-00", ["IMG_0002.jpg", "IMG_0001.jpg"])])

    def test_untimed_photos_ordered_by_sequence(self):
        records = [rec("IMG_0003.jpg", seq=3), rec("IMG_0001.jpg", seq=1), rec("IMG_0002.jpg", seq=2)]
        self.assertEqual(ids_and_names(group_events(records)),
                         [("cam__IMG_0001.jpg", ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"])])

    def test_event_id_is_camera_and_first_time(self):
        records = [rec("x/IMG_2.jpg", "North gate/2025-05 card", "2025-05-03T07:30:15")]
        self.assertEqual(group_events(records)[0]["event_id"], "North_gate_2025-05_card__2025-05-03T07-30-15")

    def test_root_camera_is_named_root(self):
        for camera in (".", ""):
            with self.subTest(camera=camera):
                events = group_events([rec("IMG_1.jpg", camera, "2024-05-03T07:30:00")])
                self.assertEqual(events[0]["event_id"], "root__2024-05-03T07-30-00")

    def test_untimed_event_id_uses_the_first_file_name(self):
        events = group_events([rec("sub dir/IMG 0001.JPG", "cam", None, 1)])
        self.assertEqual(events[0]["event_id"], "cam__IMG_0001.JPG")

    def test_unicode_camera_names_are_kept(self):
        events = group_events([rec("מצלמה צפון/a.jpg", "מצלמה צפון", "2024-05-03T07:30:00")])
        self.assertEqual(events[0]["event_id"], "מצלמה_צפון__2024-05-03T07-30-00")

    def test_ids_are_stable_when_photos_are_added_elsewhere(self):
        base = [rec("a", "north", "2024-05-03T07:30:00"), rec("b", "north", "2024-05-03T07:30:30"),
                rec("c", "south", "2024-05-03T08:00:00")]
        before = {e["event_id"] for e in group_events(base)}
        more = base + [rec("d", "aaa", "2024-05-03T07:00:00"), rec("e", "north", "2024-05-04T09:00:00"),
                       rec("f", "south", "2024-05-02T08:00:00")]
        after = {e["event_id"] for e in group_events(more)}
        self.assertLessEqual(before, after)
        self.assertEqual(len(after), 5)

    def test_colliding_ids_get_numbered_suffixes(self):
        t = "2024-05-03T07:30:00"
        records = [rec("1", "cam 1", t), rec("2", "cam/1", t), rec("3", "cam:1", t)]
        self.assertEqual([e["event_id"] for e in group_events(records)],
                         ["cam_1__2024-05-03T07-30-00", "cam_1__2024-05-03T07-30-00__02",
                          "cam_1__2024-05-03T07-30-00__03"])

    def test_records_are_returned_unchanged(self):
        records = [rec("a", when="2024-05-03T07:30:00"), rec("b", when="2024-05-03T07:30:10")]
        records[0]["item"] = object()
        events = group_events(records)
        self.assertIs(events[0]["records"][0], records[0])
        self.assertIs(events[0]["records"][1], records[1])
        self.assertNotIn("event_id", records[0])

    def test_every_record_appears_exactly_once(self):
        records = [rec(f"{cam}/{i}", cam, f"2024-05-03T07:{(i * 7) % 60:02d}:00" if i % 3 else None, i)
                   for cam in ("a", "b") for i in range(12)]
        events = group_events(records, 90)
        flattened = [r["relative_path"] for e in events for r in e["records"]]
        self.assertEqual(sorted(flattened), sorted(r["relative_path"] for r in records))
        self.assertEqual(len({e["event_id"] for e in events}), len(events))

    def test_no_records(self):
        self.assertEqual(group_events([]), [])


class OpenRgbTests(TempDirTestCase):
    @staticmethod
    def halves():
        array = np.zeros((20, 40, 3), np.uint8)
        array[:, :20] = (255, 0, 0)
        array[:, 20:] = (0, 0, 255)
        return Image.fromarray(array)

    def assertColour(self, pixel, expected, delta=12):
        for got, want in zip(pixel, expected):
            self.assertAlmostEqual(got, want, delta=delta)

    def test_exif_orientation_is_applied(self):
        for suffix in ("jpg", "png"):
            with self.subTest(format=suffix):
                path = self.tmp / f"rotated.{suffix}"
                exif = Image.Exif()
                exif[0x0112] = 6  # stored sideways: rotate 90 degrees clockwise to view
                self.halves().save(path, exif=exif)
                image = open_rgb(path)
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (20, 40))
                self.assertColour(image.getpixel((10, 5)), (255, 0, 0))  # left half is now on top
                self.assertColour(image.getpixel((10, 35)), (0, 0, 255))

    def test_upside_down_orientation(self):
        path = self.tmp / "flipped.jpg"
        exif = Image.Exif()
        exif[0x0112] = 3
        self.halves().save(path, exif=exif, quality=95)
        image = open_rgb(path)
        self.assertEqual(image.size, (40, 20))
        self.assertColour(image.getpixel((5, 10)), (0, 0, 255))

    def test_plain_photo_is_unchanged(self):
        path = self.tmp / "plain.png"
        self.halves().save(path)
        image = open_rgb(path)
        self.assertEqual(image.size, (40, 20))
        self.assertEqual(image.getpixel((5, 10)), (255, 0, 0))

    def test_16_bit_png_is_rescaled_not_clipped(self):
        values = np.tile(np.linspace(1000, 3000, 64).astype(np.uint16), (8, 1))
        path = self.tmp / "deep.png"
        Image.fromarray(values).save(path)
        with Image.open(path) as raw:
            self.assertTrue(raw.mode.startswith("I"))
        image = open_rgb(path)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (64, 8))
        dark, middle, bright = image.getpixel((0, 0)), image.getpixel((32, 0)), image.getpixel((63, 0))
        self.assertEqual(dark, (0, 0, 0))
        self.assertEqual(bright, (255, 255, 255))
        self.assertTrue(100 < middle[0] < 160, middle)
        self.assertEqual(middle[0], middle[1])
        row = [image.getpixel((x, 0))[0] for x in range(64)]
        self.assertEqual(row, sorted(row))

    def test_constant_16_bit_image(self):
        path = self.tmp / "flat.png"
        Image.fromarray(np.full((8, 8), 4000, np.uint16)).save(path)
        image = open_rgb(path)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (8, 8))

    def test_float_image_with_missing_values(self):
        values = np.full((8, 8), np.nan, np.float32)
        values[:, :2] = 1.0
        values[:, 4:] = 5.0
        path = self.tmp / "float.tif"
        Image.fromarray(values, "F").save(path)
        image = open_rgb(path)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(image.getpixel((3, 0)), (0, 0, 0))  # NaN becomes the low end
        self.assertEqual(image.getpixel((6, 0)), (255, 255, 255))

    def test_all_nan_float_image(self):
        path = self.tmp / "nan.tif"
        Image.fromarray(np.full((4, 4), np.nan, np.float32), "F").save(path)
        self.assertEqual(open_rgb(path).size, (4, 4))

    def test_other_modes_become_rgb(self):
        sources = {"rgba.png": Image.new("RGBA", (6, 4), (10, 20, 30, 0)),
                   "grey.png": Image.new("L", (6, 4), 77),
                   "palette.png": Image.new("RGB", (6, 4), (200, 100, 0)).convert("P"),
                   "cmyk.tif": Image.new("CMYK", (6, 4), (0, 0, 0, 0)),
                   "picture.webp": Image.new("RGB", (6, 4), (0, 128, 0)),
                   "picture.bmp": Image.new("RGB", (6, 4), (0, 128, 0))}
        for name, image in sources.items():
            with self.subTest(name=name):
                image.save(self.tmp / name)
                loaded = open_rgb(self.tmp / name)
                self.assertEqual(loaded.mode, "RGB")
                self.assertEqual(loaded.size, (6, 4))
        self.assertEqual(open_rgb(self.tmp / "grey.png").getpixel((0, 0)), (77, 77, 77))

    def test_file_is_closed_after_loading(self):
        path = make_photo(self.tmp / "a.jpg", people=1)
        open_rgb(path)
        path.unlink()  # fails on Windows if the file were still open
        self.assertFalse(path.exists())

    def test_corrupt_file_raises(self):
        path = self.touch("broken.jpg", b"not an image at all")
        with self.assertRaises(UnidentifiedImageError):
            open_rgb(path)

    def test_truncated_jpeg_raises(self):
        full = self.tmp / "full.jpg"
        Image.effect_noise((600, 400), 80).convert("RGB").save(full, quality=95)
        data = full.read_bytes()
        path = self.touch("truncated.jpg", data[: len(data) // 2])
        with self.assertRaises(OSError):
            open_rgb(path)


class FileHashTests(TempDirTestCase):
    def test_sha256_of_the_file_bytes(self):
        data = os.urandom(1024 * 1024 * 2 + 12345)  # spans several read chunks
        path = self.touch("big.bin", data)
        self.assertEqual(file_hash(path), hashlib.sha256(data).hexdigest())
        self.assertEqual(file_hash(str(path)), hashlib.sha256(data).hexdigest())

    def test_empty_file(self):
        self.assertEqual(file_hash(self.touch("empty.jpg", b"")), hashlib.sha256(b"").hexdigest())


if __name__ == "__main__":
    unittest.main()
