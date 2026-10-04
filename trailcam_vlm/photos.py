"""Photo discovery, capture times, camera ids and event grouping.

Same conventions as trailcam-expert-export v2: EXIF orientation is applied,
capture time comes from EXIF (DateTimeOriginal, then DateTimeDigitized, then
DateTime) or the file modification time, cameras are grouped by folder (or a
regex), and photos of one camera less than ``gap`` seconds apart form one
event. Relative order is what matters; camera clocks are often wrong.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path, PurePosixPath

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXIF_TIME_TAGS = ((0x9003, 0x8769, "exif_original"), (0x9004, 0x8769, "exif_digitized"), (0x0132, None, "exif_datetime"))


def discover(input_dir, excluded=(), recursive=True):
    input_dir = Path(input_dir)
    root = input_dir.resolve()
    # An excluded folder that contains the input folder (e.g. --output set to the input folder) excludes nothing.
    excluded = [e for e in (Path(p).resolve() for p in excluded) if not root.is_relative_to(e)]
    found = []
    for current, dirs, files in os.walk(input_dir):
        current = Path(current)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (current / d).is_symlink()
                         and not any((current / d).resolve().is_relative_to(e) for e in excluded)) if recursive else []
        for name in sorted(files):
            path = current / name
            if path.suffix.lower() in EXTENSIONS and not path.is_symlink():
                found.append(path)
    return sorted(found, key=lambda p: p.relative_to(input_dir).as_posix().casefold())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_rgb(path):
    """EXIF-oriented 8-bit RGB PIL image; 16-bit/float images are rescaled, not clipped."""
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
            import numpy as np
            array = np.asarray(image, dtype=np.float64)
            finite = array[np.isfinite(array)]
            low, high = (np.percentile(finite, (0.5, 99.5)) if finite.size else (0.0, 1.0))
            high = high if high > low else low + 1.0
            array = np.clip((np.nan_to_num(array, nan=low) - low) / (high - low), 0, 1) * 255
            return Image.fromarray(array.astype(np.uint8), "L").convert("RGB")
        return image.convert("RGB")


def _parse_exif_time(value):
    text = str(value or "").strip().rstrip("\x00").strip()
    try:
        return datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def capture_time(path):
    """(ISO capture time or None, source)."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            exif = image.getexif()
            for tag, ifd, source in EXIF_TIME_TAGS:
                block = exif.get_ifd(ifd) if ifd else exif
                moment = _parse_exif_time(block.get(tag))
                if moment and moment.year > 1980:
                    return moment.isoformat(), source
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).replace(microsecond=0).isoformat(), "file_mtime"
    except (OSError, OverflowError, ValueError):  # missing file or a file time outside the platform's range
        return None, None


def sequence_number(path):
    digits = re.findall(r"\d+", Path(path).stem)
    try:
        return int(digits[-1]) if digits and len(digits[-1]) < 12 else None
    except ValueError:
        return None


def camera_id_for(relative_path, mode="folder", pattern=None):
    relative_path = str(relative_path).replace("\\", "/")
    if mode == "single":
        return "all"
    if mode == "regex" and pattern:
        match = re.search(pattern, relative_path)
        if match and match.groupdict().get("camera"):
            return match.group("camera")
    parent = str(PurePosixPath(relative_path).parent)
    return parent if parent not in ("", ".") else "."


def group_events(records, gap_seconds=60.0):
    """Split each camera's photos (sorted by time, then sequence, then path) at gaps > gap_seconds.

    Photos without a time join their predecessor only when their sequence
    numbers are within 3. Returns a list of events (lists of records) with
    stable ids ``{camera}__{first time or first file name}`` (collisions get
    ``__02``, ``__03``).
    """
    def key(r):
        return (r.get("capture_time") is None, r.get("capture_time") or "", r.get("sequence_number") or 0,
                r["relative_path"])

    def continues(previous, record):
        a, b = previous.get("capture_time"), record.get("capture_time")
        if a and b:
            delta = (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
            return 0 <= delta <= gap_seconds
        sa, sb = previous.get("sequence_number"), record.get("sequence_number")
        return sa is not None and sb is not None and abs(sb - sa) <= 3

    cameras = {}
    for record in records:
        cameras.setdefault(record["camera_id"], []).append(record)
    events, seen = [], {}
    for camera in sorted(cameras):
        current = []
        for record in sorted(cameras[camera], key=key):
            if current and not continues(current[-1], record):
                events.append(current)
                current = []
            current.append(record)
        if current:
            events.append(current)
    output = []
    for event in events:
        first = event[0]
        stamp = (first.get("capture_time") or "").replace(":", "-") or PurePosixPath(first["relative_path"]).name
        camera = first["camera_id"] if first["camera_id"] not in ("", ".") else "root"
        base = re.sub(r"[^\w.-]+", "_", f"{camera}__{stamp}").strip("._") or "event"
        seen[base] = seen.get(base, 0) + 1
        output.append({"event_id": base if seen[base] == 1 else f"{base}__{seen[base]:02d}", "records": event})
    return output
