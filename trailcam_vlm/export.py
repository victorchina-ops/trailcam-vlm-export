"""CSV rows with the same column names as trailcam-expert-export v2.

Blank means unassessed or unavailable; 0 means assessed and absent. Image
rows count appearances in one photo; event rows count each person once per
visit (from a multi-photo model request when the event has several photos).
Ages are four categories that add up to the people count: adults, teens,
children and age_unknown (the model's "age_unclear").
"""
from __future__ import annotations

import csv
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

from .prompts import AGE_GROUPS, DIRECTIONS, OBJECTS

# Range labels use an en dash: Excel turns "3-4" / "5-10" into dates.
COUNT_CLASSES = ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 10, "5–10"), (11, None, ">10"))
OBJECT_FIELDS = ("bicycles", "strollers", "motorcycles", "atv_utv", "other_vehicles", "dogs", "backpacks",
                 "large_bags", "large_bags_uncertain")
# Export column for each age group the model answers (prompts.AGE_GROUPS); teens are a category of their own.
AGE_COLUMNS = {"adults": "adults", "teens": "teens", "children": "children", "age_unclear": "age_unknown"}
AGE_FIELDS = tuple(AGE_COLUMNS.values())

IMAGE_FIELDS = ["run_id", "image_id", "relative_path", "sha256", "camera_id", "capture_time", "time_source",
                "event_id", "event_frame", "event_image_count", "status", "error", "cache_hit", "model",
                "prompt_version", "seconds", "requests", "repaired", "consistency_problems",
                "people_count_assessable", "people_total", "people_near", "people_class", "people_near_class",
                *AGE_FIELDS, *[f"dir_{d}" for d in DIRECTIONS], *OBJECT_FIELDS, "scene_uncertain", "notes"]
EVENT_FIELDS = ["run_id", "event_id", "camera_id", "start", "end", "duration_seconds", "image_count", "method",
                "people_unique", "people_max_frame", "people_near_max_frame", "people_near_unique",
                "people_unique_class", "people_near_unique_class", *AGE_FIELDS,
                *[f"dir_{d}" for d in DIRECTIONS], *OBJECT_FIELDS, "status", "error",
                "seconds", "requests", "repaired", "consistency_problems", "needs_review", "review_reasons",
                "images", "notes"]
SUMMED_FIELDS = ["people_max_frame", "people_unique", "people_near_max_frame", "people_near_unique", *AGE_FIELDS,
                 *[f"dir_{d}" for d in DIRECTIONS], *OBJECT_FIELDS]
SUMMARY_FIELDS = ["run_id", "camera_id", "date", "events", "images", *SUMMED_FIELDS, "events_without_unique_count",
                  "events_needing_review"]
IMAGES_CELL_LIMIT = 32000  # Excel cells hold at most 32,767 characters
PEOPLE_FIELDS = ("people_total", "people_near", *AGE_GROUPS)  # answer keys blanked for unassessable photos


def count_class(value):
    if value is None:
        return None
    for low, high, label in COUNT_CLASSES:
        if value >= low and (high is None or value <= high):
            return label
    return None


def _ages(answer):
    """The answer's age groups under their export column names."""
    return {column: answer.get(key) for key, column in AGE_COLUMNS.items()}


def image_row(item, run_id):
    answer = item.get("answer") or {}
    info = item.get("info") or {}
    row = dict.fromkeys(IMAGE_FIELDS)
    row.update(run_id=run_id, image_id=item["image_id"], relative_path=item["relative_path"], sha256=item["sha256"],
               camera_id=item.get("camera_id"), capture_time=item.get("capture_time"),
               time_source=item.get("time_source"), event_id=item.get("event_id"),
               event_frame=item.get("event_frame"), event_image_count=item.get("event_image_count"),
               status=item["status"], error=item.get("error"), cache_hit=item.get("cache_hit"),
               model=item.get("model"), prompt_version=item.get("prompt_version"), seconds=info.get("seconds"),
               requests=info.get("requests"), repaired=info.get("repaired"),
               consistency_problems="; ".join(info.get("problems") or []) or None)
    if item["status"] != "ok":
        return row
    assessable = answer.get("people_count_assessable") is not False and answer.get("people_total") is not None
    row["people_count_assessable"] = assessable
    for key in ("people_total", "people_near", *OBJECTS, "scene_uncertain", "notes"):
        row[key] = answer.get(key)
    row.update(_ages(answer))
    if not assessable:  # blank = not assessed; a contradictory answer must not export counts as 0
        row.update(dict.fromkeys(("people_total", "people_near", *AGE_FIELDS)))
    row["people_class"] = count_class(row["people_total"])
    row["people_near_class"] = count_class(row["people_near"])
    directions = answer.get("direction_counts") if assessable else None
    for d in DIRECTIONS:
        row[f"dir_{d}"] = directions.get(d) if isinstance(directions, dict) else None
    return row


def _photo_people(answer):
    """The photo's answer with blank people fields when the photo was judged not assessable."""
    if answer.get("people_count_assessable") is False or answer.get("people_total") is None:
        return {**answer, **dict.fromkeys(PEOPLE_FIELDS), "direction_counts": None}
    return answer


def _images_cell(paths):
    text = json.dumps(paths, ensure_ascii=False)
    while len(text) > IMAGES_CELL_LIMIT and len(paths) > 1:
        paths = paths[: max(1, len(paths) * 3 // 4)]
        text = json.dumps(paths + [f"... first {len(paths)} listed; see the images CSV"], ensure_ascii=False)
    return text


def _max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def event_row(event, run_id):
    """Event totals from the multi-photo answer, else from the single photo or per-photo maxima."""
    images = [i for i in event["items"] if i["status"] == "ok"]
    answers = [_photo_people(i.get("answer") or {}) for i in images]
    row = dict.fromkeys(EVENT_FIELDS)
    times = sorted(i["capture_time"] for i in event["items"] if i.get("capture_time"))
    row.update(run_id=run_id, event_id=event["event_id"], camera_id=event["items"][0].get("camera_id"),
               start=times[0] if times else None, end=times[-1] if times else None,
               image_count=len(event["items"]), images=_images_cell([i["relative_path"] for i in event["items"]]))
    if times:
        from datetime import datetime
        row["duration_seconds"] = (datetime.fromisoformat(times[-1]) - datetime.fromisoformat(times[0])).total_seconds()
    row["people_max_frame"] = _max(a.get("people_total") for a in answers)
    row["people_near_max_frame"] = _max(a.get("people_near") for a in answers)
    reasons = []
    multi = event.get("answer")
    if multi is not None:
        info = event.get("info") or {}
        row.update(method="multi_photo_model", status="ok", seconds=info.get("seconds"), requests=info.get("requests"),
                   repaired=info.get("repaired"), consistency_problems="; ".join(info.get("problems") or []) or None,
                   people_unique=multi.get("unique_all"), people_near_unique=multi.get("unique_near"),
                   notes=multi.get("notes"))
        row.update(_ages(multi))
        for key in OBJECTS:
            row[key] = multi.get(key)
        directions = multi.get("direction_counts")
        for d in DIRECTIONS:
            row[f"dir_{d}"] = directions.get(d) if isinstance(directions, dict) else None
        if info.get("problems"):
            reasons.append("inconsistent_model_answer")
        # The busiest single photo is a floor for the visit: missing people are added as age unknown and
        # direction unclear (validation: fewer large-group undercounts, better count classes).
        unique, busiest = row["people_unique"], row["people_max_frame"]
        if unique is not None and busiest is not None:
            if unique < busiest:
                reasons.append("unique_below_busiest_photo")
                row["people_unique"] = busiest
                for key in ("age_unknown", "dir_unclear"):
                    if row[key] is not None:
                        row[key] += busiest - unique
                if row["age_unknown"] is None and None not in (row["adults"], row["teens"], row["children"]):
                    row["age_unknown"] = max(0, busiest - row["adults"] - row["teens"] - row["children"])
            elif unique > 2 * busiest and unique - busiest >= 3:
                reasons.append("unique_far_above_busiest_photo")
        if unique is None:
            reasons.append("unique_counts_not_available")
        if row["people_near_unique"] is not None and row["people_near_max_frame"] is not None:
            row["people_near_unique"] = max(row["people_near_unique"], row["people_near_max_frame"])
        if row["people_near_unique"] is not None and row["people_unique"] is not None:
            row["people_near_unique"] = min(row["people_near_unique"], row["people_unique"])
    elif len(images) == 1 and len(event["items"]) == 1:
        single = answers[0]
        info = images[0].get("info") or {}
        row.update(method="single_photo", status="ok", people_unique=single.get("people_total"),
                   people_near_unique=single.get("people_near"), notes=single.get("notes"),
                   consistency_problems="; ".join(info.get("problems") or []) or None)
        if info.get("problems"):
            reasons.append("inconsistent_model_answer")
        row.update(_ages(single))
        for key in OBJECTS:
            row[key] = single.get(key)
        directions = single.get("direction_counts")
        for d in DIRECTIONS:
            row[f"dir_{d}"] = directions.get(d) if isinstance(directions, dict) else None
    else:
        # No multi-photo answer (disabled, failed, or a failed photo): per-photo maxima are a lower bound.
        # Ages and directions come from the busiest photo, so they still add up to people_max_frame.
        row.update(method="per_photo_max", status=event.get("status") or ("ok" if images else "error"),
                   error=event.get("error"), people_unique=None, people_near_unique=None)
        counted = [a for a in answers if a.get("people_total") is not None]
        busiest = max(counted, key=lambda a: a["people_total"]) if counted else {}
        row.update(_ages(busiest))
        directions = busiest.get("direction_counts")
        for d in DIRECTIONS:
            row[f"dir_{d}"] = directions.get(d) if isinstance(directions, dict) else None
        for key in OBJECTS:
            row[key] = _max(a.get(key) for a in answers)
        reasons.append("unique_counts_not_available")
    row["people_unique_class"] = count_class(row["people_unique"])
    row["people_near_unique_class"] = count_class(row["people_near_unique"])
    if len(images) < len(event["items"]):
        reasons.append("photo_analysis_failed")
    if event.get("sampled"):
        reasons.append("event_sampled_for_model")
    if any(i.get("time_source") in (None, "file_mtime") for i in event["items"]) and len(event["items"]) > 1:
        reasons.append("times_from_file_dates")
    row["needs_review"] = bool(reasons)
    row["review_reasons"] = "; ".join(reasons) or None
    return row


def summary_rows(event_rows, run_id):
    groups = defaultdict(list)
    for row in event_rows:
        groups[(row["camera_id"], (row["start"] or "unknown")[:10])].append(row)
    rows = []
    for (camera, date), group in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        out = {"run_id": run_id, "camera_id": camera, "date": date, "events": len(group),
               "images": sum(r["image_count"] for r in group),
               "events_without_unique_count": sum(r.get("people_unique") is None for r in group),
               "events_needing_review": sum(bool(r["needs_review"]) for r in group)}
        # Sums cover the events that have a value; events_without_unique_count says how many lack people_unique.
        for key in SUMMED_FIELDS:
            values = [r.get(key) for r in group]
            out[key] = sum(v for v in values if v is not None) if any(v is not None for v in values) else None
        rows.append(out)
    return rows


def spreadsheet_safe(value):
    if isinstance(value, str) and (value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))):
        return "'" + value
    return value


def write_csv(path, rows, fields, escape=True):
    """Atomic UTF-8 (with BOM) CSV; ``escape=False`` copies values read from an export as they are."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k: spreadsheet_safe(v) if escape else v for k, v in row.items()} for row in rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
