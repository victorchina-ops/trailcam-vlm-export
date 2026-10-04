"""Prompts and JSON schemas sent to the local vision-language model.

The definitions follow the trailcam-expert-export validation labels, so the
two tools can be scored on the same labels. Bump PROMPT_VERSION whenever a
prompt or schema changes: cached answers are keyed by it.
"""
from __future__ import annotations

PROMPT_VERSION = "vlm_full_v1"
DIRECTIONS = ("left", "right", "toward", "away", "stationary", "unclear")
AGE_GROUPS = ("adults", "teens", "children", "age_unclear")
OBJECTS = ("bicycles", "strollers", "motorcycles", "atv_utv", "other_vehicles", "dogs",
           "backpacks", "large_bags", "large_bags_uncertain")

_COMMON = """\
The black data bar at the bottom of a photo (date, temperature, camera name) is not scene content.
PEOPLE: count every visible real person once, including riders, partly hidden people, distant background people and
carried infants; scan left to right once; do not count shadows, reflections, pictures or statues.
NEAR ZONE: people passing through the near and middle ground that the camera monitors (roughly within 25 m); exclude
distant background crowds far behind them.
AGE: child = clearly a pre-teen child (roughly under 12, including toddlers and carried babies), judged from body
proportions (head size relative to body, limb length), height compared with nearby adults at a similar distance, and
context (held by the hand, carried, in a stroller or baby carrier); teens = adolescents; adults = grown-ups;
age_unclear = too small, dark, blurred or hidden to judge. Do not guess.
DIRECTION of travel relative to the image: left = toward the LEFT edge of the image as seen by the viewer, right = toward
the RIGHT edge (never the person's own left or right), toward = approaching the camera, away = receding from it,
stationary = standing or sitting still, unclear = cannot tell. Use the dominant component for diagonal movement.
OBJECTS: count objects, not their users. bicycles = ridden or parked bicycles and e-bikes. strollers = baby
strollers/prams/pushchairs (not wheelchairs or carts). motorcycles = motorbikes, dirt bikes, motor scooters.
atv_utv = quad bikes/ATVs, side-by-side off-road vehicles (RZR/Ranger/buggy), golf carts and small utility carts.
other_vehicles = cars, jeeps, pickups, trucks, buses, tractors. dogs = dogs only (no wildlife). backpacks = all
visible backpacks. large_bags = clearly bulky trekking or frame packs (e.g. rising above the shoulders or reaching
the hips, sleeping mat attached), luggage or large duffel bags; ordinary daypacks, handbags and shopping or garbage
bags are not large. large_bags_uncertain = bags whose large/small size is ambiguous (never also in large_bags).
Use 0 for categories checked and absent. Do not invent hidden people or objects."""

PHOTO_PROMPT = f"""\
Analyze this trail-camera photograph using only visible evidence and return JSON matching the schema.
{_COMMON}
people_total = all visible people; people_near = people in the near zone (people_near <= people_total).
adults + teens + children + age_unclear must equal people_total.
The six direction_counts must sum to people_total; in a single photo judge direction from gait, posture, facing and
bicycle orientation.
Set people_count_assessable false only when the photo is too dark, blurred or occluded to count people at all; then
set the people, age and direction fields to null.
scene_uncertain = true when darkness, crowding, blur, occlusion or cropping makes any count uncertain.
notes: at most one short sentence about material ambiguity, otherwise an empty string."""

EVENT_PROMPT = """\
These {n} photos are consecutive triggers of the SAME fixed trail camera, taken seconds apart, in time order
(photo 1 first). They show one visit. Count UNIQUE people and objects across all the photos: link the same person
across photos by clothing, bags, hats, size and plausible movement, and count each person ONCE. People who appear in
only one photo still count.
""" + _COMMON + """
unique_all = all unique people in any photo, including distant background people; unique_near = unique people in the
near zone (unique_near <= unique_all); max_in_one_photo = the largest number of people visible in a single photo.
adults + teens + children + age_unclear must equal unique_all.
Direction: judge each unique person's movement across the photos (position and size change between photos; for people
seen once, from gait and facing). The six direction_counts must sum to unique_all.
Objects: the number of DIFFERENT objects across the whole visit (not the sum over photos).
notes: at most one short sentence about material ambiguity, otherwise an empty string."""

_COUNT = {"type": ["integer", "null"], "minimum": 0}
_DIRECTION_COUNTS = {"type": ["object", "null"],
                     "properties": {d: {"type": "integer", "minimum": 0} for d in DIRECTIONS},
                     "required": list(DIRECTIONS)}

PHOTO_SCHEMA = {
    "type": "object",
    "properties": {
        "people_count_assessable": {"type": "boolean"},
        "people_total": _COUNT, "people_near": _COUNT,
        **{k: _COUNT for k in AGE_GROUPS},
        "direction_counts": _DIRECTION_COUNTS,
        **{k: _COUNT for k in OBJECTS},
        "scene_uncertain": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["people_count_assessable", "people_total", "people_near", *AGE_GROUPS, "direction_counts",
                 *OBJECTS, "scene_uncertain", "notes"],
}

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "unique_all": _COUNT, "unique_near": _COUNT, "max_in_one_photo": _COUNT,
        **{k: _COUNT for k in AGE_GROUPS},
        "direction_counts": _DIRECTION_COUNTS,
        **{k: _COUNT for k in OBJECTS},
        "notes": {"type": "string"},
    },
    "required": ["unique_all", "unique_near", "max_in_one_photo", *AGE_GROUPS, "direction_counts", *OBJECTS,
                 "notes"],
}


def _count(value):
    """A non-negative integer count, or None for anything else (null, negative, text, booleans)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    return value if isinstance(value, int) and value >= 0 else None


def clean_answer(answer, schema):
    """The answer reduced to the schema's fields with checked types, or None when it is not an object.

    Counts become non-negative integers or None, direction_counts a dict of six
    integers or None, flags booleans or None and notes a string, so a malformed
    answer (fresh or cached) can never break the export.
    """
    if not isinstance(answer, dict):
        return None
    clean = {}
    for key in schema["properties"]:
        value = answer.get(key)
        if key == "direction_counts":
            # A missing direction counts as 0 (the sum check then asks for a correction); an invalid value
            # (text, negative) blanks all directions rather than exporting 0.
            counts = {d: _count(value[d]) if d in value else 0 for d in DIRECTIONS} if isinstance(value, dict) else {}
            clean[key] = counts if counts and None not in counts.values() else None
        elif key in ("people_count_assessable", "scene_uncertain"):
            clean[key] = value if isinstance(value, bool) else None
        elif key == "notes":
            clean[key] = value.strip() if isinstance(value, str) else ""
        else:
            clean[key] = _count(value)
    return clean


def consistency_problems(answer, total_key):
    """Sum/ordering problems in an answer (empty list when consistent)."""
    problems = []
    answer = clean_answer(answer, PHOTO_SCHEMA if total_key == "people_total" else EVENT_SCHEMA) or {}
    total = answer.get(total_key)
    if total is None:
        return problems
    near_key = "people_near" if total_key == "people_total" else "unique_near"
    if answer.get(near_key) is not None and answer[near_key] > total:
        problems.append(f"{near_key} {answer[near_key]} > {total_key} {total}")
    ages = [answer.get(k) for k in AGE_GROUPS]
    if None not in ages and sum(ages) != total:
        problems.append(f"age groups sum to {sum(ages)}, not {total_key} {total}")
    directions = answer.get("direction_counts")
    if isinstance(directions, dict) and sum(directions.get(d, 0) for d in DIRECTIONS) != total:
        problems.append(f"direction counts sum to {sum(directions.get(d, 0) for d in DIRECTIONS)}, not {total_key} {total}")
    return problems


def repair_note(problems, total_key):
    return ("Your previous JSON was inconsistent: " + "; ".join(problems) +
            f". Return corrected JSON in which the age groups and the six direction counts each sum to {total_key}.")
