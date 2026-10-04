"""Prompts, JSON schemas and the answer consistency checks (no model)."""
from __future__ import annotations

import copy
import json
import unittest

import fakes_vlm  # noqa: F401  (puts the repository on sys.path)
from fakes_vlm import event_answer, photo_answer

from trailcam_vlm import export, prompts
from trailcam_vlm.prompts import (AGE_GROUPS, DIRECTIONS, EVENT_PROMPT, EVENT_SCHEMA, OBJECTS, PHOTO_PROMPT,
                                  PHOTO_SCHEMA, PROMPT_VERSION, consistency_problems, repair_note)


def mixed_photo(**overrides):
    """3 people: 2 near; 2 adults + 1 child; one each left/right/toward."""
    answer = {"people_total": 3, "people_near": 2, "adults": 2, "teens": 0, "children": 1, "age_unclear": 0,
              "direction_counts": {"left": 1, "right": 1, "toward": 1, "away": 0, "stationary": 0, "unclear": 0}}
    answer.update(overrides)
    return answer


class ConsistencyProblemsTests(unittest.TestCase):
    def test_consistent_photo_answer_has_no_problems(self):
        self.assertEqual(consistency_problems(mixed_photo(), "people_total"), [])
        self.assertEqual(consistency_problems(photo_answer(4), "people_total"), [])

    def test_all_zero_answer_is_consistent(self):
        self.assertEqual(consistency_problems(photo_answer(0), "people_total"), [])

    def test_unknown_total_skips_every_check(self):
        answer = mixed_photo(people_total=None, people_near=9, adults=5)
        self.assertEqual(consistency_problems(answer, "people_total"), [])
        self.assertEqual(consistency_problems({}, "people_total"), [])

    def test_near_above_total_is_a_problem(self):
        problems = consistency_problems(mixed_photo(people_near=4), "people_total")
        self.assertEqual(problems, ["people_near 4 > people_total 3"])

    def test_near_equal_to_total_is_fine(self):
        self.assertEqual(consistency_problems(mixed_photo(people_near=3), "people_total"), [])

    def test_missing_near_is_not_checked(self):
        self.assertEqual(consistency_problems(mixed_photo(people_near=None), "people_total"), [])

    def test_age_groups_must_sum_to_total(self):
        problems = consistency_problems(mixed_photo(adults=3), "people_total")
        self.assertEqual(problems, ["age groups sum to 4, not people_total 3"])
        problems = consistency_problems(mixed_photo(adults=0, children=0), "people_total")
        self.assertEqual(problems, ["age groups sum to 0, not people_total 3"])

    def test_teens_and_unclear_count_towards_the_age_sum(self):
        answer = mixed_photo(adults=0, teens=1, children=1, age_unclear=1)
        self.assertEqual(consistency_problems(answer, "people_total"), [])

    def test_age_check_skipped_when_any_age_group_is_missing(self):
        for key in AGE_GROUPS:
            with self.subTest(missing=key):
                answer = mixed_photo(**{key: None})
                self.assertEqual(consistency_problems(answer, "people_total"), [])
                answer = mixed_photo()
                del answer[key]
                self.assertEqual(consistency_problems(answer, "people_total"), [])

    def test_direction_counts_must_sum_to_total(self):
        directions = {"left": 3, "right": 1, "toward": 0, "away": 0, "stationary": 0, "unclear": 0}
        problems = consistency_problems(mixed_photo(direction_counts=directions), "people_total")
        self.assertEqual(problems, ["direction counts sum to 4, not people_total 3"])

    def test_missing_direction_keys_count_as_zero(self):
        self.assertEqual(consistency_problems(mixed_photo(direction_counts={"left": 3}), "people_total"), [])
        self.assertEqual(consistency_problems(mixed_photo(direction_counts={"left": 2}), "people_total"),
                         ["direction counts sum to 2, not people_total 3"])

    def test_unknown_keys_in_direction_counts_are_ignored(self):
        directions = {"left": 3, "north": 7}
        self.assertEqual(consistency_problems(mixed_photo(direction_counts=directions), "people_total"), [])

    def test_null_or_non_object_direction_counts_are_not_checked(self):
        for value in (None, [1, 2], "left", 3):
            with self.subTest(direction_counts=value):
                self.assertEqual(consistency_problems(mixed_photo(direction_counts=value), "people_total"), [])

    def test_every_problem_is_reported_in_order(self):
        answer = mixed_photo(people_near=5, adults=5,
                             direction_counts={"left": 0, "right": 0, "toward": 0, "away": 0, "stationary": 0,
                                               "unclear": 0})
        self.assertEqual(consistency_problems(answer, "people_total"),
                         ["people_near 5 > people_total 3", "age groups sum to 6, not people_total 3",
                          "direction counts sum to 0, not people_total 3"])

    def test_event_answers_use_the_unique_keys(self):
        self.assertEqual(consistency_problems(event_answer(3, 2), "unique_all"), [])
        answer = event_answer(3, 2, unique_near=4, people_near=99)
        self.assertEqual(consistency_problems(answer, "unique_all"), ["unique_near 4 > unique_all 3"])
        answer = event_answer(3, 2, children=1)
        self.assertEqual(consistency_problems(answer, "unique_all"), ["age groups sum to 4, not unique_all 3"])

    def test_event_answer_without_unique_total_is_not_checked(self):
        self.assertEqual(consistency_problems(event_answer(None, 2, adults=7), "unique_all"), [])

    def test_answer_is_not_modified(self):
        answer = mixed_photo(adults=9)
        before = copy.deepcopy(answer)
        consistency_problems(answer, "people_total")
        self.assertEqual(answer, before)


class RepairNoteTests(unittest.TestCase):
    def test_lists_every_problem_and_names_the_total(self):
        note = repair_note(["people_near 4 > people_total 3", "age groups sum to 4, not people_total 3"],
                           "people_total")
        self.assertIn("people_near 4 > people_total 3; age groups sum to 4, not people_total 3", note)
        self.assertTrue(note.startswith("Your previous JSON was inconsistent: "))
        self.assertIn("each sum to people_total.", note)

    def test_event_note_names_unique_all(self):
        note = repair_note(["direction counts sum to 1, not unique_all 2"], "unique_all")
        self.assertIn("direction counts sum to 1, not unique_all 2", note)
        self.assertIn("each sum to unique_all", note)

    def test_note_of_real_problems_round_trip(self):
        problems = consistency_problems(mixed_photo(adults=5), "people_total")
        self.assertIn(problems[0], repair_note(problems, "people_total"))


class SchemaTests(unittest.TestCase):
    def test_photo_schema_requires_every_property(self):
        self.assertEqual(set(PHOTO_SCHEMA["required"]), set(PHOTO_SCHEMA["properties"]))
        self.assertEqual(len(PHOTO_SCHEMA["required"]), len(set(PHOTO_SCHEMA["required"])))

    def test_event_schema_requires_every_property(self):
        self.assertEqual(set(EVENT_SCHEMA["required"]), set(EVENT_SCHEMA["properties"]))
        self.assertEqual(len(EVENT_SCHEMA["required"]), len(set(EVENT_SCHEMA["required"])))

    def test_photo_schema_fields(self):
        expected = {"people_count_assessable", "people_total", "people_near", *AGE_GROUPS, "direction_counts",
                    *OBJECTS, "scene_uncertain", "notes"}
        self.assertEqual(set(PHOTO_SCHEMA["properties"]), expected)
        self.assertEqual(PHOTO_SCHEMA["type"], "object")

    def test_event_schema_fields(self):
        expected = {"unique_all", "unique_near", "max_in_one_photo", *AGE_GROUPS, "direction_counts", *OBJECTS,
                    "notes"}
        self.assertEqual(set(EVENT_SCHEMA["properties"]), expected)
        self.assertEqual(EVENT_SCHEMA["type"], "object")

    def test_counts_are_nullable_non_negative_integers(self):
        for schema in (PHOTO_SCHEMA, EVENT_SCHEMA):
            for key, spec in schema["properties"].items():
                if key in ("direction_counts", "notes", "people_count_assessable", "scene_uncertain"):
                    continue
                with self.subTest(key=key):
                    self.assertEqual(spec["type"], ["integer", "null"])
                    self.assertEqual(spec["minimum"], 0)

    def test_flags_and_notes_types(self):
        self.assertEqual(PHOTO_SCHEMA["properties"]["people_count_assessable"], {"type": "boolean"})
        self.assertEqual(PHOTO_SCHEMA["properties"]["scene_uncertain"], {"type": "boolean"})
        self.assertEqual(PHOTO_SCHEMA["properties"]["notes"], {"type": "string"})
        self.assertEqual(EVENT_SCHEMA["properties"]["notes"], {"type": "string"})

    def test_direction_counts_schema(self):
        for schema in (PHOTO_SCHEMA, EVENT_SCHEMA):
            spec = schema["properties"]["direction_counts"]
            self.assertEqual(spec["type"], ["object", "null"])
            self.assertEqual(list(spec["properties"]), list(DIRECTIONS))
            self.assertEqual(spec["required"], list(DIRECTIONS))
            for d in DIRECTIONS:
                self.assertEqual(spec["properties"][d], {"type": "integer", "minimum": 0})

    def test_schemas_are_json_serialisable(self):
        for schema in (PHOTO_SCHEMA, EVENT_SCHEMA):
            self.assertEqual(json.loads(json.dumps(schema)), schema)

    def test_fixture_answers_match_the_schemas(self):
        self.assertEqual(set(photo_answer(1)), set(PHOTO_SCHEMA["properties"]))
        self.assertEqual(set(event_answer(1, 1)), set(EVENT_SCHEMA["properties"]))

    def test_categories(self):
        self.assertEqual(DIRECTIONS, ("left", "right", "toward", "away", "stationary", "unclear"))
        self.assertEqual(AGE_GROUPS, ("adults", "teens", "children", "age_unclear"))
        self.assertEqual(tuple(OBJECTS), tuple(export.OBJECT_FIELDS))


class PromptTextTests(unittest.TestCase):
    def test_event_prompt_formats_the_photo_count(self):
        text = EVENT_PROMPT.format(n=4)
        self.assertIn("These 4 photos are consecutive triggers", text)
        self.assertNotIn("{", text)
        self.assertNotIn("}", text)

    def test_photo_prompt_has_no_unformatted_placeholders(self):
        self.assertNotIn("{", PHOTO_PROMPT)
        self.assertNotIn("}", PHOTO_PROMPT)

    def test_prompts_define_every_requested_category(self):
        for text in (PHOTO_PROMPT, EVENT_PROMPT.format(n=3)):
            for word in (*DIRECTIONS, *OBJECTS, "teens", "age_unclear", "direction_counts"):
                with self.subTest(word=word):
                    self.assertIn(word, text)

    def test_prompts_name_their_schema_totals(self):
        for key in ("people_total", "people_near", "people_count_assessable", "scene_uncertain"):
            self.assertIn(key, PHOTO_PROMPT)
        for key in ("unique_all", "unique_near", "max_in_one_photo"):
            self.assertIn(key, EVENT_PROMPT)

    def test_data_bar_and_near_zone_rules_shared(self):
        for text in (PHOTO_PROMPT, EVENT_PROMPT):
            self.assertIn("black data bar", text)
            self.assertIn("NEAR ZONE", text)
            self.assertIn("never the person's own left or right", text)

    def test_prompt_version(self):
        self.assertIsInstance(PROMPT_VERSION, str)
        self.assertTrue(PROMPT_VERSION.strip())
        self.assertIs(prompts.PROMPT_VERSION, PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
