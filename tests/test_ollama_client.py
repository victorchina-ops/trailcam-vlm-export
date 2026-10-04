"""Ollama client: model-name matching, JPEG encoding and the structured chat request.

``requests.get`` / ``requests.post`` are always replaced; no server is contacted.
"""
from __future__ import annotations

import base64
import io
import json
import types
import unittest
from unittest import mock

import fakes_vlm  # noqa: F401  (puts the repository on sys.path)
import requests
from fakes_vlm import FakeResponse, chat_response, event_answer, photo_answer, tags_response
from PIL import Image

from trailcam_vlm.ollama_client import Ollama, OllamaError, encode_jpeg, normalize_model_name
from trailcam_vlm.prompts import PHOTO_SCHEMA, consistency_problems, repair_note

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


class NormalizeModelNameTests(unittest.TestCase):
    def test_bare_name_gets_latest_tag(self):
        self.assertEqual(normalize_model_name("gemma4"), "gemma4:latest")

    def test_explicit_tag_is_kept(self):
        self.assertEqual(normalize_model_name("gemma4:26b"), "gemma4:26b")
        self.assertEqual(normalize_model_name("gemma4:latest"), "gemma4:latest")

    def test_case_and_whitespace_are_ignored(self):
        self.assertEqual(normalize_model_name("  Gemma4:26B \n"), "gemma4:26b")
        self.assertEqual(normalize_model_name("Qwen3-VL"), "qwen3-vl:latest")

    def test_registry_port_is_not_mistaken_for_a_tag(self):
        self.assertEqual(normalize_model_name("registry.local:5000/library/gemma4"),
                         "registry.local:5000/library/gemma4:latest")
        self.assertEqual(normalize_model_name("registry.local:5000/library/gemma4:26b"),
                         "registry.local:5000/library/gemma4:26b")

    def test_namespaced_names(self):
        self.assertEqual(normalize_model_name("hf.co/user/model"), "hf.co/user/model:latest")
        self.assertEqual(normalize_model_name("hf.co/user/model:Q4_K_M"), "hf.co/user/model:q4_k_m")

    def test_empty_names(self):
        self.assertEqual(normalize_model_name(None), ":latest")
        self.assertEqual(normalize_model_name(""), ":latest")


class EncodeJpegTests(unittest.TestCase):
    @staticmethod
    def decode(text):
        raw = base64.b64decode(text)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return raw, image

    def test_long_side_is_capped_and_aspect_kept(self):
        image = Image.new("RGB", (4000, 3000), (10, 200, 30))
        raw, decoded = self.decode(encode_jpeg(image, 1000))
        self.assertTrue(raw.startswith(b"\xff\xd8"))
        self.assertEqual(decoded.format, "JPEG")
        self.assertEqual(decoded.size, (1000, 750))

    def test_portrait_photo(self):
        _, decoded = self.decode(encode_jpeg(Image.new("RGB", (600, 1200)), 300))
        self.assertEqual(decoded.size, (150, 300))

    def test_small_photo_is_not_enlarged(self):
        _, decoded = self.decode(encode_jpeg(Image.new("RGB", (320, 240)), 2048))
        self.assertEqual(decoded.size, (320, 240))

    def test_original_is_not_modified(self):
        image = Image.new("RGB", (800, 600), (1, 2, 3))
        encode_jpeg(image, 100)
        self.assertEqual(image.size, (800, 600))

    def test_result_is_ascii_base64(self):
        text = encode_jpeg(Image.new("RGB", (64, 64), (255, 0, 0)), 64)
        self.assertIsInstance(text, str)
        text.encode("ascii")
        self.assertEqual(base64.b64encode(base64.b64decode(text)).decode("ascii"), text)

    def test_quality_changes_the_encoding(self):
        image = Image.effect_noise((256, 256), 64).convert("RGB")
        low = len(base64.b64decode(encode_jpeg(image, 256, quality=20)))
        high = len(base64.b64decode(encode_jpeg(image, 256, quality=95)))
        self.assertLess(low, high)

    def test_colours_survive(self):
        _, decoded = self.decode(encode_jpeg(Image.new("RGB", (64, 64), (200, 30, 40)), 64))
        for got, want in zip(decoded.convert("RGB").getpixel((32, 32)), (200, 30, 40)):
            self.assertAlmostEqual(got, want, delta=6)


class AvailableTests(unittest.TestCase):
    def check(self, model, response=None, side_effect=None, host="http://localhost:11434"):
        get = mock.Mock(return_value=response, side_effect=side_effect)
        with mock.patch.object(requests, "get", get), \
                mock.patch.object(requests, "post", side_effect=AssertionError("no chat expected")):
            result = Ollama(model, host).available()
        return result, get

    def test_listed_model_is_available(self):
        result, get = self.check("gemma4:26b", tags_response("llava:7b", "gemma4:26b"))
        self.assertTrue(result)
        get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)

    def test_bare_name_matches_the_latest_tag(self):
        self.assertTrue(self.check("gemma4", tags_response("gemma4:latest"))[0])

    def test_latest_tag_matches_a_bare_listing(self):
        self.assertTrue(self.check("gemma4:latest", FakeResponse({"models": [{"name": "gemma4"}]}))[0])

    def test_names_are_case_insensitive(self):
        self.assertTrue(self.check("Gemma4:26B", tags_response("gemma4:26b"))[0])

    def test_model_field_is_also_matched(self):
        listing = FakeResponse({"models": [{"name": "my-alias", "model": "gemma4:26b"}]})
        self.assertTrue(self.check("gemma4:26b", listing)[0])
        listing = FakeResponse({"models": [{"model": "gemma4:26b"}]})
        self.assertTrue(self.check("gemma4:26b", listing)[0])

    def test_other_tag_of_the_same_model_does_not_match(self):
        self.assertFalse(self.check("gemma4:26b", tags_response("gemma4:latest", "gemma4:e4b"))[0])
        self.assertFalse(self.check("gemma4", tags_response("gemma4:26b"))[0])

    def test_unlisted_model_is_unavailable(self):
        self.assertFalse(self.check("gemma4:26b", tags_response())[0])
        self.assertFalse(self.check("gemma4:26b", FakeResponse({}))[0])

    def test_non_dict_json_is_unavailable(self):
        for payload in ([{"name": "gemma4:26b"}], "gemma4:26b", None, 42):
            with self.subTest(payload=payload):
                self.assertFalse(self.check("gemma4:26b", FakeResponse(payload))[0])

    def test_malformed_model_list_is_unavailable(self):
        for models in (None, "gemma4:26b", 5):
            with self.subTest(models=models):
                self.assertFalse(self.check("gemma4:26b", FakeResponse({"models": models}))[0])

    def test_non_dict_entries_are_skipped(self):
        listing = FakeResponse({"models": ["gemma4:26b", None, 3, {"name": "gemma4:26b"}]})
        self.assertTrue(self.check("gemma4:26b", listing)[0])
        listing = FakeResponse({"models": ["gemma4:26b", ["gemma4:26b"]]})
        self.assertFalse(self.check("gemma4:26b", listing)[0])

    def test_connection_problems_mean_unavailable(self):
        for error in (requests.ConnectionError("refused"), requests.Timeout("slow"), OSError("socket")):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(self.check("gemma4:26b", side_effect=error)[0])

    def test_invalid_json_means_unavailable(self):
        response = FakeResponse(json_error=ValueError("Expecting value"))
        self.assertFalse(self.check("gemma4:26b", response)[0])

    def test_trailing_slash_in_host_is_removed(self):
        _, get = self.check("gemma4:26b", tags_response("gemma4:26b"), host="http://gpu-box:11434/")
        get.assert_called_once_with("http://gpu-box:11434/api/tags", timeout=5)


class ChatJsonTests(unittest.TestCase):
    """``chat_json`` with a scripted sequence of server replies."""

    def chat(self, replies, client=None, **kwargs):
        post = mock.Mock(side_effect=list(replies))
        client = client or Ollama("gemma4:26b", "http://localhost:11434/", timeout=42.0)
        client.retry_pause = 0
        kwargs.setdefault("schema", SCHEMA)
        with mock.patch.object(requests, "post", post), \
                mock.patch.object(requests, "get", side_effect=AssertionError("no tags request expected")):
            result = client.chat_json(kwargs.pop("prompt", "Count."), kwargs.pop("images", ["aW1n"]),
                                      kwargs.pop("schema"), **kwargs)
        return result, post

    @staticmethod
    def bodies(post):
        return [c.kwargs["json"] for c in post.call_args_list]

    def test_request_body(self):
        (answer, info), post = self.chat([chat_response({"n": 2})], prompt="How many?", images=("aaa", "bbb"),
                                         schema=PHOTO_SCHEMA, max_tokens=321)
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args, ("http://localhost:11434/api/chat",))
        self.assertEqual(kwargs["timeout"], 42.0)
        body = kwargs["json"]
        self.assertEqual(body["model"], "gemma4:26b")
        self.assertIs(body["stream"], False)
        self.assertIs(body["think"], False)
        self.assertEqual(body["format"], PHOTO_SCHEMA)
        self.assertEqual(body["messages"], [{"role": "user", "content": "How many?", "images": ["aaa", "bbb"]}])
        self.assertEqual(body["options"], {"temperature": 0, "num_ctx": 16384, "num_predict": 321})
        json.dumps(body)  # the body must be serialisable as JSON

    def test_default_answer_budget(self):
        _, post = self.chat([chat_response({"n": 2})])
        self.assertEqual(self.bodies(post)[0]["options"]["num_predict"], 700)

    def test_custom_context_and_model(self):
        client = Ollama("qwen3-vl:8b", "http://h:1", timeout=5, num_ctx=4096)
        _, post = self.chat([chat_response({"n": 0})], client=client)
        body = self.bodies(post)[0]
        self.assertEqual(body["model"], "qwen3-vl:8b")
        self.assertEqual(body["options"]["num_ctx"], 4096)
        self.assertEqual(post.call_args.args, ("http://h:1/api/chat",))
        self.assertEqual(post.call_args.kwargs["timeout"], 5)

    def test_returns_the_answer_and_request_info(self):
        (answer, info), _ = self.chat([chat_response({"n": 2, "extra": [1]})])
        self.assertEqual(answer, {"n": 2, "extra": [1]})
        self.assertEqual(info["requests"], 1)
        self.assertIs(info["repaired"], False)
        self.assertEqual(info["problems"], [])
        self.assertIsInstance(info["seconds"], float)
        self.assertGreaterEqual(info["seconds"], 0.0)

    def test_seconds_add_up_over_requests(self):
        clock = iter([10.0, 12.5, 20.0, 21.0])
        fake_time = types.SimpleNamespace(perf_counter=lambda: next(clock))
        with mock.patch("trailcam_vlm.ollama_client.time", fake_time):
            (_, info), _ = self.chat([chat_response("{oops"), chat_response({"n": 1})])
        self.assertEqual(info["seconds"], 3.5)
        self.assertEqual(info["requests"], 2)

    def test_unparseable_answer_is_retried_once_with_a_bigger_budget(self):
        (answer, info), post = self.chat([chat_response('{"n": 2'), chat_response({"n": 2})], max_tokens=700)
        self.assertEqual(answer, {"n": 2})
        self.assertEqual(info["requests"], 2)
        self.assertIs(info["repaired"], False)
        first, second = self.bodies(post)
        self.assertEqual(first["options"]["num_predict"], 700)
        self.assertEqual(second["options"]["num_predict"], 1050)
        self.assertEqual(second["messages"], first["messages"])  # same question, nothing appended

    def test_empty_answer_counts_as_unparseable(self):
        (answer, info), post = self.chat([chat_response(""), chat_response({"n": 0})])
        self.assertEqual(answer, {"n": 0})
        self.assertEqual(post.call_count, 2)

    def test_non_object_json_counts_as_unparseable(self):
        for content in ("[1, 2]", "3", '"text"', "null"):
            with self.subTest(content=content):
                (answer, info), post = self.chat([chat_response(content), chat_response({"n": 1})])
                self.assertEqual(answer, {"n": 1})
                self.assertEqual(post.call_count, 2)

    def test_two_unparseable_answers_raise(self):
        with self.assertRaises(OllamaError) as caught:
            self.chat([chat_response("not json"), chat_response("[1]")])
        self.assertIn("unparseable answer", str(caught.exception))
        self.assertIn("not a JSON object", str(caught.exception))

    def test_only_one_retry_for_unparseable_answers(self):
        post = mock.Mock(side_effect=[chat_response("x"), chat_response("y"), chat_response({"n": 1})])
        with mock.patch.object(requests, "post", post):
            with self.assertRaises(OllamaError):
                Ollama().chat_json("p", [], SCHEMA)
        self.assertEqual(post.call_count, 2)

    def test_inconsistent_answer_is_repaired_once(self):
        first = photo_answer(3, adults=2)  # ages sum to 2, not 3
        fixed = photo_answer(3)
        check = mock.Mock(side_effect=lambda a: consistency_problems(a, "people_total"))
        repair = mock.Mock(side_effect=lambda p: repair_note(p, "people_total"))
        (answer, info), post = self.chat([chat_response(first), chat_response(fixed)], prompt="Count people.",
                                         images=["img1"], schema=PHOTO_SCHEMA, check=check, repair=repair)
        self.assertEqual(answer, fixed)
        self.assertEqual(info["requests"], 2)
        self.assertIs(info["repaired"], True)
        self.assertEqual(info["problems"], [])
        repair.assert_called_once_with(["age groups sum to 2, not people_total 3"])
        self.assertEqual(check.call_count, 2)
        first_body, second_body = self.bodies(post)
        self.assertEqual(first_body["messages"], [{"role": "user", "content": "Count people.", "images": ["img1"]}])
        self.assertEqual(second_body["messages"], [
            {"role": "user", "content": "Count people.", "images": ["img1"]},
            {"role": "assistant", "content": json.dumps(first)},
            {"role": "user", "content": repair_note(["age groups sum to 2, not people_total 3"], "people_total")},
        ])
        self.assertEqual(second_body["options"]["num_predict"], first_body["options"]["num_predict"])
        self.assertEqual(second_body["format"], PHOTO_SCHEMA)

    def test_assistant_turn_repeats_the_raw_answer_text(self):
        raw = '{"n":  5}'
        check = lambda a: ["n too big"] if a["n"] > 4 else []  # noqa: E731
        _, post = self.chat([chat_response(raw), chat_response({"n": 4})], check=check, repair=lambda p: "fix")
        self.assertEqual(self.bodies(post)[1]["messages"][1], {"role": "assistant", "content": raw})

    def test_still_inconsistent_after_repair_returns_the_second_answer(self):
        check = lambda a: [f"bad {a['n']}"]  # noqa: E731
        (answer, info), post = self.chat([chat_response({"n": 1}), chat_response({"n": 2})], check=check,
                                         repair=lambda p: "fix: " + "; ".join(p))
        self.assertEqual(post.call_count, 2)
        self.assertEqual(answer, {"n": 2})
        self.assertIs(info["repaired"], True)
        self.assertEqual(info["problems"], ["bad 2"])

    def test_no_repair_without_a_repair_message(self):
        (answer, info), post = self.chat([chat_response({"n": 1})], check=lambda a: ["bad"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(info["problems"], ["bad"])
        self.assertIs(info["repaired"], False)

    def test_consistent_answer_needs_one_request(self):
        (answer, info), post = self.chat([chat_response(event_answer(2, 1))],
                                         check=lambda a: consistency_problems(a, "unique_all"),
                                         repair=lambda p: repair_note(p, "unique_all"))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(info["problems"], [])

    def test_unparseable_then_inconsistent_is_returned_without_repair(self):
        (answer, info), post = self.chat([chat_response("{"), chat_response({"n": 9})], check=lambda a: ["bad"],
                                         repair=lambda p: "fix")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(answer, {"n": 9})
        self.assertEqual(info["problems"], ["bad"])
        self.assertIs(info["repaired"], False)

    def test_unparseable_repair_keeps_the_first_answer(self):
        (answer, info), post = self.chat([chat_response({"n": 9}), chat_response("garbled")],
                                         check=lambda a: ["bad"], repair=lambda p: "fix")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(answer, {"n": 9})
        self.assertIs(info["repaired"], True)
        self.assertEqual(info["problems"], ["bad"])

    def test_images_are_copied_into_the_request(self):
        images = ["a", "b"]
        _, post = self.chat([chat_response({"n": 1})], images=images)
        sent = self.bodies(post)[0]["messages"][0]["images"]
        self.assertEqual(sent, images)
        self.assertIsNot(sent, images)

    def test_http_error_status_raises_ollama_error(self):
        for status in (400, 404, 500, 503):
            with self.subTest(status=status):
                failure = FakeResponse({"error": "boom"}, status_code=status)
                with self.assertRaises(OllamaError) as caught:
                    self.chat([failure, failure, chat_response({"n": 1})])
                self.assertIn("HTTPError", str(caught.exception))
                self.assertIsInstance(caught.exception.__cause__, requests.HTTPError)

    def test_client_errors_are_not_retried(self):
        for status in (400, 404):
            with self.subTest(status=status):
                post = mock.Mock(side_effect=[FakeResponse({}, status_code=status), chat_response({"n": 1})])
                with mock.patch.object(requests, "post", post), mock.patch.object(Ollama, "retry_pause", 0):
                    with self.assertRaises(OllamaError):
                        Ollama().chat_json("p", [], SCHEMA)
                self.assertEqual(post.call_count, 1)

    def test_server_errors_and_dropped_connections_are_retried_once(self):
        for first in (FakeResponse({"error": "llama runner process has terminated"}, status_code=500),
                      requests.ConnectionError("reset")):
            with self.subTest(first=type(first).__name__):
                (answer, info), post = self.chat([first, chat_response({"n": 1})])
                self.assertEqual(answer, {"n": 1})
                self.assertEqual((post.call_count, info["requests"]), (2, 2))

    def test_timeouts_are_not_retried(self):
        with self.assertRaises(OllamaError):
            self.chat([requests.Timeout("read timed out"), chat_response({"n": 1})])

    def test_connection_errors_raise_ollama_error(self):
        for error in (requests.ConnectionError("refused"), requests.Timeout("read timed out"),
                      requests.exceptions.ChunkedEncodingError("cut")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(OllamaError) as caught:
                    self.chat([error, error])  # a dropped connection is retried once
                self.assertIn(type(error).__name__, str(caught.exception))

    def test_response_without_message_raises_ollama_error(self):
        for payload in ({}, {"message": {}}, {"error": "model not found"}):
            with self.subTest(payload=payload):
                with self.assertRaises(OllamaError) as caught:
                    self.chat([FakeResponse(payload)])
                self.assertIn("KeyError", str(caught.exception))

    def test_invalid_response_json_raises_ollama_error(self):
        errors = (ValueError("Expecting value"),
                  requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(OllamaError):
                    self.chat([FakeResponse(json_error=error)])

    def test_error_in_repair_request_keeps_the_first_answer(self):
        gone = requests.ConnectionError("gone")
        (answer, info), post = self.chat([chat_response({"n": 1}), gone, gone], check=lambda a: ["bad"],
                                         repair=lambda p: "fix")
        self.assertEqual(answer, {"n": 1})
        self.assertEqual(info["problems"], ["bad"])
        self.assertEqual(post.call_count, 3)

    # The docstring contract: OllamaError (not TypeError) when no parseable answer arrives.
    def test_non_object_response_body_raises_ollama_error(self):
        with self.assertRaises(OllamaError):
            self.chat([FakeResponse([{"message": {"content": "{}"}}])])

    def test_null_message_content_raises_ollama_error(self):
        with self.assertRaises(OllamaError):
            self.chat([FakeResponse({"message": {"role": "assistant", "content": None}}),
                       FakeResponse({"message": {"role": "assistant", "content": None}})])


if __name__ == "__main__":
    unittest.main()
