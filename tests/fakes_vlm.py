"""Shared test fixtures for trailcam_vlm: synthetic photos and a fake Ollama server.

Nothing here touches the network or a GPU. ``FakeOllama`` replaces
``requests.get`` / ``requests.post`` (the client imports ``requests`` lazily,
so patching the module attributes intercepts every call) and answers like a
local Ollama server would.

Synthetic photos carry what the fake model "sees" in their pixel size, which
survives JPEG re-encoding: ``width = 300 + people`` and ``height = 200 +
bicycles``. The fake decodes every image it is sent, so tests can check which
photos reached the model and at what size.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import io
import json as jsonlib
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import requests  # noqa: E402
from PIL import Image  # noqa: E402

from trailcam_vlm.prompts import AGE_GROUPS, DIRECTIONS, OBJECTS  # noqa: E402

BASE_WIDTH, BASE_HEIGHT = 300, 200


def make_photo(path, people=0, bicycles=0, when=None, orientation=None, color=(90, 120, 80), quality=95):
    """Write a synthetic photo; ``when`` (datetime or EXIF text) becomes DateTimeOriginal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (BASE_WIDTH + people, BASE_HEIGHT + bicycles), color)
    exif = Image.Exif()
    exif[0x010F] = "FakeTrailCam"  # Make; Pillow drops PNG EXIF that holds only the Exif sub-IFD
    if when is not None:
        text = when.strftime("%Y:%m:%d %H:%M:%S") if isinstance(when, datetime) else str(when)
        exif.get_ifd(0x8769)[0x9003] = text
    if orientation is not None:
        exif[0x0112] = orientation
    options = {"exif": exif}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        options["quality"] = quality
    image.save(path, **options)
    return path


def set_mtime(path, moment):
    stamp = moment.timestamp()
    os.utime(path, (stamp, stamp))


def decode_image(image_b64):
    """(people, bicycles, (width, height), format) encoded in a synthetic photo sent to the model."""
    with Image.open(io.BytesIO(base64.b64decode(image_b64))) as image:
        width, height = image.size
        return width - BASE_WIDTH, height - BASE_HEIGHT, (width, height), image.format


def photo_answer(people=0, **overrides):
    """A consistent per-photo answer: everyone an adult walking left."""
    answer = {"people_count_assessable": True, "people_total": people, "people_near": people,
              "adults": people, "teens": 0, "children": 0, "age_unclear": 0,
              "direction_counts": {d: (people if d == "left" else 0) for d in DIRECTIONS},
              **{k: 0 for k in OBJECTS}, "scene_uncertain": False, "notes": ""}
    answer.update(overrides)
    return answer


def event_answer(unique=0, max_in_one_photo=0, **overrides):
    """A consistent multi-photo answer: every unique person an adult walking left."""
    answer = {"unique_all": unique, "unique_near": unique, "max_in_one_photo": max_in_one_photo,
              "adults": unique, "teens": 0, "children": 0, "age_unclear": 0,
              "direction_counts": {d: (unique if d == "left" else 0) for d in DIRECTIONS},
              **{k: 0 for k in OBJECTS}, "notes": ""}
    answer.update(overrides)
    return answer


assert set(AGE_GROUPS) == {"adults", "teens", "children", "age_unclear"}


class FakeResponse:
    """The parts of ``requests.Response`` the client uses."""

    def __init__(self, payload=None, status_code=200, json_error=None):
        self._payload = payload
        self.status_code = status_code
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return copy.deepcopy(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error", response=self)


def chat_response(content):
    """An /api/chat reply carrying ``content`` (a string, or anything JSON-serialisable)."""
    if not isinstance(content, str):
        content = jsonlib.dumps(content)
    return FakeResponse({"model": "fake", "message": {"role": "assistant", "content": content},
                         "done": True, "done_reason": "stop"})


def tags_response(*names):
    return FakeResponse({"models": [{"name": n, "model": n} for n in names]})


class FakeOllama:
    """In-process stand-in for a local Ollama server (``/api/tags`` and ``/api/chat``).

    By default photo requests are answered from the synthetic photo size and
    event requests with ``unique_all = busiest photo + 1`` (0 when nobody is
    visible). Set ``photo_reply(people, bicycles, body)`` or
    ``event_reply(contents, body)`` to return a custom answer (dict or string),
    a ``FakeResponse``, an exception to raise, or ``None`` for the default.
    """

    def __init__(self, models=("gemma4:26b",), tags_error=None):
        self.models = list(models)
        self.tags_error = tags_error
        self.tag_requests = []
        self.ps_requests = []
        self.vram_share = 1.0  # share of each loaded model in GPU memory reported by /api/ps (None: no report)
        self.unload_requests = []
        self.on_unload = None  # called on an unload request (e.g. to simulate a reload into GPU memory)
        self.chat_requests = []
        self.photo_reply = None
        self.event_reply = None

    # -- requests replacements -------------------------------------------------
    def get(self, url, timeout=None, **kwargs):
        if url.endswith("/api/ps"):
            self.ps_requests.append({"url": url, "timeout": timeout})
            if self.vram_share is None:
                return FakeResponse({"models": []})
            size = 18_000_000_000
            return FakeResponse({"models": [{"name": m, "model": m, "size": size,
                                             "size_vram": int(size * self.vram_share)} for m in self.models]})
        self.tag_requests.append({"url": url, "timeout": timeout})
        if self.tags_error is not None:
            raise self.tags_error
        return tags_response(*self.models)

    def post(self, url, json=None, timeout=None, **kwargs):
        body = copy.deepcopy(json)
        if url.endswith("/api/generate"):
            self.unload_requests.append({"url": url, "body": body})
            if self.on_unload:
                self.on_unload()
            return FakeResponse({"model": body.get("model"), "done": True, "done_reason": "unload"})
        images = [decode_image(i) for i in body["messages"][0].get("images", [])]
        is_event = "unique_all" in body["format"]["properties"]
        self.chat_requests.append({"url": url, "timeout": timeout, "body": body, "images": images,
                                   "kind": "event" if is_event else "photo"})
        if is_event:
            reply = self.event_reply(images, body) if self.event_reply else None
            if reply is None:
                busiest = max(p for p, *_ in images)
                reply = event_answer(busiest + 1 if busiest else 0, busiest,
                                     bicycles=max(b for _, b, *_ in images))
        else:
            people, bicycles = images[0][0], images[0][1]
            reply = self.photo_reply(people, bicycles, body) if self.photo_reply else None
            if reply is None:
                reply = photo_answer(people, bicycles=bicycles)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, FakeResponse):
            return reply
        return chat_response(reply)

    # -- helpers -----------------------------------------------------------------
    @property
    def photo_requests(self):
        return [r for r in self.chat_requests if r["kind"] == "photo"]

    @property
    def event_requests(self):
        return [r for r in self.chat_requests if r["kind"] == "event"]

    def reset(self):
        self.tag_requests.clear()
        self.ps_requests.clear()
        self.unload_requests.clear()
        self.chat_requests.clear()

    @contextlib.contextmanager
    def installed(self):
        from trailcam_vlm.ollama_client import Ollama
        with mock.patch.object(requests, "get", self.get), mock.patch.object(requests, "post", self.post),                 mock.patch.object(Ollama, "retry_pause", 0):
            yield self
