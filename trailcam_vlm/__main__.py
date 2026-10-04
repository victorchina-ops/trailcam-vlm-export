"""Analyze a folder of trail-camera photos with a local vision-language model (Ollama).

Usage examples:
    python -m trailcam_vlm "D:\\Trail Photos\\Camera 1"
    python -m trailcam_vlm --input images --limit 20          # quick test on 20 photos
    python -m trailcam_vlm --model qwen3-vl:30b-a3b --no-events
    python -m trailcam_vlm --profile cpu "D:\\Trail Photos"     # computer without a suitable GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .export import EVENT_FIELDS, IMAGE_FIELDS, SUMMARY_FIELDS, event_row, image_row, summary_rows, write_csv
from .ollama_client import Ollama, OllamaError, encode_jpeg
from .photos import camera_id_for, capture_time, discover, file_hash, group_events, open_rgb, sequence_number
from .prompts import (EVENT_PROMPT, EVENT_SCHEMA, PHOTO_PROMPT, PHOTO_SCHEMA, PROMPT_VERSION, clean_answer,
                      consistency_problems, repair_note)

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"input_dir": "images", "output_dir": "exports", "cache_dir": ".cache", "model": "gemma4:26b",
            "ollama_host": "http://localhost:11434", "max_side": 2048, "event_max_side": 1280,
            "max_event_frames": 6, "event_gap_seconds": 60, "camera_id_mode": "folder",
            "camera_id_pattern": None, "events": True, "timeout": 600, "recursive": True,
            "device": "auto", "num_ctx": 16384, "num_threads": None, "keep_alive": None}
NUMBER = (int, float)
DEVICES = ("auto", "gpu", "cpu")
SETTING_TYPES = {"input_dir": str, "output_dir": str, "cache_dir": str, "model": str, "ollama_host": str,
                 "max_side": int, "event_max_side": int, "max_event_frames": int, "event_gap_seconds": NUMBER,
                 "camera_id_mode": ("folder", "regex", "single"), "camera_id_pattern": (str, type(None)),
                 "events": bool, "timeout": NUMBER, "recursive": bool, "device": DEVICES, "num_ctx": int,
                 "num_threads": (int, type(None)), "keep_alive": (str, int, type(None))}
# Ollama keep_alive: whole seconds ("3600", "-1" = until Ollama stops) or a duration with units ("30m", "1h30m")
KEEP_ALIVE = re.compile(r"^-?\d+$|^-?(\d+(\.\d+)?(ns|us|µs|ms|s|m|h))+$")
PROFILE_DIR = "profiles"
CONSECUTIVE_FAILURES = 3  # failed photos in a row before checking that Ollama still answers


class ModelUnavailable(RuntimeError):
    """Ollama or the model cannot be used: the run stops (exit code 2)."""


def _type_name(kind, allowed):
    if kind is type(None):
        return "null"
    if kind in (str, bool):
        return {str: "text", bool: "true/false"}[kind]
    if kind is float or (kind is int and float in allowed):
        return "a number"
    return "a whole number" if kind is int else kind.__name__


def check_settings(settings, path):
    """settings.json values must have the right type: argparse does not check defaults."""
    for key, value in settings.items():
        if key.startswith("_"):  # comments
            continue
        expected = SETTING_TYPES.get(key)
        if expected is None:
            raise ValueError(f"{path}: unknown setting {key!r} (allowed: {', '.join(SETTING_TYPES)})")
        if isinstance(expected, tuple) and all(isinstance(e, str) for e in expected):
            if value not in expected:
                raise ValueError(f"{path}: {key} must be one of {', '.join(expected)}, not {value!r}")
            continue
        types = expected if isinstance(expected, tuple) else (expected,)
        if isinstance(value, str) and (int in types or float in types):
            try:  # argparse converts text defaults of numeric options; check here to name the file
                (int if float not in types else float)(value)
                continue
            except ValueError:
                pass
        if isinstance(value, bool) and bool not in types or not isinstance(value, types):
            names = " or ".join(dict.fromkeys(_type_name(t, types) for t in types))
            raise ValueError(f"{path}: {key} must be {names}, not {json.dumps(value)}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_settings(path):
    """A checked settings object from a JSON file (keys starting with "_" are comments)."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except ValueError as exc:  # also UnicodeDecodeError
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    check_settings(loaded, path)
    return {k: v for k, v in loaded.items() if not k.startswith("_")}


def profile_path(name):
    """profiles/<name>.json in this folder, or a path to a JSON file."""
    folder = ROOT / PROFILE_DIR
    if re.fullmatch(r"[\w-]+", name):
        path = folder / f"{name}.json"
        if path.exists():
            return path
    path = Path(name).expanduser()
    if path.suffix.lower() == ".json" and path.is_file():
        return path
    available = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
    raise ValueError(f"Profile {name!r} not found (available: {', '.join(available) or 'none'}; or a .json file).")


def arguments(argv=None):
    # Precedence: built-in defaults < settings.json < --profile file < command-line options.
    settings = dict(DEFAULTS)
    settings_path = ROOT / "settings.json"
    if settings_path.exists():
        settings.update(load_settings(settings_path))
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--profile")
    profile = early.parse_known_args(argv)[0].profile
    if profile:
        settings.update(load_settings(profile_path(profile)))
    parser = argparse.ArgumentParser(prog="python -m trailcam_vlm", description="Trail-camera photo analysis with a local vision-language model "
                                                 "(Ollama). Slower than trailcam-expert-export; runs on a GPU "
                                                 "or, much more slowly, on the CPU.")
    parser.add_argument("directory", nargs="?", help="Input folder (or use --input)")
    parser.add_argument("--input", help="Input folder; default images/")
    parser.add_argument("--output", default=settings["output_dir"], help="Parent folder for dated export folders")
    parser.add_argument("--cache", default=settings["cache_dir"], help="Cache folder for model answers")
    parser.add_argument("--model", default=settings["model"], help="Ollama vision model (default gemma4:26b)")
    parser.add_argument("--ollama-host", default=settings["ollama_host"])
    parser.add_argument("--max-side", type=int, default=settings["max_side"],
                        help="Long side (px) of each photo sent for per-photo analysis (default 2048)")
    parser.add_argument("--event-max-side", type=int, default=settings["event_max_side"],
                        help="Long side (px) of each photo in a multi-photo event request (default 1280)")
    parser.add_argument("--max-event-frames", type=int, default=settings["max_event_frames"],
                        help="Photos per multi-photo request; longer events send this many photos spread "
                             "over the visit, always including the busiest one (default 6)")
    parser.add_argument("--event-gap", type=float, default=settings["event_gap_seconds"],
                        help="Seconds between photos that start a new event (default 60)")
    parser.add_argument("--camera-id", dest="camera_id_mode", choices=("folder", "regex", "single"),
                        default=settings["camera_id_mode"])
    parser.add_argument("--camera-pattern", default=settings["camera_id_pattern"],
                        help='Regex with a named group "camera" (with --camera-id regex)')
    parser.add_argument("--events", action=argparse.BooleanOptionalAction, default=settings["events"],
                        help="Ask the model once more per multi-photo event to count unique people (default on)")
    parser.add_argument("--limit", type=int, default=0, help="Analyze only the first N photos (quick tests)")
    parser.add_argument("--timeout", type=float, default=settings["timeout"], help="Seconds per model request")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=settings["recursive"])
    parser.add_argument("--force", action="store_true", help="Ask the model again instead of using cached answers")
    parser.add_argument("--profile", help="Settings for the hardware: gpu or cpu (profiles/<name>.json), or a .json "
                                          "file. Applied after settings.json; command-line options still win")
    parser.add_argument("--device", choices=DEVICES, default=settings["device"],
                        help="auto: Ollama places the model; gpu: the same, with a warning when the model does not "
                             "fit in GPU memory; cpu: run the model in system memory on the CPU (default auto)")
    parser.add_argument("--num-ctx", type=int, default=settings["num_ctx"],
                        help="Model context length in tokens (default 16384; must hold the photos of a request)")
    parser.add_argument("--threads", dest="num_threads", type=int, default=settings["num_threads"],
                        help="CPU threads for Ollama (default: Ollama decides)")
    parser.add_argument("--keep-alive", default=settings["keep_alive"],
                        help='How long Ollama keeps the model loaded after the last request, e.g. "30m", "1h" '
                             '(default: Ollama\'s own, 5 minutes)')
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if args.input and args.directory:
        parser.error("Use either a positional input folder or --input, not both.")
    for name in ("directory", "input", "output", "cache", "profile"):
        if '"' in str(getattr(args, name) or ""):
            parser.error(f'The folder {getattr(args, name)} contains a quote character. Remove the backslash before '
                         'the closing quote, for example "D:\\Trail Photos" instead of "D:\\Trail Photos\\".')
    source = " (from the command line, settings.json or the --profile file)"
    for name in ("max_side", "event_max_side"):
        if not 256 <= getattr(args, name) <= 4096:
            parser.error(f"--{name.replace('_', '-')} must be between 256 and 4096{source}.")
    if not 2 <= args.max_event_frames <= 12:
        parser.error(f"--max-event-frames must be between 2 and 12{source}.")
    if not math.isfinite(args.event_gap) or args.event_gap <= 0:
        parser.error(f"--event-gap must be a finite positive number of seconds{source}.")
    if args.limit < 0:
        parser.error("--limit must be >= 0.")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 2_000_000:
        parser.error("--timeout must be a finite positive number of seconds, at most 2000000" + source + ".")
    if not 2048 <= args.num_ctx <= 262144:
        parser.error("--num-ctx (num_ctx) must be between 2048 and 262144" + source + ".")
    if args.num_threads is not None and not 1 <= args.num_threads <= 1024:
        parser.error("--threads (num_threads) must be a positive number of CPU threads, at most 1024" + source + ".")
    if isinstance(args.keep_alive, str):
        args.keep_alive = args.keep_alive.strip()
        if not KEEP_ALIVE.match(args.keep_alive):
            parser.error('--keep-alive (keep_alive) must be whole seconds such as "3600" or a duration such as '
                         '"30m" or "1h"' + source + ".")
        if re.fullmatch(r"-?\d+", args.keep_alive):
            args.keep_alive = int(args.keep_alive)  # Ollama reads a plain number as seconds
    if args.camera_id_mode == "regex":
        try:
            if "camera" not in re.compile(args.camera_pattern or "").groupindex:
                parser.error('--camera-pattern needs a named group (?P<camera>...).')
        except re.error as exc:
            parser.error(f"Invalid --camera-pattern: {exc}")
    args.input = args.input or args.directory or settings["input_dir"]
    for name in ("input", "output", "cache"):
        value = Path(getattr(args, name)).expanduser()
        setattr(args, name, (value if value.is_absolute() else ROOT / value).resolve())
    return args


def progress_summary(answer):
    """One console line with every parameter (the CSV has the full detail)."""
    if not answer:
        return ""
    d = answer.get("direction_counts") if isinstance(answer.get("direction_counts"), dict) else {}
    a = answer.get
    return (f"people={a('people_total')} near={a('people_near')} | adults={a('adults')} teens={a('teens')} "
            f"children={a('children')} unknown={a('age_unclear')} | dir L{d.get('left')} R{d.get('right')} "
            f"T{d.get('toward')} A{d.get('away')} S{d.get('stationary')} U{d.get('unclear')} | "
            f"bikes={a('bicycles')} strollers={a('strollers')} moto={a('motorcycles')} atv/utv={a('atv_utv')} "
            f"vehicles={a('other_vehicles')} dogs={a('dogs')} backpacks={a('backpacks')} "
            f"large_bags={a('large_bags')} large_bags_uncertain={a('large_bags_uncertain')}")


def _cache_key(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


def read_cache(cache_file, schema):
    """(answer, info) from a cache file, or None when it is missing, damaged or not an answer."""
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        answer = clean_answer(cached["answer"], schema)
        info = cached["info"] if isinstance(cached["info"], dict) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None if answer is None else (answer, info)


def analyze_photo(client, args, item):
    """Fill item answer/info/status from the cache or one model request."""
    cache_file = args.cache / "photos" / _cache_key(args.model, PROMPT_VERSION, args.max_side) / f"{item['sha256']}.json"
    cached = None if args.force else read_cache(cache_file, PHOTO_SCHEMA)
    if cached:
        item.update(answer=cached[0], info=cached[1], status="ok", cache_hit=True)
        return
    image = encode_jpeg(open_rgb(item["path"]), args.max_side)  # an unreadable file fails before the model check
    client.require()
    answer, info = client.chat_json(
        PHOTO_PROMPT, [image], PHOTO_SCHEMA, max_tokens=700,
        check=lambda a: consistency_problems(a, "people_total"),
        repair=lambda problems: repair_note(problems, "people_total"))
    answer = clean_answer(answer, PHOTO_SCHEMA)
    item.update(answer=answer, info=info, status="ok", cache_hit=False)
    atomic_json(cache_file, {"answer": answer, "info": info, "model": args.model, "prompt_version": PROMPT_VERSION})


def select_frames(items, limit):
    """At most ``limit`` photos spread evenly over the visit (first and last included), always including the
    photo with the most people, in time order. Consecutive chunks would count people who cross a chunk boundary
    twice; an even spread sees nearly everyone in a visit, since walkers stay in view for several triggers."""
    if len(items) <= limit:
        return list(items)
    chosen = {round(k * (len(items) - 1) / (limit - 1)) for k in range(limit)}
    people = [(i.get("answer") or {}).get("people_total") or 0 for i in items]
    busiest = max(range(len(items)), key=lambda k: people[k])
    if busiest not in chosen:
        inner = sorted(chosen - {0, len(items) - 1}, key=lambda k: abs(k - busiest)) or sorted(chosen)
        chosen.discard(inner[0])
        chosen.add(busiest)
    return [items[k] for k in sorted(chosen)]


def analyze_event(client, args, event):
    """One multi-photo request per event (ok photos only; long events send a spread-out selection)."""
    items = [i for i in event["items"] if i["status"] == "ok"]
    if len(items) < 2:
        return
    if len(items) > args.max_event_frames:
        event["sampled"] = True  # the model sees at most max_event_frames photos at once
        items = select_frames(items, args.max_event_frames)
    shas = [i["sha256"] for i in items]
    cache_file = args.cache / "events" / _cache_key(args.model, PROMPT_VERSION, args.event_max_side) / \
        f"{_cache_key(*shas)}.json"
    cached = None if args.force else read_cache(cache_file, EVENT_SCHEMA)
    if cached:
        event.update(answer=cached[0], info=cached[1])
        return
    images = [encode_jpeg(open_rgb(i["path"]), args.event_max_side) for i in items]
    client.require()
    try:
        answer, info = client.chat_json(
            EVENT_PROMPT.format(n=len(images)), images, EVENT_SCHEMA, max_tokens=700,
            check=lambda a: consistency_problems(a, "unique_all"),
            repair=lambda problems: repair_note(problems, "unique_all"))
    except OllamaError as exc:
        event.update(status="error", error=f"event request failed: {exc}")
        return
    answer = clean_answer(answer, EVENT_SCHEMA)
    event.update(answer=answer, info=info)
    atomic_json(cache_file, {"answer": answer, "info": info, "model": args.model, "prompt_version": PROMPT_VERSION,
                             "photos": [i["relative_path"] for i in items]})


class Gate:
    """The Ollama client, checked the first time a request is really needed (fully cached runs work offline).

    After the first answer it reports where Ollama placed the model (GPU memory, system memory or both) and
    warns when that does not match the chosen device.
    """

    def __init__(self, client):
        self.client, self.checked, self.placement = client, False, None

    def require(self):
        if not self.checked:
            ok, reason = self.client.availability()
            if not ok:
                raise ModelUnavailable(reason + " (see README, 'Troubleshooting').")
            self.checked = True
            if self.client.device == "gpu":
                # Ollama reuses a loaded copy when a request does not name a layer count, so a copy that an
                # earlier CPU run left in system memory would keep this run on the CPU.
                where = self.client.placement()
                if where not in (None, "100% GPU"):
                    print(f"Model {self.client.model} is loaded {where}, for example by an earlier CPU run; "
                          "unloading it so that it loads into GPU memory.", flush=True)
                    self.client.unload()

    def chat_json(self, *args, **kwargs):
        result = self.client.chat_json(*args, **kwargs)
        if self.placement is None:
            self.placement = self.client.placement() or "unknown"
            self.report()
        return result

    def report(self):
        device, where = self.client.device, self.placement
        print(f"Model {self.client.model} runs: {where} (device setting: {device})", flush=True)
        if device == "gpu" and where not in ("100% GPU", "unknown"):
            print(f"  WARNING: the GPU profile is selected but the model is {where}; the CPU part makes it much "
                  "slower. Close other programs that use the GPU (ollama ps shows what is loaded; "
                  f"ollama stop {self.client.model} unloads this model), use a GPU with more memory, or use the "
                  "CPU profile.", file=sys.stderr, flush=True)
        elif device == "cpu" and where not in ("100% CPU", "unknown"):
            print(f"  WARNING: the CPU profile is selected but Ollama placed the model {where}.", file=sys.stderr,
                  flush=True)


def run(args):
    began = time.perf_counter()
    if not args.input.is_dir():
        raise ValueError(f"Input folder does not exist: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)  # fail now, not after hours of model requests
    paths = discover(args.input, (args.output, args.cache, ROOT / ".venv"), args.recursive)
    if args.limit:
        paths = paths[: args.limit]
    print(f"Found {len(paths)} photos in {args.input}", flush=True)
    client = Gate(Ollama(args.model, args.ollama_host, args.timeout, num_ctx=args.num_ctx, device=args.device,
                         num_threads=args.num_threads, keep_alive=args.keep_alive))
    print(f"Model {args.model} at {args.ollama_host}; device {args.device}"
          + (f" (profile {args.profile})" if args.profile else ""), flush=True)
    run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    items, failures_in_a_row, stopped = [], 0, None
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(args.input).as_posix()
        moment, source = capture_time(path)
        item = {"path": path, "relative_path": relative, "capture_time": moment, "time_source": source,
                "sequence_number": sequence_number(path), "model": args.model, "prompt_version": PROMPT_VERSION,
                "camera_id": camera_id_for(relative, args.camera_id_mode, args.camera_pattern),
                "status": "error", "sha256": "", "image_id": f"img_{index:06d}"}
        try:
            item["sha256"] = file_hash(path)
            item["image_id"] = "img_" + item["sha256"][:16]
            if stopped:
                raise OllamaError(stopped)
            analyze_photo(client, args, item)
            failures_in_a_row = 0
        except (ModelUnavailable, ImportError):
            raise
        except Exception as exc:  # one failed photo never stops the run
            item.update(status="error", error=f"{type(exc).__name__}: {exc}")
            if not stopped:
                print(f"  ERROR {relative}: {item['error']}", file=sys.stderr, flush=True)
            if isinstance(exc, OllamaError) and not stopped:
                failures_in_a_row += 1
                if failures_in_a_row >= CONSECUTIVE_FAILURES:
                    ok, reason = client.client.availability()
                    if not ok:
                        stopped = f"not analysed: Ollama stopped answering ({reason})"
                        print(f"  STOPPING model requests: {reason}. The remaining photos are listed as errors; "
                              "run again later (finished photos are cached).", file=sys.stderr, flush=True)
                    failures_in_a_row = 0
        items.append(item)
        print(f"[{index}/{len(paths)}] {'cached' if item.get('cache_hit') else item['status']}: {relative} "
              f"{progress_summary(item.get('answer') or {})}", flush=True)
    events = group_events([{**i, "item": i} for i in items], args.event_gap)
    event_list = []
    for event in events:
        members = [r["item"] for r in event["records"]]
        for frame, member in enumerate(members, 1):
            member.update(event_id=event["event_id"], event_frame=frame, event_image_count=len(members))
        entry = {"event_id": event["event_id"], "items": members}
        if args.events and len(members) > 1 and stopped:
            entry.update(status="error", error=stopped)
        elif args.events and len(members) > 1:
            try:
                analyze_event(client, args, entry)
            except (ModelUnavailable, ImportError):
                raise
            except Exception as exc:
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            answer = entry.get("answer") or {}
            unique = {**answer, "people_total": answer.get("unique_all"), "people_near": answer.get("unique_near")}
            print(f"event {entry['event_id']}: {len(members)} photos, unique "
                  f"{progress_summary(unique) if answer else entry.get('error')}", flush=True)
        event_list.append(entry)
    output = args.output / ("export_" + run_id)
    output.mkdir(parents=True, exist_ok=False)
    stem = output / ("export_" + run_id)
    image_rows = [image_row(i, run_id) for i in items]
    event_rows = [event_row(e, run_id) for e in event_list]
    write_csv(str(stem) + ".csv", image_rows, IMAGE_FIELDS)
    write_csv(str(stem) + "_events.csv", event_rows, EVENT_FIELDS)
    write_csv(str(stem) + "_camera_days.csv", summary_rows(event_rows, run_id), SUMMARY_FIELDS)
    errors = sum(i["status"] != "ok" for i in items)
    event_errors = sum(1 for e in event_list if e.get("status") == "error")
    summary = {"run_id": run_id, "version": __version__, "input_dir": str(args.input), "model": args.model,
               "ollama_host": args.ollama_host, "prompt_version": PROMPT_VERSION, "max_side": args.max_side,
               "event_max_side": args.event_max_side, "events_enabled": args.events,
               "max_event_frames": args.max_event_frames, "event_gap_seconds": args.event_gap,
               "camera_id_mode": args.camera_id_mode, "camera_id_pattern": args.camera_pattern,
               "profile": args.profile, "device": args.device, "num_ctx": args.num_ctx,
               "num_threads": args.num_threads, "keep_alive": args.keep_alive, "model_placement": client.placement,
               "photos": len(items), "cached_photos": sum(bool(i.get("cache_hit")) for i in items),
               "photos_with_errors": errors, "events": len(event_list), "event_requests_failed": event_errors,
               "stopped_early": stopped,
               "multi_photo_event_requests": sum(1 for e in event_list if e.get("answer") is not None),
               "total_seconds": round(time.perf_counter() - began, 3),
               "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(str(stem) + "_run.json", summary)
    print(f"CSV: {stem}.csv\nEvents: {stem}_events.csv ({len(event_list)} events)\n"
          f"Photos: {len(items)}; cached: {summary['cached_photos']}; errors: {errors}; "
          f"failed event requests: {event_errors}; elapsed: {summary['total_seconds']:.1f}s", flush=True)
    return 1 if errors or event_errors else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass
    try:
        return run(arguments(argv))
    except ImportError as exc:
        print(f"Error: a required Python package is missing ({exc}). Install it with:\n"
              f'  "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"', file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError, OverflowError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def only_the_launching_cmd(console_pids, own_pid, parent_pid, in_venv):
    """True when, apart from this Python (and the venv's python.exe launcher that started it), the console holds
    only one process: the cmd.exe of a double-clicked launcher. PowerShell or an open cmd window adds more."""
    others = set(console_pids) - {own_pid}
    if in_venv:
        others.discard(parent_pid)
    return len(others) == 1


def disposable_console():
    """True when a launcher was double-clicked: the window closes as soon as Python exits."""
    if os.name != "nt" or not re.search(r"(?:^|\s)/c(?:\s|$)", os.environ.get("TRAILCAM_VLM_LAUNCH", ""), re.I):
        return False
    try:
        import ctypes
        from ctypes import wintypes
        process_list = ctypes.WinDLL("kernel32", use_last_error=True).GetConsoleProcessList
        process_list.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        process_list.restype = wintypes.DWORD
        processes = (wintypes.DWORD * 64)()
        count = process_list(processes, len(processes))
        return only_the_launching_cmd(list(processes[:count]), os.getpid(), os.getppid(),
                                      sys.prefix != sys.base_prefix)
    except (AttributeError, OSError):
        return False


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:  # argparse errors (exit code 2) should also keep a double-clicked window open
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
    if code and disposable_console():
        try:
            input("Press Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
    raise SystemExit(code)
