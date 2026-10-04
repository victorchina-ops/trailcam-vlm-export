"""Minimal client for a local Ollama server (structured JSON output).

Images are sent only to the configured host, which is this computer by
default. Requests use temperature 0, no "thinking" output and a capped answer
length; an unparseable or inconsistent answer is retried once.
"""
from __future__ import annotations

import base64
import io
import json
import time


def normalize_model_name(name):
    """``gemma4`` and ``gemma4:latest`` are the same Ollama model."""
    name = str(name or "").strip().lower()
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def encode_jpeg(image, max_side, quality=90):
    """Base64 JPEG of an RGB PIL image scaled so its long side is at most ``max_side``."""
    copy = image.copy()
    copy.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OllamaError(RuntimeError):
    pass


class Ollama:
    retry_pause = 5.0  # seconds before repeating a request after a transient server error

    def __init__(self, model="gemma4:26b", host="http://localhost:11434", timeout=600.0, num_ctx=16384,
                 device="auto", num_threads=None, keep_alive=None):
        """``device``: "auto" lets Ollama place the model (GPU memory first, the rest on the CPU), "gpu" is the
        same placement but warns when the model does not fit, "cpu" keeps every layer in system memory
        (Ollama option ``num_gpu`` 0). ``num_threads`` sets Ollama's ``num_thread`` (CPU threads) and
        ``keep_alive`` how long Ollama keeps the model loaded after the last request (e.g. "1h")."""
        self.model, self.host, self.timeout, self.num_ctx = model, host.rstrip("/"), timeout, num_ctx
        self.device, self.num_threads, self.keep_alive = device, num_threads, keep_alive

    def options(self, max_tokens):
        """Ollama runtime options of one request."""
        options = {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens}
        if self.device == "cpu":
            options["num_gpu"] = 0  # number of layers offloaded to the GPU
        if self.num_threads:
            options["num_thread"] = self.num_threads
        return options

    def unload(self):
        """Ask Ollama to unload the model now (documented: empty request with keep_alive 0)."""
        try:
            import requests
            requests.post(self.host + "/api/generate", json={"model": self.model, "keep_alive": 0},
                          timeout=60).raise_for_status()
            return True
        except Exception:
            return False

    def placement(self):
        """Where Ollama holds the loaded model, like ``ollama ps``: "100% GPU", "100% CPU" or "40%/60% CPU/GPU";
        None when the server does not report it."""
        try:
            import requests
            models = requests.get(self.host + "/api/ps", timeout=5).json().get("models")
            wanted = normalize_model_name(self.model)
            for m in models if isinstance(models, list) else []:
                if isinstance(m, dict) and wanted in (normalize_model_name(m.get("name")),
                                                      normalize_model_name(m.get("model"))):
                    size, vram = float(m.get("size") or 0), float(m.get("size_vram") or 0)
                    if size <= 0 or vram > size:
                        return None
                    if vram == 0:
                        return "100% CPU"
                    if vram == size:
                        return "100% GPU"
                    cpu = round(100 * (size - vram) / size)  # as "ollama ps" prints a split
                    return f"{cpu}%/{100 - cpu}% CPU/GPU"
        except Exception:  # a report only; never fails a run
            return None
        return None

    def availability(self):
        """(True, "") when the server answers and lists the model, else (False, reason)."""
        import requests  # a missing package is reported as such, not as "model not available"
        try:
            response = requests.get(self.host + "/api/tags", timeout=5)
            response.raise_for_status()
            tags = response.json()
        except Exception as exc:  # refused, timed out, bad host, HTTP error, not JSON
            return False, f"no Ollama server answers at {self.host} ({type(exc).__name__}). Start Ollama"
        models = tags.get("models") if isinstance(tags, dict) else None
        models = models if isinstance(models, list) else []
        names = set()
        for m in models:
            if isinstance(m, dict):
                names.update(normalize_model_name(m.get(k)) for k in ("name", "model") if m.get(k))
        if normalize_model_name(self.model) in names:
            return True, ""
        return False, (f"Ollama is running at {self.host} but model {self.model} is not installed "
                       f"(installed: {', '.join(sorted(names)) or 'none'}). Run 'ollama pull {self.model}'")

    def available(self):
        """True when the server answers and lists the model."""
        try:
            return self.availability()[0]
        except ImportError:
            return False

    def chat_json(self, prompt, images_b64, schema, max_tokens=700, check=None, repair=None):
        """One structured request; returns (answer dict, info dict).

        ``check(answer)`` returns a list of consistency problems; when it is not
        empty the request is repeated once with ``repair(problems)`` appended to
        the conversation. Raises OllamaError when no parseable answer arrives.
        """
        messages = [{"role": "user", "content": prompt, "images": list(images_b64)}]
        info = {"requests": 0, "seconds": 0.0, "repaired": False, "problems": []}
        answer, last_error = None, None
        for attempt in range(2):
            body = {"model": self.model, "stream": False, "format": schema, "think": False, "messages": messages,
                    "options": self.options(max_tokens)}
            if self.keep_alive is not None:
                body["keep_alive"] = self.keep_alive
            try:
                content = self._post(body, info)
            except OllamaError:
                if answer is not None:  # the repair request failed: keep the first answer as given
                    break
                raise
            try:
                parsed = json.loads(content) if isinstance(content, str) else None
                if not isinstance(parsed, dict):
                    raise ValueError("answer is not a JSON object")
            except ValueError as exc:
                last_error = f"unparseable answer: {exc}"
                max_tokens = int(max_tokens * 1.5)
                continue
            answer = parsed
            problems = check(parsed) if check else []
            info["problems"] = problems
            if not problems or attempt == 1 or repair is None:
                break
            info["repaired"] = True
            messages = messages + [{"role": "assistant", "content": content},
                                   {"role": "user", "content": repair(problems)}]
        if answer is None:
            raise OllamaError(last_error or "no answer")
        return answer, info

    def _post(self, body, info):
        """POST /api/chat and return message.content; one retry after a dropped connection or a server
        error (e.g. the model runner restarting), none after a timeout (it would wait the full timeout again)."""
        import requests
        for attempt in range(2):
            started = time.perf_counter()
            try:
                response = requests.post(self.host + "/api/chat", json=body, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except (requests.ConnectionError, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                transient = isinstance(exc, requests.ConnectionError) or (status is not None and status >= 500)
                if attempt == 1 or not transient or isinstance(exc, requests.Timeout):
                    raise OllamaError(f"{type(exc).__name__}: {exc}") from exc
            except (requests.RequestException, KeyError, ValueError, TypeError, OverflowError) as exc:
                raise OllamaError(f"{type(exc).__name__}: {exc}") from exc
            finally:
                info["requests"] += 1
                info["seconds"] = round(info["seconds"] + time.perf_counter() - started, 3)
            time.sleep(self.retry_pause)
