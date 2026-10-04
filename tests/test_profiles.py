"""GPU and CPU profiles: settings precedence, Ollama request options and the model placement report."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from fakes_vlm import REPO, make_photo
from test_main import T0, CliTestCase

from trailcam_vlm import __main__ as cli
from trailcam_vlm.ollama_client import Ollama


class ProfileArgumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trailcam_vlm_profiles_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        patcher = mock.patch.object(cli, "ROOT", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def parse(self, *argv):
        return cli.arguments([str(a) for a in argv])

    def rejected(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            self.parse(*argv)
        self.assertEqual(caught.exception.code, 2)
        return err.getvalue()

    def write_settings(self, text):
        (self.tmp / "settings.json").write_text(text, encoding="utf-8-sig")

    def write_profile(self, name, values):
        folder = self.tmp / "profiles"
        folder.mkdir(exist_ok=True)
        (folder / f"{name}.json").write_text(json.dumps(values), encoding="utf-8")

    def test_precedence_defaults_settings_profile_command_line(self):
        self.assertEqual(self.parse().timeout, 600)
        self.write_settings(json.dumps({"timeout": 100, "max_side": 1024}))
        self.write_profile("cpu", {"device": "cpu", "timeout": 200})
        self.assertEqual((self.parse().timeout, self.parse().device), (100, "auto"))
        args = self.parse("--profile", "cpu")
        self.assertEqual((args.timeout, args.device, args.max_side, args.profile), (200, "cpu", 1024, "cpu"))
        args = self.parse("--profile", "cpu", "--timeout", 300, "--device", "auto")
        self.assertEqual((args.timeout, args.device), (300, "auto"))

    def test_profile_given_as_a_file(self):
        path = self.tmp / "elsewhere.json"
        path.write_text(json.dumps({"device": "gpu", "keep_alive": "5m"}), encoding="utf-8")
        args = self.parse("--profile", path)
        self.assertEqual((args.device, args.keep_alive), ("gpu", "5m"))

    def test_unknown_profile_lists_the_available_ones(self):
        self.write_profile("gpu", {"device": "gpu"})
        with self.assertRaises(ValueError) as caught:
            self.parse("--profile", "laptop")
        self.assertIn("'laptop' not found", str(caught.exception))
        self.assertIn("available: gpu", str(caught.exception))

    def test_profile_values_are_checked(self):
        self.write_profile("bad", {"device": "GPU"})
        with self.assertRaises(ValueError) as caught:
            self.parse("--profile", "bad")
        self.assertIn("device must be one of auto, gpu, cpu", str(caught.exception))

    def test_keep_alive_and_context_checks(self):
        self.assertEqual(self.parse("--keep-alive", "3600").keep_alive, 3600)
        self.assertEqual(self.parse("--keep-alive", "-1").keep_alive, -1)
        self.assertEqual(self.parse("--keep-alive", "1h").keep_alive, "1h")
        self.assertEqual(self.parse("--keep-alive", " 1h30m ").keep_alive, "1h30m")  # stripped, as sent
        for bad in ("forever", "1.5", "-0.5", "1 h"):
            with self.subTest(keep_alive=bad):
                self.assertIn("--keep-alive (keep_alive) must be whole seconds", self.rejected("--keep-alive", bad))
        self.assertIn("--num-ctx (num_ctx) must be between", self.rejected("--num-ctx", 1000))
        self.assertIn("--threads (num_threads) must be a positive", self.rejected("--threads", 0))
        self.write_profile("cpu", {"num_threads": 0})
        self.assertIn("settings.json or the --profile file", self.rejected("--profile", "cpu"))

    def test_broken_profile_json_names_the_file(self):
        (self.tmp / "profiles").mkdir()
        (self.tmp / "profiles" / "cpu.json").write_text('{"device": "cpu",}', encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.parse("--profile", "cpu")
        self.assertIn("cpu.json: not valid JSON", str(caught.exception))

    def test_whole_number_settings_say_so(self):
        self.write_settings(json.dumps({"num_ctx": 8192.5}))
        with self.assertRaises(ValueError) as caught:
            self.parse()
        self.assertIn("num_ctx must be a whole number", str(caught.exception))
        self.write_settings(json.dumps({"timeout": True}))
        with self.assertRaises(ValueError) as caught:
            self.parse()
        self.assertIn("timeout must be a number", str(caught.exception))

    def test_folder_with_a_quote_is_rejected(self):
        message = self.rejected('D:\\Trail Photos" --limit 5')
        self.assertIn("contains a quote character", message)
        self.assertIn("invalid choice", self.rejected("--device", "tpu"))


class RepositoryProfileTests(unittest.TestCase):
    def test_gpu_and_cpu_profiles_are_valid(self):
        for name, device in (("gpu", "gpu"), ("cpu", "cpu")):
            with self.subTest(profile=name):
                values = cli.load_settings(REPO / "profiles" / f"{name}.json")
                self.assertEqual(values["device"], device)
                for key in ("max_side", "event_max_side", "max_event_frames", "model"):
                    self.assertNotIn(key, values)  # answers and the cache stay the same on both
        cpu = cli.load_settings(REPO / "profiles" / "cpu.json")
        self.assertGreater(cpu["timeout"], cli.DEFAULTS["timeout"])

    def test_launchers_pass_their_profile(self):
        for name in ("gpu", "cpu"):
            with self.subTest(profile=name):
                text = (REPO / f"run_vlm_{name}.cmd").read_text(encoding="utf-8")
                self.assertIn(f'run_vlm.cmd" --profile {name} %*', text)

    def test_launcher_labels_are_unique_and_every_goto_has_one(self):
        import re
        for name in ("run_vlm.cmd", "run_vlm_gpu.cmd", "run_vlm_cpu.cmd"):
            with self.subTest(launcher=name):
                raw = (REPO / name).read_bytes()
                self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))  # CRLF only
                text = raw.decode("ascii")
                labels = re.findall(r"(?m)^:(\w+)\s*$", text)
                self.assertEqual(len(labels), len(set(labels)), labels)
                for target in re.findall(r"(?i)goto\s+:(\w+)", text):
                    self.assertIn(target, labels)
        text = (REPO / "run_vlm.cmd").read_text(encoding="ascii")
        install = text.index("pip install --quiet -r requirements.txt")
        self.assertLess(text.index("goto :no_python"), install)
        label = re.search(r"(?m)^:no_python\s*$", text).start()
        self.assertGreater(label, install)  # the error block comes after the install path, which falls through

    def test_settings_file_has_every_default(self):
        settings = json.loads((REPO / "settings.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(settings, cli.DEFAULTS)


class RequestOptionTests(unittest.TestCase):
    def test_device_threads_and_keep_alive_in_the_request(self):
        self.assertNotIn("num_gpu", Ollama(device="auto").options(700))
        self.assertNotIn("num_gpu", Ollama(device="gpu").options(700))
        options = Ollama(device="cpu", num_threads=6, num_ctx=8192).options(500)
        self.assertEqual(options, {"temperature": 0, "num_ctx": 8192, "num_predict": 500, "num_gpu": 0,
                                   "num_thread": 6})


class PlacementTests(CliTestCase):
    def photos(self):
        make_photo(self.input / "cam" / "a.jpg", people=1, when=T0)
        make_photo(self.input / "cam" / "b.jpg", people=2, when=T0 + timedelta(seconds=10))

    def test_cpu_profile_sends_num_gpu_zero_and_keep_alive(self):
        self.photos()
        self.fake.vram_share = 0.0
        result = self.run_cli("--device", "cpu", "--keep-alive", "1h", "--threads", 4)
        self.assertEqual(result.code, 0, result.stderr)
        for request in self.fake.chat_requests:
            self.assertEqual(request["body"]["options"]["num_gpu"], 0)
            self.assertEqual(request["body"]["options"]["num_thread"], 4)
            self.assertEqual(request["body"]["keep_alive"], "1h")
        self.assertEqual(result.run["model_placement"], "100% CPU")
        self.assertEqual((result.run["device"], result.run["keep_alive"], result.run["num_threads"]), ("cpu", "1h", 4))
        self.assertNotIn("WARNING", result.stderr)
        self.assertEqual(len(self.fake.ps_requests), 1)  # checked once per run

    def test_gpu_profile_warns_when_the_model_does_not_fit(self):
        self.photos()
        self.fake.vram_share = 0.4
        result = self.run_cli("--device", "gpu")
        self.assertEqual(result.run["model_placement"], "60%/40% CPU/GPU")
        self.assertIn("WARNING: the GPU profile is selected but the model is 60%/40% CPU/GPU", result.stderr)
        self.assertNotIn("keep_alive", self.fake.chat_requests[0]["body"])
        self.assertNotIn("num_gpu", self.fake.chat_requests[0]["body"]["options"])

    def test_full_gpu_and_unknown_placements(self):
        self.photos()
        result = self.run_cli("--device", "gpu")
        self.assertEqual(result.run["model_placement"], "100% GPU")
        self.assertNotIn("WARNING", result.stderr)
        self.fake.reset()
        self.fake.vram_share = None
        result = self.run_cli("--device", "gpu", "--force")
        self.assertEqual(result.run["model_placement"], "unknown")
        self.assertNotIn("WARNING", result.stderr)

    def test_gpu_run_unloads_a_copy_left_in_system_memory(self):
        self.photos()
        self.fake.vram_share = 0.0  # an earlier CPU run left the model in system memory

        def reload_on_gpu(*args, **kwargs):
            self.fake.vram_share = 1.0
        self.fake.on_unload = reload_on_gpu
        result = self.run_cli("--device", "gpu")
        self.assertEqual(len(self.fake.unload_requests), 1)
        self.assertIn("unloading it so that it loads into GPU memory", result.stdout)
        self.assertEqual(result.run["model_placement"], "100% GPU")
        self.assertNotIn("WARNING", result.stderr)

    def test_cpu_and_auto_runs_do_not_unload(self):
        self.photos()
        self.fake.vram_share = 0.0
        self.run_cli("--device", "cpu")
        self.run_cli("--device", "auto", "--force")
        self.assertEqual(self.fake.unload_requests, [])

    def test_cached_run_does_not_ask_where_the_model_is(self):
        self.photos()
        self.run_cli()
        self.fake.reset()
        result = self.run_cli()
        self.assertEqual(self.fake.ps_requests, [])
        self.assertIsNone(result.run["model_placement"])


if __name__ == "__main__":
    unittest.main()
