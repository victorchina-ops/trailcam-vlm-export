# Trail Camera VLM Export

Analyze a folder of trail-camera photographs with a **local vision-language model** (a VLM: a language model that can look at images), run through [Ollama](https://ollama.com) on this computer. The default model is **Gemma 4 26B** (`gemma4:26b`). Each run writes an images CSV, an events CSV, a camera-days CSV and a run JSON together in a separate dated export folder.

This is an alternative to [trailcam-expert-export v2](https://github.com/victorchina-ops/trailcam-expert-export) for **the same parameters, with the same CSV column names**. Version 2 counts with object detectors, pose keypoints and tracking; this tool instead shows each photo to the language model and asks it for the counts. Each event of several photos is shown to the model once more, all photos together, so that each person is counted once per visit.

**It is much slower than version 2.** It is meant for a GPU (launcher `run_vlm_gpu.cmd`); without one, `run_vlm_cpu.cmd` runs the model on the CPU from system memory, which works but is many times slower (see [GPU and CPU configurations](#gpu-and-cpu-configurations)). Photos are sent only to the Ollama server, which is this computer by default; nothing is uploaded elsewhere. This is version 0.1.0, an analysis and manual-validation tool, not verified visitor or demographic statistics. See [Validation results](#validation-results) and [Limitations](#limitations).

## What it measures

| Parameter | Per photo (images CSV) | Per visit (events CSV) |
|---|---|---|
| People, everyone visible | `people_total`, class `people_class` | `people_unique` (each person once), `people_max_frame` (busiest photo), class `people_unique_class` |
| People in the near zone | `people_near`, class `people_near_class` | `people_near_unique`, `people_near_max_frame`, class `people_near_unique_class` |
| Age groups | `adults`, `teens`, `children`, `age_unknown`; they add up to `people_total` | the same, one per unique person; they add up to `people_unique` |
| Direction of travel | `dir_left`, `dir_right`, `dir_toward`, `dir_away`, `dir_stationary`, `dir_unclear` | the same, one per unique person, from the movement across photos |
| Vehicles | `bicycles`, `strollers`, `motorcycles`, `atv_utv`, `other_vehicles` | different objects across the visit |
| Dogs and bags | `dogs`, `backpacks`, `large_bags`, `large_bags_uncertain` | different objects across the visit |

The definitions are in [Counting definitions](#counting-definitions). The camera-days CSV adds the event counts up per camera and capture date.

## When to use this tool or version 2

| | trailcam-expert-export v2 | This tool |
|---|---|---|
| How it counts | Four object detectors, pose keypoints, an attribute model, and colour-based tracking between photos | One vision-language model answers for the whole photo; one more request per multi-photo event |
| Hardware | CPU is enough; an NVIDIA GPU is used when present | Best with a GPU that holds the model (`run_vlm_gpu.cmd`); tested on an NVIDIA RTX 3090 (24 GB). Without a GPU, `run_vlm_cpu.cmd` uses the CPU and RAM (much slower) |
| Speed | About 1 second per photo on a CPU (i9-11900K, 8 threads) | About **6 seconds per photo** on an RTX 3090 with `gemma4:26b`, **plus one request per multi-photo event**: roughly 1 hour 40 minutes per 1,000 photos before event requests |
| Adults and children | Unknown by default; no teenager category (`adults`, `children`, `age_unknown`); opt-in height geometry; optional age expert, which also reports `vlm_teens` | Always estimated: `adults`, `teens`, `children` and `age_unknown` |
| Direction | Body facing in one photo; motion from tracking with `--events` | The model's judgement in one photo; movement across the photos of an event |
| Unique people per visit | Opt-in `--events`: tracks matched by clothing colour, size and position | On by default: the model links people across up to 6 photos at once |
| Large bags | Opt-in `--large-bags` geometry | Always estimated |
| Near zone | `--near-fraction` geometry, off by default (`people_near` = `people_total`) | The model's judgement: roughly the near and middle ground, within about 25 m |
| Evidence you can inspect | Boxes, masks, per-person details, contact sheet | Counts and a short note only; no boxes |
| Repeatability | Same results from the same cached detections | Temperature 0, but answers can change with model or Ollama versions |

**Is the language model supported in version 2?** Only for age. Version 2 has an optional age expert, `--age-model gemma4:26b`, which uses the same kind of model through Ollama to add age-group counts per photo (`vlm_adults`, `vlm_teens`, `vlm_children`, `vlm_age_unclear`). All other counts, objects, directions and events in version 2 still come from the detectors. To have the model answer every parameter, use this tool.

Use version 2 when you have no suitable GPU, when you need to process large numbers of photos quickly, or when you want boxes and per-person evidence to review. Use this tool when you want to test whether a vision-language model counts your photos better, for example for children, large bags or unique people per visit, and can accept the time. Try both on the same small set first (see [Comparing with version 2](#comparing-with-version-2-on-the-same-photos)).

## Requirements

- Windows 10 or 11. Windows is the tested platform; the Python code may run elsewhere, but that has not been tested.
- **64-bit Python 3.12 or 3.11.**
- **Ollama for Windows**, which runs the model in a local server at `http://localhost:11434`.
- **A GPU is optional.** `gemma4:26b` is a 17 GB download. With a GPU that holds the whole model (about 18 GB of GPU memory, for example a 24 GB card; tested on an NVIDIA RTX 3090 with 24 GB), use `run_vlm_gpu.cmd`. Without one, use `run_vlm_cpu.cmd`: the model then runs on the CPU from system memory, which needs about 20 GB of free RAM (32 GB installed as a sensible minimum) and is many times slower. See [GPU and CPU configurations](#gpu-and-cpu-configurations).
- About 20 GB of free disk space for the model. The Python environment is small (`requests`, `Pillow`, `numpy`).

## Setup on Windows (once)

1. Install Python, if it is missing, then reopen the terminal:

   ```powershell
   winget install -e --id Python.Python.3.12
   ```

   Alternatively, use the [official Python installer](https://www.python.org/downloads/windows/).

2. Install Ollama from [ollama.com/download](https://ollama.com/download), or:

   ```powershell
   winget install -e --id Ollama.Ollama
   ```

   Ollama starts in the background (an icon in the notification area) and serves at `http://localhost:11434`.

3. Download the model (17 GB):

   ```powershell
   ollama pull gemma4:26b
   ```

   `ollama list` shows the downloaded models.

4. Copy this `trailcam_vlm` folder to the computer, for example next to the version 2 folder.

5. Put your photographs in the `images` folder, **one subfolder per camera placement** (for example `images\North gate\` and `images\Spring trail\`), or pass any folder to the launcher (see below).

6. Run the launcher from PowerShell in this folder, so that progress and error messages stay visible:

   ```powershell
   .\run_vlm.cmd --limit 20
   ```

   On a computer with a suitable GPU you can use `.\run_vlm_gpu.cmd --limit 20` instead, which also checks that the model runs fully on the GPU; on a computer without one, use `.\run_vlm_cpu.cmd --limit 20`. See [GPU and CPU configurations](#gpu-and-cpu-configurations).

The first run of `run_vlm.cmd` (also when started through `run_vlm_gpu.cmd` or `run_vlm_cpu.cmd`, which call it) creates a local Python environment in `.venv` (with Python 3.12, else 3.11, else the `python` on the path) and installs the three dependencies from `requirements.txt`. It does not modify your global Python packages. Later runs reuse `.venv`; if the installation was interrupted, the next run installs again. The model itself is managed by Ollama, not by this tool.

## GPU and CPU configurations

The analysis is the same on a computer with a suitable GPU and on one without; only the hardware settings differ. Three launchers choose them:

| Launcher | Settings used | For |
|---|---|---|
| `run_vlm_gpu.cmd` | `settings.json`, then the GPU profile `profiles\gpu.json` | A computer with a GPU that holds the whole model |
| `run_vlm_cpu.cmd` | `settings.json`, then the CPU profile `profiles\cpu.json` | A computer without a suitable GPU |
| `run_vlm.cmd` | `settings.json` only (`device` `auto`) | Either: Ollama decides where the model runs, and the program only reports it |

`run_vlm_gpu.cmd` and `run_vlm_cpu.cmd` call `run_vlm.cmd --profile gpu` or `run_vlm.cmd --profile cpu`, followed by whatever you pass, so they take the same folder and options as `run_vlm.cmd`, and a folder can be dragged onto them:

```powershell
.\run_vlm_cpu.cmd "D:\Trail Photos\Camera 1" --limit 10
```

### What each profile sets

| Setting (option) | GPU profile | CPU profile | `run_vlm.cmd` (`settings.json`) |
|---|---|---|---|
| `device` (`--device`) | `gpu` | `cpu` | `auto` |
| `timeout` (`--timeout`), seconds per request | `600` | `3600` | `600` |
| `num_ctx` (`--num-ctx`), context length in tokens | `16384` | `16384` | `16384` |
| `num_threads` (`--threads`), CPU threads | not set, so `null` from `settings.json` | `null` | `null`: Ollama decides |
| `keep_alive` (`--keep-alive`) | `"10m"` | `"1h"` | `null`: Ollama's default, 5 minutes |
| Model, image sizes, photos per event request, events | from `settings.json` | from `settings.json` | `settings.json` |

Each value comes from the last place that sets it: built-in defaults, then `settings.json`, then the profile, then the command line. For example, `.\run_vlm_cpu.cmd --timeout 7200` uses the CPU profile with a two-hour limit per request, and `.\run_vlm_gpu.cmd --keep-alive 30m` keeps the model loaded for 30 minutes after the run.

`--profile NAME` reads `profiles\NAME.json`; `--profile` with the path of a `.json` file reads that file (when started through a launcher, a relative path is taken from this folder). An unknown name stops the run with a list of the available profiles. Without `--profile`, only `settings.json` and the command line apply.

### Where the model runs: `device`

| `device` | Effect |
|---|---|
| `auto` | Ollama places the model: as much as fits in GPU memory, the rest in system memory on the CPU; on a computer without a GPU, all of it on the CPU. The placement is reported, not checked. |
| `gpu` | The same placement, but the program warns when the model is not 100% on the GPU. It cannot force the model onto the GPU. Before the first request it also checks whether the model is still loaded in system memory, for example by an earlier CPU run (Ollama would keep using that copy), and if so unloads it so that it loads into GPU memory. |
| `cpu` | Every request asks Ollama for `num_gpu` 0 (no model layers on the GPU), so every layer stays in system memory and runs on the CPU, even when a GPU is present. The program warns when Ollama reports another placement. |

At the start of a run the console names the model, the device and the profile, for example `Model gemma4:26b at http://localhost:11434; device gpu (profile gpu)`. After the first model answer of the run, the program asks Ollama where the model is loaded (as `ollama ps` shows it) and prints:

```text
Model gemma4:26b runs: 100% GPU (device setting: gpu)
```

The placement is `100% GPU`, `100% CPU`, a split such as `60%/40% CPU/GPU`, or `unknown` when Ollama does not report it. With the GPU profile, a split or `100% CPU` adds a `WARNING` line: the CPU part makes the run much slower. The same value is saved as `model_placement` in the run JSON; it is `null` when the model gave no answer in this run (every answer came from the cache, or every request failed).

After a CPU run, the model can stay in system memory for the CPU profile's keep-alive time (1 hour). The GPU profile unloads such a copy itself; a run with `device` `auto` (plain `run_vlm.cmd`) keeps using it. Run `ollama stop gemma4:26b` first in that case.

### Memory and speed

- **GPU:** to run `100% GPU`, `gemma4:26b` needs about 18 GB of GPU memory, for example a 24 GB card such as the RTX 3090 (tested) or RTX 4090. On a card with less memory Ollama puts the rest of the model in system memory: the run works but is much slower, and the GPU profile warns. Other programs that use GPU memory, including a version 2 run on the GPU, leave less room for the model.
- **CPU:** about 20 GB of free RAM (the 17 GB model plus working memory); 32 GB installed is a sensible minimum. With too little memory Windows starts swapping to disk and the run becomes extremely slow.
- **Speed on the CPU:** many times slower than on a GPU. It has not been timed here, so time a `--limit 10` run first and extrapolate from the `seconds` column of the images CSV. Each request may take minutes, and the first request of a run also loads the model from disk; the CPU profile therefore allows 3600 seconds per request.
- **Smaller requests:** `--max-side 1536` and `--event-max-side 1024` send smaller images, which are faster to process; the effect on accuracy has not been measured, and the answers are cached separately from those at the default sizes. `--no-events` skips the multi-photo requests (no unique counts per visit).

### Keeping the model loaded, threads and context

- **`keep_alive`** is how long Ollama keeps the model in memory after the last request; the program sends it with every request. During a run the requests follow each other, so the model stays loaded anyway; the setting matters between runs. The CPU profile keeps the model for 1 hour, so runs started one after another (for example one camera folder after another) do not have to load it from disk again; the GPU profile keeps it for 10 minutes. While loaded, the model keeps its memory (on the CPU, about 20 GB of RAM); `ollama stop gemma4:26b` frees it at once. Values: a duration such as `"30m"` or `"1h"`, a number of seconds such as `3600`, or `-1` to keep the model until Ollama stops. `null` sends nothing, and Ollama uses its own default (5 minutes, unless the Ollama server has the environment variable `OLLAMA_KEEP_ALIVE`).
- **`num_threads`** (`--threads N`) is the number of CPU threads Ollama computes with (its `num_thread` option). `null`, as in both profiles, lets Ollama decide; it normally uses the physical cores. A smaller number, for example `--threads 6`, leaves cores free for other work and makes the analysis slower. It matters mainly on the CPU.
- **`num_ctx`** (`--num-ctx`) is the context length in tokens: the room for the question, the photos and the answer of one request, 16384 in both profiles. It must hold the photos of a multi-photo request; a longer context needs more memory. It can be set from 2048 to 262144.

### CPU and GPU answers are comparable and share the cache

Both profiles set only the device, timeout, context length, threads and keep-alive time, not the model, the image sizes or the photos per event request, so the model is shown the same images with the same questions on either computer. The cache is keyed by the photo content, model name, prompt version and image size, not by the hardware or these settings, so photos answered on the CPU are not asked again in a later GPU run (and the other way round). Answers can differ slightly between CPU and GPU computation; for a clean CPU-versus-GPU comparison, run the second one with `--force` or with a new `--cache` folder.

### Your own profile

Copy `profiles\cpu.json` or `profiles\gpu.json` to a new name in the `profiles` folder, for example `profiles\laptop.json`, and edit it:

```json
{
  "_description": "Laptop without a GPU; 6 threads leave cores free for other work.",
  "device": "cpu",
  "timeout": 3600,
  "num_threads": 6,
  "keep_alive": "1h"
}
```

Then run with its name:

```powershell
.\run_vlm.cmd "D:\Trail Photos\Camera 1" --profile laptop
```

A profile uses the keys of `settings.json` (see [the key table](#persistent-defaults-settingsjson)); keys it leaves out keep their `settings.json` value. Keys that start with `_` are comments. Values are checked as in `settings.json`: an unknown key or a wrong type stops the run with a message naming the key. To keep answers comparable with the supplied profiles and the cache shared, leave `model`, `max_side`, `event_max_side` and `max_event_frames` out of a hardware profile.

## Quick test on a small set

Test on a few photos before a long run. The examples use `run_vlm.cmd`; `run_vlm_gpu.cmd` and `run_vlm_cpu.cmd` take the same arguments. On a computer without a GPU, use `run_vlm_cpu.cmd`, and start with fewer photos, for example `.\run_vlm_cpu.cmd "D:\Trail Photos\Camera 1" --limit 10`. Either analyze only the first photos of a folder:

```powershell
.\run_vlm.cmd "D:\Trail Photos\Camera 1" --limit 20
```

or copy 20–50 photos into a separate folder and analyze that folder:

```powershell
.\run_vlm.cmd "D:\Trail Photos\Test set"
```

`--limit 20` takes the first 20 photos in alphabetical order of their paths below the input folder, so they usually all come from the first camera folder, and an event can be cut off at the 20th photo. A separate test folder lets you choose the photos: include several cameras, day and night photos, groups of different sizes, some photos without people, and **every photo of the events** you choose. A separate folder is also the way to compare with version 2 on exactly the same photos, because version 2 has no `--limit` option.

While it runs, each photo prints one line with every parameter, and each multi-photo event one more line with its unique counts, such as:

```text
[3/20] ok: North gate/IMG_0042.JPG people=2 near=2 | adults=1 teens=0 children=1 unknown=0 | dir L0 R2 T0 A0 S0 U0 | bikes=0 strollers=0 moto=0 atv/utv=0 vehicles=0 dogs=1 backpacks=1 large_bags=0 large_bags_uncertain=0
event North_gate__2025-05-03T07-30-15: 3 photos, unique people=3 near=3 | adults=2 teens=0 children=1 unknown=0 | dir L0 R3 T0 A0 S0 U0 | bikes=0 strollers=0 moto=0 atv/utv=0 vehicles=0 dogs=1 backpacks=2 large_bags=0 large_bags_uncertain=0
```

`ok` becomes `cached` for a photo answered from the cache and `error` for a failed photo. `unknown` is the age group `age_unknown`. `dir L R T A S U` are the directions left, right, toward, away, stationary and unclear. A failed event request prints its error message instead of the counts. The CSVs have the full detail.

Then open the images CSV, check a few rows against the photos, and look at `seconds` (time per photo) and the run JSON's `total_seconds` to estimate how long your full folder would take. The first request of a run can take longer while Ollama loads the model into memory. After the first model answer the console shows where the model runs, for example `Model gemma4:26b runs: 100% GPU (device setting: gpu)`; `ollama ps` shows the same (see [Where the model runs](#where-the-model-runs-device)).

## Select an input folder and options

Drag a folder onto `run_vlm.cmd` (or `run_vlm_gpu.cmd`, `run_vlm_cpu.cmd`), or pass a quoted folder path:

```powershell
.\run_vlm.cmd "D:\Trail Photos\Camera 1"
```

Use explicit input and output folders when desired:

```powershell
.\run_vlm.cmd --input "D:\Trail Photos" --output "D:\Trail Exports VLM"
```

Relative paths are resolved from this repository folder, not from the current terminal folder. Subfolders are included by default. Folders whose names start with `.`, the output and cache folders, `.venv`, symlinked files and symlinked folders are skipped. Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` and `.webp`, case-insensitively. Photos are opened with their EXIF orientation applied; 16-bit and floating-point greyscale images are rescaled to 8 bits.

| Option | Default | Behavior |
|---|---|---|
| `FOLDER` or `--input FOLDER` | `images` | Input folder. Give it either as the first argument or with `--input`, not both. |
| `--output FOLDER` | `exports` | Parent folder for exports; each run creates a new dated subfolder inside it. |
| `--cache FOLDER` | `.cache` | Folder for cached model answers. |
| `--model NAME` | `gemma4:26b` | Ollama vision model. A name without a tag means `:latest`. See [Other models](#other-models). |
| `--ollama-host URL` | `http://localhost:11434` | Address of the Ollama server. Photos are sent there. |
| `--limit N` | `0` (all) | Analyze only the first N photos (alphabetical by relative path). For quick tests. |
| `--events` / `--no-events` | on | Send each multi-photo event to the model once more, all photos together, to count unique people. `--no-events` saves those requests; events are still formed, but `people_unique` stays blank for them. |
| `--max-event-frames N` | `6` | Photos per multi-photo request, from 2 to 12. See [Long events](#long-events). |
| `--event-gap SECONDS` | `60` | A longer gap between consecutive photos of one camera starts a new event. A finite positive number. |
| `--camera-id MODE` | `folder` | How photos are assigned to cameras: `folder`, `regex` (with `--camera-pattern`) or `single`. See [Camera ids](#camera-ids). |
| `--camera-pattern REGEX` | none | Regular expression with a named group `camera`, searched in each photo's relative path. Used only with `--camera-id regex`. |
| `--max-side PX` | `2048` | Long side, in pixels, of each photo sent for per-photo analysis; from 256 to 4096. Smaller photos are not enlarged. |
| `--event-max-side PX` | `1280` | Long side of each photo in a multi-photo event request; from 256 to 4096. |
| `--timeout SECONDS` | `600` (CPU profile `3600`) | Maximum time for one model request. A request that takes longer fails. |
| `--recursive` / `--no-recursive` | on | `--no-recursive` analyzes only files directly in the input folder. |
| `--force` | off | Ask the model again instead of using cached answers. See [Caching and `--force`](#caching-and---force). |
| `--profile NAME` | none (`gpu` with `run_vlm_gpu.cmd`, `cpu` with `run_vlm_cpu.cmd`) | Hardware settings from `profiles\NAME.json`, or from a `.json` file given by its path. Applied after `settings.json`; other command-line options still win. See [GPU and CPU configurations](#gpu-and-cpu-configurations). |
| `--device MODE` | `auto` | `auto`: Ollama places the model; `gpu`: the same, with a warning when the model is not 100% on the GPU; `cpu`: the model runs in system memory on the CPU. See [Where the model runs](#where-the-model-runs-device). |
| `--num-ctx TOKENS` | `16384` | Model context length; it must hold the photos of a request. From 2048 to 262144. |
| `--threads N` | Ollama decides | CPU threads Ollama computes with, from 1 to 1024. |
| `--keep-alive DURATION` | Ollama's default (5 minutes) | How long Ollama keeps the model loaded after the last request, for example `10m`, `1h`, `3600` (seconds) or `-1` (until Ollama stops). GPU profile `10m`, CPU profile `1h`. |
| `--version` | | Print the version (0.1.0). |
| `--help` | | List the options. |

For example, analyze a folder with another model and without event requests:

```powershell
.\run_vlm.cmd "D:\Trail Photos" --model qwen3-vl:30b-a3b --no-events
```

`--max-side` and `--event-max-side` change what the model sees. The defaults were used for the timing above; the effect of other sizes on speed and accuracy has not been measured.

### Persistent defaults: `settings.json`

A profile (`--profile`, see [GPU and CPU configurations](#gpu-and-cpu-configurations)) overrides these defaults, and command-line options override both. Values are checked when the program starts: a wrong type (for example `"events": "false"` instead of `false`), an unknown `camera_id_mode` or `device`, or an unknown key stops the run with a message naming the key. Keys that start with `_` are ignored and can hold comments.

```json
{
  "input_dir": "images",
  "output_dir": "exports",
  "cache_dir": ".cache",
  "model": "gemma4:26b",
  "ollama_host": "http://localhost:11434",
  "max_side": 2048,
  "event_max_side": 1280,
  "max_event_frames": 6,
  "event_gap_seconds": 60,
  "camera_id_mode": "folder",
  "camera_id_pattern": null,
  "events": true,
  "timeout": 600,
  "recursive": true,
  "device": "auto",
  "num_ctx": 16384,
  "num_threads": null,
  "keep_alive": null
}
```

| Key | Option | Value |
|---|---|---|
| `input_dir` | `--input` | Folder, relative to this repository folder or absolute |
| `output_dir` | `--output` | Folder |
| `cache_dir` | `--cache` | Folder |
| `model` | `--model` | Ollama model name |
| `ollama_host` | `--ollama-host` | URL |
| `max_side` | `--max-side` | Whole number, 256–4096 |
| `event_max_side` | `--event-max-side` | Whole number, 256–4096 |
| `max_event_frames` | `--max-event-frames` | Whole number, 2–12 |
| `event_gap_seconds` | `--event-gap` | Positive number of seconds |
| `camera_id_mode` | `--camera-id` | `"folder"`, `"regex"` or `"single"` |
| `camera_id_pattern` | `--camera-pattern` | Regular expression, or `null` |
| `events` | `--events` / `--no-events` | `true` or `false` |
| `timeout` | `--timeout` | Positive number of seconds |
| `recursive` | `--recursive` / `--no-recursive` | `true` or `false` |
| `device` | `--device` | `"auto"`, `"gpu"` or `"cpu"` |
| `num_ctx` | `--num-ctx` | Whole number, 2048–262144 |
| `num_threads` | `--threads` | Whole number, 1–1024, or `null` (Ollama decides) |
| `keep_alive` | `--keep-alive` | A duration such as `"10m"` or `"1h"`, a number of seconds, or `null` (Ollama's default) |

Write numbers as numbers (`60`, not `"60"`) and switches as `true` or `false`. The file must contain one JSON object; out-of-range numbers stop the run with a message, as on the command line. `--limit`, `--force` and `--profile` have no setting and apply only to one run. Profiles in the `profiles` folder use the same keys.

A regular expression is easiest to set in `settings.json`, because the Windows command shell treats `<` and `>` as redirection characters. In `settings.json`, write backslashes twice (`\\d`), or use `[0-9]` instead. When passing a pattern to `run_vlm.cmd` from PowerShell, wrap it in both kinds of quotes: `--camera-pattern '"^(?P<camera>[^/]+)/"'`.

## What each run writes

Each run creates its own dated subfolder inside the output folder:

```text
exports/
  export_2026-10-04_14-30-15_123456/
    export_2026-10-04_14-30-15_123456.csv               one row per photo
    export_2026-10-04_14-30-15_123456_events.csv        one row per event
    export_2026-10-04_14-30-15_123456_camera_days.csv   one row per camera and date
    export_2026-10-04_14-30-15_123456_run.json          settings and totals
```

The folder and file names use the local date and time with microseconds, so runs never overwrite each other. Each export is a snapshot of the photos currently in the input folder, including photos answered from the cache. Use the CSVs of one export together, and do not combine rows from different exports.

The CSVs use UTF-8 with a BOM for spreadsheet compatibility. Yes/no values are written as `True` or `False`. Text cells that start with `=`, `+`, `-` or `@` get a leading `'`, so spreadsheet programs do not treat them as formulas.

**Blank means not assessed or not available; `0` means assessed and absent.** For example, every answer column of a failed photo is blank, and so are the people, age and direction columns of a photo the model judged too dark to count.

### Images CSV

One row for every photo found. Counts are **appearances in that photo**: a person photographed three times in a burst appears in three rows. Use the events CSV for visits.

| Column | Meaning |
|---|---|
| `run_id` | The run timestamp, as in the folder name |
| `image_id` | `img_` and the first 16 characters of `sha256`; identical files share it |
| `relative_path` | Path below the input folder, with `/` separators. The key for joining with labels and with version 2 exports. |
| `sha256` | Content hash of the file; the cache key |
| `camera_id` | Camera, see [Camera ids](#camera-ids) |
| `capture_time` | Date and time the photo was taken, ISO format, camera clock, no time zone |
| `time_source` | `exif_original`, `exif_digitized`, `exif_datetime` or `file_mtime` (file modification time) |
| `event_id` | The event the photo belongs to |
| `event_frame` | Position of the photo in its event (1 = earliest) |
| `event_image_count` | Number of photos in the event |
| `status` | `ok`, or `error` when the photo could not be analyzed |
| `error` | The error message of a failed photo |
| `cache_hit` | `True` when the answer came from the cache, `False` when the model was asked in this run |
| `model` | Ollama model name |
| `prompt_version` | Version of the questions and answer format sent to the model (`vlm_full_v1`) |
| `seconds` | Model time for this photo, over all its requests; for cached rows, the time of the original request |
| `requests` | HTTP requests sent for this photo: usually 1; up to 4 with a repeated or correction request and an automatic retry after a dropped connection or server error |
| `repaired` | `True` when the model was asked to correct an inconsistent answer |
| `consistency_problems` | Inconsistencies left in the final answer, for example `age groups sum to 3, not people_total 4`; blank when consistent |
| `people_count_assessable` | `False` when the model judged the photo too dark, blurred or occluded to count people; the people, age and direction columns are then blank |
| `people_total` | Every visible person, including distant and background people |
| `people_near` | People in the near zone |
| `people_class`, `people_near_class` | `people_total` and `people_near` as [count classes](#count-classes) |
| `adults`, `teens`, `children`, `age_unknown` | Age groups; they add up to `people_total`. `age_unknown` holds the people the model could not judge (its answer `age_unclear`); teenagers are only in `teens`. See [Adults, teenagers and children](#adults-teenagers-and-children). |
| `dir_left`, `dir_right`, `dir_toward`, `dir_away`, `dir_stationary`, `dir_unclear` | Direction of travel per person; they add up to `people_total` |
| `bicycles`, `strollers`, `motorcycles`, `atv_utv`, `other_vehicles`, `dogs`, `backpacks` | Objects visible in the photo |
| `large_bags` | Clearly large bags |
| `large_bags_uncertain` | Bags whose size is ambiguous; never also counted in `large_bags` |
| `scene_uncertain` | `True` when darkness, crowding, blur, occlusion or cropping makes any count uncertain |
| `notes` | The model's short note about a material ambiguity; usually empty |

### Events CSV

One row per event: a run of photos from one camera with short gaps between them (see [Events](#events)).

| Column | Meaning |
|---|---|
| `run_id` | The run timestamp |
| `event_id` | Camera id and first capture time, for example `North_gate__2025-05-03T07-30-15` |
| `camera_id` | Camera |
| `start`, `end` | Capture times of the first and last photo |
| `duration_seconds` | `end` minus `start` |
| `image_count` | Photos in the event, including failed ones |
| `method` | Where the visit counts come from: `multi_photo_model`, `single_photo` or `per_photo_max` (see below) |
| `people_unique` | Different people in the event, each counted once |
| `people_max_frame` | The largest `people_total` of any analyzed photo of the event (MaxN) |
| `people_near_unique` | Different people in the near zone |
| `people_near_max_frame` | The largest `people_near` of any analyzed photo |
| `people_unique_class`, `people_near_unique_class` | `people_unique` and `people_near_unique` as count classes |
| `adults`, `teens`, `children`, `age_unknown` | Age groups of the unique people; they add up to `people_unique` (for method `per_photo_max`, to `people_max_frame`) |
| `dir_left` … `dir_unclear` | Direction of each unique person |
| `bicycles` … `large_bags_uncertain` | Objects (see the methods below) |
| `status` | `ok` or `error` |
| `error` | Why the multi-photo request failed, when it did |
| `seconds`, `requests`, `repaired`, `consistency_problems` | As in the images CSV, for the multi-photo request. For method `single_photo`, `consistency_problems` is the photo's; otherwise blank |
| `needs_review` | `True` when `review_reasons` is not empty |
| `review_reasons` | Reasons to check the event, separated by `; ` (see below) |
| `images` | JSON list of the event's photos by relative path, in capture order |
| `notes` | The model's note |

The `method` column says how the visit counts were made:

| `method` | When | Counts |
|---|---|---|
| `multi_photo_model` | Events with at least two analyzed photos, with `--events` on | One request with all photos of the event (at most `--max-event-frames`, see [Long events](#long-events)). `people_unique`, ages and directions count each person once; objects are the number of **different** objects across the visit, not the sum over photos. `people_unique` is never below `people_max_frame` (see `unique_below_busiest_photo`). |
| `single_photo` | Events of one photo | Copied from that photo's answer |
| `per_photo_max` | `--no-events`, a failed multi-photo request, or fewer than two analyzed photos | `people_unique` and `people_near_unique` blank; ages and directions from the busiest photo (so they add up to `people_max_frame`); objects the maximum over the photos. These are lower bounds for the visit. |

| Review reason | Meaning |
|---|---|
| `inconsistent_model_answer` | The event answer (or, for a one-photo event, the photo's answer) was still inconsistent after one correction request (see `consistency_problems`) |
| `unique_below_busiest_photo` | The multi-photo answer counted fewer people than the busiest single photo shows. `people_unique` was raised to `people_max_frame` (and `people_near_unique` to at least `people_near_max_frame`); the added people are counted in `age_unknown` and `dir_unclear`. In validation this floor improved the count classes. |
| `unique_far_above_busiest_photo` | `people_unique` is more than twice `people_max_frame` and at least 3 higher: possibly people counted twice |
| `unique_counts_not_available` | No unique count: method `per_photo_max`, or a multi-photo answer without a people count |
| `photo_analysis_failed` | At least one photo of the event failed |
| `event_sampled_for_model` | The event had more analyzed photos than `--max-event-frames`; the model saw a selection (see [Long events](#long-events)) |
| `times_from_file_dates` | A multi-photo event in which a photo had no EXIF time; it may combine separate visits |

### Camera-days CSV

Event totals added up per camera and capture date (the date of each event's `start`, or `unknown`).

| Column | Meaning |
|---|---|
| `run_id`, `camera_id`, `date` | Run, camera and date |
| `events`, `images` | Number of events and of photos |
| `people_max_frame`, `people_unique`, `people_near_max_frame`, `people_near_unique` | Sums over the day's events |
| `adults`, `teens`, `children`, `age_unknown` | Sums over the day's events |
| `dir_left` … `dir_unclear`, `bicycles` … `large_bags_uncertain` | Sums over the day's events |
| `events_without_unique_count` | Events of the day with a blank `people_unique` (not covered by the `people_unique` sum) |
| `events_needing_review` | Events of the day with `needs_review` |

Blank event values are left out of a sum, and a sum is blank only when every event of the day is blank. When `events_without_unique_count` is not 0 (for example with `--no-events`, or after a failed event request), `people_unique` covers only some of the day's events; `people_max_frame` covers all of them. There are no class columns, because classes cannot be added up.

### Run JSON

| Key | Meaning |
|---|---|
| `run_id`, `version`, `finished_at_utc` | Run timestamp, package version, finish time in UTC |
| `input_dir`, `model`, `ollama_host`, `prompt_version` | What was analyzed, with which model, where, and with which questions |
| `max_side`, `event_max_side`, `events_enabled`, `max_event_frames`, `event_gap_seconds`, `camera_id_mode`, `camera_id_pattern` | Settings of the run |
| `profile` | The `--profile` given (a name such as `gpu` or `cpu`, or a file path), or `null` |
| `device`, `num_ctx`, `num_threads`, `keep_alive` | Hardware settings of the run; `null` threads or keep-alive time means Ollama's own default |
| `model_placement` | Where Ollama held the model after the first model answer of the run: `100% GPU`, `100% CPU`, a split such as `60%/40% CPU/GPU`, or `unknown` when Ollama did not report it; `null` when the model gave no answer in this run (everything from the cache, or every request failed) |
| `photos`, `cached_photos`, `photos_with_errors` | Photos found, answered from the cache, and failed |
| `events`, `multi_photo_event_requests`, `event_requests_failed` | Number of events, of events with a multi-photo answer (new or from the cache), and of failed multi-photo requests |
| `stopped_early` | Blank, or why model requests were stopped (Ollama stopped answering in the middle of a run) |
| `total_seconds` | Elapsed time of the whole run |

The program exits with code 0 when every photo and event request succeeded, 1 when some failed (the export is still written), and 2 when the run could not start or the model could not be reached (for example a missing input folder, an invalid option or setting, Ollama not running, or the model not downloaded).

A request that fails because the connection dropped or the server reported an error (for example a restarting model runner) is repeated once after 5 seconds; a timeout is not repeated. After 3 failed photos in a row the program checks that Ollama still answers; if not, it stops sending requests, lists the remaining photos as errors and still writes the export. Run again later: finished photos come from the cache.

## Counting definitions

The model is given the same definitions as the version 2 validation labels, so both tools can be scored against the same labels. The black data bar at the bottom of a photo (date, temperature, camera name) is not part of the scene. The model is asked to use only visible evidence and not to invent hidden people or objects.

### People: everyone visible or the near zone

| Column | Counts | Use it for |
|---|---|---|
| `people_total` | Every visible real person once, including riders, partly hidden people, distant background people and carried infants; not shadows, reflections, pictures or statues | How many people are in view, for example crowding at a viewpoint |
| `people_near` | People passing through the near and middle ground that the camera monitors, roughly within 25 m, leaving out distant background crowds | People passing the camera: trail use and visitor counts |

The near zone is the **model's judgement** of the scene, not a measured distance, and there is no setting for it. It is closer to the camera owner's labelling convention ("the group passing the camera") than to version 2's default, where `people_near` equals `people_total` unless `--near-fraction` is set. The model is asked to keep `people_near` at most `people_total`. Decide before the analysis which count you will report, and state the definition with the results.

### Count classes

Every people count also comes as a group-size class: `0`, `1`, `2`, `3–4`, `5–10` or `>10`, the same classes as in version 2. The images CSV has `people_class` (of `people_total`) and `people_near_class`; the events CSV has `people_unique_class` and `people_near_unique_class`. A class is blank when its count is blank.

The ranges use an **en dash** (`3–4`, not `3-4`), because Excel turns `3-4` and `5-10` into dates. Type the en dash when filtering on these values. Counting errors of one or two people often leave the class unchanged, so classes are useful to describe group sizes, for example the share of visits by groups of 3–4. Use the counts for totals: classes cannot be added up.

### Adults, teenagers and children

- `children`: clearly a pre-teen child, roughly under 12, including toddlers and carried babies. The model is told to judge from body proportions (head size relative to the body, limb length), height compared with adults at a similar distance, and context (held by the hand, carried, in a stroller or baby carrier).
- `teens`: adolescents.
- `adults`: grown-ups.
- `age_unknown`: too small, dark, blurred or hidden to judge (the model's answer `age_unclear`). The model is told not to guess.
- `adults + teens + children + age_unknown = people_total` (per event: `= people_unique`).

Teenagers are a category of their own and are not counted in `age_unknown`. Version 2 and its validation labels have no teenager category: version 2 exports `adults`, `children` and `age_unknown`, and the labels count teenagers as `age_unclear`, which version 2's evaluation script compares with `age_unknown`. See [Comparing with version 2](#comparing-with-version-2-on-the-same-photos) for how to compare the ages. Where a teenager ends and an adult or a child begins is uncertain in a photo, so treat `teens` as an estimate (see [Validation results](#validation-results)).

These are judgements of apparent age from appearance, not identity claims or actual ages.

### Direction of travel

Directions are relative to the image, not compass directions, one per person:

- `dir_left` and `dir_right`: moving toward the **left or right edge of the image as seen by the viewer**, never the person's own left or right.
- `dir_toward` and `dir_away`: approaching or moving away from the camera.
- `dir_stationary`: standing or sitting still. `dir_unclear`: cannot tell.
- Diagonal movement counts as its dominant component.

In a **single photo**, the model judges direction from gait, posture, facing and bicycle orientation: a judgement, not a measured movement. In a **multi-photo event**, it judges each unique person's movement across the photos (position and size change between photos), and for people seen in only one photo, from gait and facing. The six directions add up to `people_total` per photo and to `people_unique` per event.

### Vehicles, dogs and bags

Objects are counted, not their users: two people on one tandem bicycle are one bicycle.

| Column | Counts |
|---|---|
| `bicycles` | Ridden or parked bicycles and e-bikes |
| `strollers` | Baby strollers, prams and pushchairs; not wheelchairs or carts |
| `motorcycles` | Motorbikes, dirt bikes and motor scooters |
| `atv_utv` | Quad bikes/ATVs, side-by-side off-road vehicles (RZR, Ranger, buggy), golf carts and small utility carts |
| `other_vehicles` | Cars, jeeps, pickups, trucks, buses and tractors |
| `dogs` | Dogs only; not wildlife |
| `backpacks` | All visible backpacks, of any size |
| `large_bags` | Clearly bulky trekking or frame packs (for example rising above the shoulders or reaching the hips, or with a sleeping mat attached), luggage, or large duffel bags. Ordinary daypacks, handbags, and shopping or garbage bags are not large. |
| `large_bags_uncertain` | Bags whose size is ambiguous; never also in `large_bags` |

Per photo, an object column counts the objects visible in that photo. Per multi-photo event, it counts the **different** objects across the whole visit: a bicycle seen in three photos is one bicycle. Version 2 and its event labels use the maximum visible in any one photo instead; see [Comparing with version 2](#comparing-with-version-2-on-the-same-photos).

### Consistency checks

After each answer, the program checks that `people_near` is not larger than `people_total`, and that the age groups and the six directions each add up to `people_total` (for events, the same with `people_near_unique` and `people_unique`). If not, the model is shown the problem once and asked for a corrected answer; if that correction request fails, the first answer is kept. Problems that remain are reported in `consistency_problems`, and for events also as `needs_review`; the counts are exported as the model gave them. An answer that is not valid JSON is asked for once more; a second invalid answer makes the photo, or the event request, fail. Values of the wrong type (text instead of a number, a negative count) are treated as blank; one invalid direction value blanks all six direction columns of that answer, and a missing direction counts as 0. A photo the model marks as not assessable (`people_count_assessable` false) has blank people, age and direction columns even if the model also gave numbers.

## Cameras, capture times and events

### Camera ids

Every photo is assigned a camera id. Events never combine photos from different cameras.

| Mode | Camera id | Example: `North gate/2025-05 card/IMG_0042.JPG` |
|---|---|---|
| `folder` (default) | The photo's parent folder, relative to the input folder; `.` for photos directly in the input folder | `North gate/2025-05 card` |
| `regex` | The named group `camera` of the first match of `--camera-pattern` in the relative path; the folder when the pattern does not match | With `^(?P<camera>[^/]+)/`: `North gate` |
| `single` | `all` for every photo | `all` |

Keep **one folder per camera placement**, and start a new folder when a camera is moved or re-aimed.

### Capture times

The capture time comes from the photo's EXIF data (`DateTimeOriginal`, then `DateTimeDigitized`, then `DateTime`, to the second, and only for years after 1980), otherwise from the file's modification time. `time_source` says which was used. Times are the camera's own clock without a time zone. Copying files often gives them the same modification time, so a multi-photo event in which any photo has a file time is flagged `times_from_file_dates`. Copy photos with their EXIF data intact.

### Events

For each camera, photos are sorted by capture time (then by the number in the filename, for example `IMAG0571` → 571, then by path). A new event starts when the gap to the previous photo is longer than `--event-gap` (default 60 seconds). A photo without a usable time stays with its neighbour only when their filename numbers differ by at most 3. Events are formed in every run, also with `--no-events`; that switch only skips the multi-photo requests.

Event ids look like `North_gate_2025-05_card__2025-05-03T07-30-15`: the camera id (`root` for photos directly in the input folder) and the first capture time, or the first filename when no photo has a time. An id depends only on its own event, so it stays the same in later exports while the event keeps the same photos. When two events would get the same id, the later ones get `__02`, `__03`, and so on.

With `--events` on (the default), for an event of two or more analyzed photos, the model is shown all photos together (each scaled to `--event-max-side`, in time order, photo 1 first) and asked to link the same person across photos by clothing, bags, hats, size and plausible movement, and to count each person once. People who appear in only one photo still count. The result is `people_unique`, next to `people_max_frame` (MaxN, the busiest single photo) from the per-photo answers. Review both, preferably as classes, and check events flagged `needs_review`. A person who passes again after the event gap is counted again in the later event.

### Long events

A multi-photo request holds at most `--max-event-frames` photos (default 6). When an event has more analyzed photos, the model is shown **a selection spread evenly over the visit**: the first and last photo, evenly spaced photos in between, and always the photo with the most people, in time order. Walkers usually stay in view for several triggers, so a spread-out selection sees nearly everyone, whereas splitting the event into consecutive parts would count people who cross from one part to the next twice. A person seen only in a photo that was not selected is missed. Such events are flagged `event_sampled_for_model`; `people_max_frame` still covers every photo, and `people_unique` is never below it. For long events, raise `--max-event-frames` (up to 12; more photos per request need more memory and time), or lower `--event-gap` if the event joins separate visits.

## Caching and `--force`

Every answer is saved in the cache folder (`.cache` by default) as soon as it arrives:

- per photo, in `.cache\photos\`, keyed by the photo's content hash (SHA256), the model name, the prompt version and `--max-side`;
- per multi-photo event, in `.cache\events\`, keyed by the content hashes of the photos sent, the model name, the prompt version and `--event-max-side`.

**Add more photos and run again:** only new or changed photos, and events whose photos changed, are sent to the model. A renamed or moved photo keeps its cached answer, because the key is its content. Changing `--model`, `--max-side` or the prompt version asks the model again for every photo; changing `--event-max-side` asks again for every event. Changing `--event-gap`, `--camera-id`, `--camera-pattern` or `--max-event-frames` reuses the photo answers, regroups the events, and sends only the new combinations of photos. Switching back to an earlier setting reuses its answers. The device, threads, context length, keep-alive time and timeout are not part of the keys, so runs with the GPU and the CPU profile share their answers (see [GPU and CPU configurations](#gpu-and-cpu-configurations)).

Failed requests are not cached and are retried on the next run. `--force` asks the model again for every photo and event of the run and replaces the cached answers. Use it after updating a model under the same name (see [Limitations](#limitations)), or delete the cache folder, or use a new one with `--cache`.

Ollama is contacted only when something is not in the cache. A run in which every photo and event is cached (for example to export again after changing `--event-gap`) works without the server.

## Other models

Any Ollama model that accepts images and supports structured (JSON schema) output can be tried with `--model`, for example:

```powershell
ollama pull qwen3-vl:30b-a3b
.\run_vlm.cmd "D:\Trail Photos\Test set" --model qwen3-vl:30b-a3b
```

Only `gemma4:26b` was tested; other models, including `qwen3-vl:30b-a3b`, are untested with this tool. Models differ in accuracy, speed, memory needs and how reliably they follow the answer format; a model that does not return valid JSON makes photos fail with an `unparseable answer` error. Answers are cached per model, so you can run several models on the same test folder and compare their exports against the same labels. The `model` column and the run JSON record which model answered.

## Privacy

- Photos are sent only to the Ollama server named by `--ollama-host` (`http://localhost:11434`, this computer, by default). Pointing it at another computer sends the photos there over the network, unencrypted with `http://`.
- The program makes no other network requests. The first run of `run_vlm.cmd` downloads the Python packages, and `ollama pull` downloads the model; neither sends photos.
- Your photos, exports, cache and Python environment are excluded from version control by the supplied `.gitignore`. The cache holds the model's answers, not the photos; event entries also list the photos' relative paths.
- CSVs contain filenames, capture times and the model's notes. Share outputs deliberately.

## Comparing with version 2 on the same photos

The column names match version 2, so the version 2 evaluation script can score both exports against the same labels. It is part of [trailcam-expert-export](https://github.com/victorchina-ops/trailcam-expert-export), uses only the Python standard library, and loads no model. See its [docs/VALIDATION.md](https://github.com/victorchina-ops/trailcam-expert-export/blob/main/docs/VALIDATION.md) for the labelling definitions and the report columns.

In the commands below, `VLM` stands for this tool's folder and `V2` for the folder of version 2 (a clone of trailcam-expert-export); replace them with the real paths, for example `D:\Tools\trailcam-vlm-export` and `D:\Tools\trailcam-expert-export`. `DATE` stands for an export's timestamp.

1. Put the test photos in one folder, for example `D:\Trail Photos\Test set`, with complete events.
2. Run version 2 on that folder, with the experimental outputs you want to compare switched on (by default it leaves ages unknown, large bags blank and events off):

   ```powershell
   cd V2
   .\run_analysis.cmd "D:\Trail Photos\Test set" --events --large-bags
   ```

   Add `--age-model gemma4:26b` to get version 2's age expert as well; its counts are in the `vlm_*` columns (`vlm_adults`, `vlm_teens`, `vlm_children`, `vlm_age_unclear`), not in `adults`, `children` and `age_unknown`.
3. Run this tool on the same folder:

   ```powershell
   cd VLM
   .\run_vlm.cmd "D:\Trail Photos\Test set"
   ```

   Both exports then have the same `relative_path` for each photo.
4. Make a blank labels file from either images CSV, and label every photo following the version 2 definitions:

   ```powershell
   cd V2
   .\.venv\Scripts\python.exe -m trailcam.evaluate --template "VLM\exports\export_DATE\export_DATE.csv" --out labels.csv
   ```

5. Make copies of this tool's export with teenagers counted in `age_unknown`. Version 2 and its labels have no teenager category: the label `age_unclear` includes teenagers, and the script scores it against the export's `age_unknown`, while this tool keeps teenagers in `teens`. The copies change only `age_unknown` (teens + age_unknown); `adults` and `children` compare directly.

   ```powershell
   cd VLM
   .\.venv\Scripts\python.exe -m trailcam_vlm.v2_ages "exports\export_DATE"
   ```

   This writes `export_DATE_v2ages.csv` and `export_DATE_events_v2ages.csv` next to the originals. On the validation photos, the copies gave the version 2 script the same age scores as an export made before teenagers became a separate category; the originals would make every teenager look like a missed `age_unclear` person.
6. Score each export against the same labels (replace `DATE` with each export's timestamp), using the copies for this tool:

   ```powershell
   cd V2
   .\.venv\Scripts\python.exe -m trailcam.evaluate --labels labels.csv --export "exports\export_DATE\export_DATE.csv" --events "exports\export_DATE\export_DATE_events.csv"
   .\.venv\Scripts\python.exe -m trailcam.evaluate --labels labels.csv --export "VLM\exports\export_DATE\export_DATE_v2ages.csv" --events "VLM\exports\export_DATE\export_DATE_events_v2ages.csv"
   ```

   With image labels and `--events`, the script derives event labels for every event whose photos are all labelled, using each export's own `event_id` column, so the two tools' event ids need not match. Add `--json report.json` to keep the full report.

Keep these differences in mind:

- Ages: the script compares `adults`, `children` and the label `age_unclear` (with `age_unknown`); it does not score `teens`. Version 2's age expert reports teenagers in `vlm_teens`; its documentation compares `vlm_teens + vlm_age_unclear` with the label `age_unclear`.
- It does not score `people_near` or the classes. To check them, add your own `people_near` label column and use the short Python comparison in version 2's [docs/VALIDATION.md](https://github.com/victorchina-ops/trailcam-expert-export/blob/main/docs/VALIDATION.md#near-zone-counts-classes-and-the-age-expert). Version 2's `people_near` equals `people_total` unless `--near-fraction` is set; this tool's is the model's near-zone judgement.
- Event objects: this tool counts different objects across the visit; version 2 and the derived event labels use the maximum in any one photo. Compare objects at photo level.
- Version 2 splits events also when the camera model or image size changes, and reads capture times to fractions of a second; event ids of the two tools are usually, but not always, the same.

To see where the two tools disagree before labelling, join the two images CSVs on `relative_path`, for example with Python:

```python
import csv

csv.field_size_limit(100_000_000)  # version 2 CSVs have large evidence cells

def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {row["relative_path"]: row for row in csv.DictReader(handle)}

v2 = rows(r"V2\exports\export_DATE\export_DATE.csv")    # replace V2, VLM and DATE
vlm = rows(r"VLM\exports\export_DATE\export_DATE.csv")
for path in sorted(v2.keys() & vlm.keys()):
    a, b = v2[path]["people_total"], vlm[path]["people_total"]
    if a != b:
        print(path, "v2:", a, "vlm:", b)
```

Disagreements are where labels are most informative; to estimate accuracy, label photos chosen by a fixed rule, not only the disagreements.

## Validation results

Measured on 2026-10-04 with `gemma4:26b` (Ollama, RTX 3090), prompt version `vlm_full_v1`, default settings, against the same labels used for version 2 (see its [docs/VALIDATION.md](https://github.com/victorchina-ops/trailcam-expert-export/blob/main/docs/VALIDATION.md)). Version 2 numbers are its default settings (no `--age-model`, `--large-bags`, `--near-fraction` or geometry age), scored the same way. These are small samples from five Israeli trail cameras; treat the differences as indications, not guarantees for other cameras.

**157 single photos** (the 100 People_100 photos plus 57 photos chosen for crowds, vehicles, dogs and bags; the adjudicated labels count everyone visible; the manual labels are the ecologist's own counts for 100 of them):

| Measure | Version 2 (detectors) | This tool |
|---|---|---|
| People, everyone visible: count class correct (0 / 1 / 2 / 3–4 / 5–10 / >10) | 0.78 | **0.88** |
| People, everyone visible: mean absolute error / mean error | 1.51 / +0.07 | **1.05** / −0.97 |
| People vs manual counts: class correct (`people_total`) | 0.72 | **0.86** |
| Children present (F1, adjudicated / manual) | not reported by default | **0.86 / 0.83** |
| Bicycles present (F1) | 0.96 | **1.00** |
| Strollers present (F1) | 0.75 | **0.89** |
| Dogs present (F1) | 0.88 | **0.94** |
| Large bags present (F1) | not reported by default | **0.70** |
| Any motor vehicle present (F1) | 0.86 | **1.00** |
| Direction in one photo: share of labelled people with a matching direction | 0.24 | **0.52** |
| Direction in one photo: share of assigned directions that match | **0.56** | 0.54 |
| Time per photo | about 1 s (CPU) | about 6 s (RTX 3090) |

The people error depends strongly on group size: for 1 or 2 people the model was almost always exact (mean absolute error 0.07 and 0.00), for 3–4 people 0.51, for 5–10 people 1.18, and for more than 10 people 3.6, nearly always too low (for example 78 counted of 91). It tends to leave out small, distant background people. Children: 30 of 37 photos with children found, 3 false alarms. Version 2 with its optional Gemma age expert (`--age-model`) reached a children F1 of 0.73–0.81 on the same photos.

Teenagers: the camera owner's descriptions of the 100 People_100 photos mention teenagers in 8 photos. The model reported teens in 6 of those 8 and in 11 other photos (3 of them described by the owner as "young adults", others with boys, girls or kids): a presence F1 of 0.48. Over all 157 photos, 24% of the people were reported as teens, mostly in groups of young people. The boundaries between teens and adults and between teens and children are uncertain, so treat `teens` as an estimate.

**28 multi-photo visits** (People_Series_100: 3–4 photos each, 2–100 people; labels give the unique people per visit):

| Measure | Version 2 tracking (`--events`) | This tool |
|---|---|---|
| Unique people, everyone visible: count class correct | 0.82 | **0.89** |
| Unique people: mean relative error | 0.25 | **0.20** |
| Unique people: mean absolute error / mean error | **5.1** / −2.8 | 5.6 / −5.5 |
| Unique people in the near zone: count class correct | 0.82 | **0.86** |
| Children present in the visit (F1) | not reported by default | **0.88** (11 of 14, no false alarms) |
| Dominant direction of the visit correct | 12 of 28 | **17 of 28** |

Small visits (up to 5 people) were counted exactly or within one. Large groups were undercounted: 31 of 57, 23 of 47, 68 of 100. The absolute error is therefore dominated by the few largest groups, where version 2 does slightly better; for count classes and relative error this tool does better. In 5 of 28 visits the multi-photo answer was below the busiest single photo; the export raises it to that photo's count (`unique_below_busiest_photo`), which improved the class accuracy from 0.86 to 0.89 in this sample.

What this means in practice: for small groups, children, bicycles, strollers, dogs and large bags this tool was more accurate than version 2's defaults on these labels, at about 6 times the time on a GPU. For large crowds both tools undercount; use the count classes and check crowded photos and visits by eye.

## Limitations

- **Model mistakes.** The model can miss small, distant or partly hidden people, count a person twice, confuse categories (for example a motorcycle and an ATV, or a dog and wildlife), or report objects that are not there. It gives no boxes, so the only way to check a count is to look at the photo. Large groups and night infrared photos are the hardest.
- **Single-photo direction is a judgement** from posture, gait and facing, not a measured movement; a person can look or stand sideways while walking toward the camera.
- **Unique people depend on the model linking people across photos.** Similar-looking hikers can be merged and one person can be counted twice. Large groups are undercounted (see [Validation results](#validation-results)). For events longer than `--max-event-frames` the model sees a selection of the photos, flagged `event_sampled_for_model` (see [Long events](#long-events)).
- **Events depend on capture times.** Wrong or reset camera clocks, missing EXIF data, or a gap setting that does not match the camera produce wrong events. A person returning after the event gap is counted again.
- **The near zone** (about 25 m) is a judgement by the model, not a measured distance, and can vary between cameras and photos.
- **Ages** are apparent ages judged from appearance; they are not identity claims or actual ages. `teens` is the least certain group: the boundaries with adults and with children are uncertain (see [Validation results](#validation-results)).
- **Nondeterminism across model versions.** Requests use temperature 0, so the same model usually gives the same answer for the same photo, but a different model version, Ollama version, GPU or setting can change answers. The cache is keyed by the model name, not by its exact version, so after `ollama pull` updates a model, run with `--force` or a new `--cache` folder. Record the model, `prompt_version` and the Ollama version (`ollama --version`) with your results.
- **Speed and hardware.** About 6 seconds per photo on an RTX 3090, plus event requests; much slower if the model does not fit in GPU memory or runs on the CPU (see [GPU and CPU configurations](#gpu-and-cpu-configurations)).
- **Counts in the images CSV are appearances**, not unique people or visits; summing photos double-counts. Use the events CSV for visits.
- **License of the model is separate.** Ollama and the models it downloads are separate software with their own licenses and terms of use. This package does not install or distribute them; check the model's terms before use.
- Check a labelled sample from your own cameras before relying on counts or demographic estimates.

## Troubleshooting

| Symptom | Action |
|---|---|
| `Error: no Ollama server answers at http://localhost:11434 ... Start Ollama` | Start Ollama (from the Start menu, or `ollama serve` in another terminal). Check `--ollama-host`. |
| `Error: Ollama is running at ... but model gemma4:26b is not installed (installed: ...)` | Run `ollama pull gemma4:26b`, or pass one of the installed models with `--model`. |
| `Error: a required Python package is missing` | The environment is incomplete. Run the `pip install` command printed with the message, or delete the `.venv` folder and run `run_vlm.cmd` again. |
| `STOPPING model requests: ...` | Ollama stopped answering during the run (crashed, closed, or out of memory). Restart Ollama and run again; finished photos come from the cache. |
| Python is missing or the environment cannot be created | Install 64-bit Python 3.12 or 3.11, reopen the terminal, and run `run_vlm.cmd` again. To rebuild the environment, delete the `.venv` folder. |
| Very slow (tens of seconds or minutes per photo) | The model probably does not fit in GPU memory and runs partly or fully on the CPU. The console line `Model ... runs:` and `ollama ps` show the split. Close other programs that use the GPU (including a version 2 run on the GPU), use a GPU with more memory, or try a smaller model. On a computer without a GPU this is expected; see [GPU and CPU configurations](#gpu-and-cpu-configurations). |
| `WARNING: the GPU profile is selected but the model is ...` | The model does not fit in GPU memory, and the part on the CPU makes the run much slower. Close other programs that use the GPU (`ollama ps` shows which models are loaded; `ollama stop MODEL` unloads one), use a GPU with more memory, or use `run_vlm_cpu.cmd`. The run continues. |
| `Model ... is loaded 100% CPU, for example by an earlier CPU run; unloading it ...` | Information only: the GPU profile found the model still in system memory and reloads it into GPU memory. |
| `The folder ... contains a quote character` | A quoted folder ended in a backslash, for example `"D:\Trail Photos\"`; Windows then reads the closing quote as part of the folder. Remove the last backslash: `"D:\Trail Photos"`. |
| `WARNING: the CPU profile is selected but Ollama placed the model ...` | Ollama reported a placement other than `100% CPU`. Check with `ollama ps`. The run continues. |
| `Error: Profile 'NAME' not found (available: ...)` | No `profiles\NAME.json` exists and NAME is not the path of a `.json` file. Use one of the listed names, or give the full path of your profile file. |
| Photos fail with a timeout error | The request took longer than `--timeout` (600 s; 3600 s with the CPU profile). Raise it, for example `--timeout 7200`. On a computer without a GPU, use `run_vlm_cpu.cmd`. Failed photos are retried on the next run. |
| Photos fail with `unparseable answer` | The model did not return valid JSON twice. Run again (failed photos are retried), or use another model. |
| Many rows have `consistency_problems` | The model's age groups or directions do not add up to its people count. Check those photos; the counts are exported as the model gave them. In validation 2 of 157 photos needed a correction request and none remained inconsistent. |
| Every event contains only one photo | Check `capture_time` and `time_source`. Without burst mode, or with a capture delay longer than `--event-gap`, events have one photo each. |
| Many events are flagged `times_from_file_dates` | The photos have no EXIF capture time, so file modification times were used. Copy the photos with their EXIF data intact. |
| Many events are flagged `event_sampled_for_model` | Events are longer than `--max-event-frames`. See [Long events](#long-events). |
| `--camera-pattern` is rejected or "The system cannot find the file specified" appears | The pattern needs a named group `(?P<camera>...)`. Put it in `settings.json`, or wrap it as `'"..."'` in PowerShell. |
| The window closes before messages can be read | When `run_vlm.cmd`, `run_vlm_gpu.cmd` or `run_vlm_cpu.cmd` is double-clicked (or a folder is dropped on it), it waits for Enter after an error. To keep all output, open PowerShell in this folder and run `.\run_vlm.cmd` there. |
| `Cannot open the folder of this launcher` | The tool folder is on a network path that the command shell cannot use. Copy the folder to a local drive. |

Run one analysis at a time: two runs share the GPU and each becomes slower.

## Development and license

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests replace the Ollama server with a fake one; they never contact a real server or use the GPU. They do not establish accuracy on camera photos.

The code is in `trailcam_vlm`: `prompts.py` (the questions and answer formats sent to the model), `ollama_client.py` (requests to Ollama, the device options and where the model runs), `photos.py` (photo discovery, capture times, camera ids and events), `export.py` (CSV columns), `v2_ages.py` (copies for version 2's evaluation script) and `__main__.py` (command line, settings, profiles and run). The hardware profiles are in `profiles`. If you change a prompt or answer format, change `PROMPT_VERSION` in `prompts.py`, so that cached answers to the old questions are not reused.

The project is distributed under [AGPL-3.0](LICENSE). Ollama and any model it runs are not part of this package and have their own licenses.
