# Setup

## Python version — use 3.12, not whatever `python`/`py` defaults to

This machine's default `python`/`py` resolves to Python 3.14. **Do not use it.**
Nuitka's `--mingw64` flag hard-fails on Python 3.13+ (`Alireza_Spectrogram_Viewer`'s
last build hit exactly this: `FATAL: Error, cannot use '--mingw64' on Python
version 3.13 or higher`). Always create the venv with an explicit 3.12 interpreter:

```powershell
C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
```

(Adjust the path if Python 3.12 lives elsewhere on your machine — just don't
let it resolve to `python`/`py` on PATH without checking `python --version` first.)

## Install dependencies

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU-only PyTorch is installed by default. If you have an NVIDIA GPU and want
CUDA acceleration:

```powershell
.venv\Scripts\pip uninstall torch torchaudio torchvision -y
.venv\Scripts\pip install -r requirements-cuda.txt
```

## The denoiser model

The first pipeline stage needs a UNet denoiser checkpoint (`01_ACS.pk`, several
tens of MB) that is **not** committed to this repo (too large, and not ours to
redistribute in git). The app downloads it automatically on first run and
caches it locally. If you need it manually for a CLI test, the direct
download link is:

```
https://faubox.rrze.uni-erlangen.de/dl/fi93Jos1YBfsvkeAwkRMwv/01_ACS.pk
```

(Note: the share-page URL in the original project notes,
`.../getlink/fi93Jos1YBfsvkeAwkRMwv/...`, returns an HTML landing page, not
the file itself — use the `/dl/` link above for a direct download.)

## Running in development

```powershell
.venv\Scripts\python.exe app\main.py
```

## Building the Windows executable

```powershell
.\build_exe.ps1
```

Produces `build\main.dist\HyraxID.exe`. Known gotchas:

- Must be built from the Python 3.12 venv (see above) — `--mingw64` fails on 3.13+.
- MSVC Build Tools are available on this machine as a fallback (`--msvc=latest`)
  if `--mingw64` ever regresses.
- xgboost loads a native shared library via its own path-finding logic —
  verify the GarbageCollector filter step actually works in the compiled
  exe, not just that the GUI launches.
- numba/llvmlite JIT-cache to disk; the app sets `NUMBA_CACHE_DIR` at startup
  to a writable app-data directory so this doesn't try to write into a
  possibly read-only install directory.
