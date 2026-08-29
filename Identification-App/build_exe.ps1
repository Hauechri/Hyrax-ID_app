<#
Builds Hyrax-ID into a standalone Windows app with Nuitka.

Run from the project root (the folder containing app/, engine/, models/):
    .venv\Scripts\python.exe -m pip install -U nuitka ordered-set zstandard
    .\build_exe.ps1

Output lands in dist\main.dist\ -- that whole folder is the app; main.exe
inside it is what you ship/run. engine\ and models\ are copied in next to
main.exe as plain, uncompiled files (see the note in paths.py / worker.py:
pipeline_runner.py etc. are imported dynamically via a sys.path.insert done
by paths.py, which Nuitka's static import tracer can't see -- so they must
travel as data, not get compiled in).
#>

$ErrorActionPreference = "Stop"

# Sanity checks up front, so a typo'd path fails fast with a clear message
# instead of Nuitka choking on it 5 minutes into the build.
foreach ($p in @("app\main.py", "app\assets\icon.ico", "engine", "models")) {
    if (-not (Test-Path $p)) {
        throw "Expected '$p' relative to the current folder, but it doesn't exist. Run this script from the project root (the folder containing app\, engine\, models\)."
    }
}

$cpuCount = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

$nuitkaArgs = @(
    "app\main.py"
    "--standalone"
    "--output-dir=dist"
    "--output-filename=Hyrax-ID.exe"
    "--enable-plugin=pyside6"
    "--windows-icon-from-ico=app\assets\icon.ico"
    "--windows-console-mode=disable"
    "--include-data-dir=app\assets=assets"
    "--include-data-dir=engine=engine"
    "--include-data-dir=models=models"
    "--include-package=torch"
    "--include-package=torchaudio"
    "--include-package=librosa"
    "--include-package=xgboost"
    "--include-package=ultralytics"
    "--include-package=soundfile"
    "--include-package=resampy"
    "--include-package=numpy"
    "--include-package=numba"
    "--assume-yes-for-downloads"
    "--jobs=$cpuCount"
)

python -m nuitka @nuitkaArgs

# Native commands like `python` don't raise a PowerShell error on failure --
# they just set $LASTEXITCODE -- so check it explicitly or a failed build
# would silently fall through to the "finished" message below.
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka exited with code $LASTEXITCODE -- see the output above for the actual error."
}

Write-Host ""
Write-Host "Build finished. App is at: dist\main.dist\Hyrax-ID.exe"
Write-Host "Ship the whole dist\main.dist\ folder -- not just the .exe."