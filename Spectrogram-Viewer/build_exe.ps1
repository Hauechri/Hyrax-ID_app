# Builds a standalone Windows exe with Nuitka.
# Output: build\main.dist\SpectrogramViewer.exe (+ all its bundled dependencies
# in that same folder). Zip up the whole main.dist folder to hand it to someone.
# Data (annotation file, audio) is not bundled; users pick it in the app.
#
# Usage:  .\build_exe.ps1   (run from this folder, with the venv set up)

$ErrorActionPreference = "Stop"

.venv\Scripts\python.exe -m nuitka `
    --standalone `
    --low-memory `
    --enable-plugins=pyside6 `
    --include-package=parselmouth `
    --windows-console-mode=disable `
    --windows-icon-from-ico=app\assets\icon.ico `
    --include-data-dir=app\assets=assets `
    --product-name="Acoustic Spectrogram Viewer" `
    --file-description="Acoustic Spectrogram Viewer" `
    --file-version=0.2.0.0 `
    --product-version=0.2.0.0 `
    --output-filename=SpectrogramViewer.exe `
    --output-dir=build `
    --assume-yes-for-downloads `
    app\main.py
