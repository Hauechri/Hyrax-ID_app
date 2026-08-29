"""
Resource path resolution, dev vs. Nuitka-frozen. Also puts the vendored
engine on sys.path so `import pipeline_runner` etc. works the same way in
both modes.
"""
import os
import sys

if "__compiled__" in globals():
    # Nuitka --standalone build: the exe sits next to bundled models/ and
    # assets/ (see build_exe.ps1's --include-data-dir flags).
    APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ENGINE_DIR = os.path.join(APP_DIR, "engine")
MODELS_DIR = os.path.join(APP_DIR, "models")

DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "detector", "best.pt")
GARBAGE_FILTER_MODEL_PATH = os.path.join(MODELS_DIR, "boost", "XGBRich.joblib")
ANIMAL_CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "ecapa_tdnn", "best_model.pt")
DENOISER_ACA_MODEL_PATH = os.path.join(MODELS_DIR, "animal-clean", "01_ACS.pk")

if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)


def user_data_dir():
    """Writable, per-user directory for the downloaded denoiser cache etc.
    Survives a Nuitka --standalone reinstall, unlike APP_DIR."""
    import platformdirs

    path = platformdirs.user_data_dir("HyraxID", "HyraxID")
    os.makedirs(path, exist_ok=True)
    return path
