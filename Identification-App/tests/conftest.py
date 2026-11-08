"""Puts engine/ and app/ on sys.path and stubs heavy packages that are not installed (torch, soundfile,
ultralytics...), so the tests run anywhere. With the real packages installed nothing is stubbed."""
import importlib.util
import os
import sys
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "engine"), os.path.join(ROOT, "app")):
    if p not in sys.path:
        sys.path.insert(0, p)
for mod in ("torch", "torchaudio", "soundfile", "ultralytics", "xgboost"):
    if importlib.util.find_spec(mod) is None:
        stub = MagicMock()
        if mod == "torch":
            stub.Tensor = type("Tensor", (), {})      # SciPy's array-type checks call issubclass(x, torch.Tensor)
        sys.modules[mod] = stub
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CONFIGS = os.path.join(ROOT, "tests", "configs")
