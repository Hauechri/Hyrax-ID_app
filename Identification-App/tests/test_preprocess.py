"""Engine tests: config loading, rendering parity with the YOLO app, detector time mapping, pipeline wiring.

Parity with the YOLO app itself runs when YOLO_APP_DIR points at the folder that contains `core/`
(the unzipped TRAIN folder); otherwise that one test is skipped.
"""
import glob
import importlib
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import CONFIGS
from hyrax_id.preprocess import (PreprocessConfig, PreprocessConfigError, find_settings_for_weights,
                                 SETTINGS_FILENAME)

CONFIG_FILES = sorted(glob.glob(os.path.join(CONFIGS, "*.json")))


def _audio(cfg, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(cfg.window_samples) / cfg.sample_rate
    y = 0.3 * np.sin(2 * np.pi * (2000 + 3000 * t / t[-1]) * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    return (y + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _load_json(name):
    with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("path", CONFIG_FILES, ids=[os.path.basename(p) for p in CONFIG_FILES])
def test_every_example_config_renders(path):
    cfg = PreprocessConfig.from_file(path)
    img = cfg.render(_audio(cfg))
    assert img.shape == (cfg.image.height, cfg.image.width, 3) and img.dtype == np.uint8
    assert img.std() > 0, "image is flat"
    assert cfg.imgsz == max(cfg.image.width, cfg.image.height)
    # a recording shorter / longer than the window must still give the same image size
    assert cfg.render(_audio(cfg)[: cfg.window_samples // 3]).shape == img.shape
    assert cfg.render(np.tile(_audio(cfg), 2)).shape == img.shape


@pytest.mark.skipif(not os.environ.get("YOLO_APP_DIR"), reason="set YOLO_APP_DIR to the unzipped TRAIN folder")
@pytest.mark.parametrize("path", CONFIG_FILES, ids=[os.path.basename(p) for p in CONFIG_FILES])
def test_pixel_identical_to_yolo_app(path):
    root = os.environ["YOLO_APP_DIR"]
    sys.path.insert(0, os.path.dirname(root))
    pkg = os.path.basename(root)
    ref_cfg = importlib.import_module(f"{pkg}.core.config")
    ref_compose = importlib.import_module(f"{pkg}.core.audio_image").compose
    with open(path, encoding="utf-8") as f:
        ai = ref_cfg.AudioImageConfig.from_dict(json.load(f)["audio_image"])
    cfg = PreprocessConfig.from_file(path)
    y = _audio(cfg, seed=3)
    ref, _ = ref_compose(y, ai.channels, ai.image)
    assert np.array_equal(cfg.render(y), ref)


def test_window_lengths_come_from_the_file():
    assert PreprocessConfig.from_file(os.path.join(CONFIGS, "MELDELTAPCEN.json")).window_samples == 240000
    assert PreprocessConfig.from_file(os.path.join(CONFIGS, "SigSpecPCEN_Strat.json")).window_samples == 144000


def test_imgsz_rule_matches_yolo_app():
    d = _load_json("MELDELTAPCEN.json")
    assert PreprocessConfig.from_dict(d).imgsz == 980                  # imgsz_auto -> max(w, h)
    d["train"]["imgsz_auto"] = False
    assert PreprocessConfig.from_dict(d).imgsz == 640                  # explicit imgsz
    d.pop("train")
    assert PreprocessConfig.from_dict(d).imgsz == 980                  # no train section: auto


def test_overlap_and_hop():
    cfg = PreprocessConfig.from_file(os.path.join(CONFIGS, "MELDELTAPCEN.json"))
    assert cfg.overlap_percent == 50.0 and cfg.hop_samples() == 120000
    assert cfg.hop_samples(0) == 240000 and cfg.hop_samples(75) == 60000
    with pytest.raises(PreprocessConfigError):
        cfg.hop_samples(100)
    no_inf = _load_json("Spectrodynamics.json")                         # this file has no inference section
    assert PreprocessConfig.from_dict(no_inf).hop_samples() == 120000   # falls back to 50 %


def test_bad_configs_give_clear_errors():
    d = _load_json("MELDELTAPCEN.json")
    with pytest.raises(PreprocessConfigError, match="audio_image"):
        PreprocessConfig.from_dict({"dataset": {}})
    bad = json.loads(json.dumps(d)); bad["audio_image"]["channels"][0]["representation"] = "bogus"
    with pytest.raises(PreprocessConfigError, match="unknown representation 'bogus'"):
        PreprocessConfig.from_dict(bad)
    bad = json.loads(json.dumps(d)); bad["audio_image"]["channels"][1]["params"]["nonsense"] = 1
    with pytest.raises(PreprocessConfigError, match="Channel G"):
        PreprocessConfig.from_dict(bad)
    bad = json.loads(json.dumps(d)); bad["audio_image"]["channels"].pop()
    with pytest.raises(PreprocessConfigError, match="three channels"):
        PreprocessConfig.from_dict(bad)
    bad = json.loads(json.dumps(d)); bad["version"] = 99
    with pytest.raises(PreprocessConfigError, match="newer"):
        PreprocessConfig.from_dict(bad)
    with pytest.raises(PreprocessConfigError, match="not found"):
        PreprocessConfig.from_file("/no/such/file.json")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{not json")
    with pytest.raises(PreprocessConfigError, match="Could not read"):
        PreprocessConfig.from_file(f.name)


def test_model_check_warnings():
    cfg = PreprocessConfig.from_file(os.path.join(CONFIGS, "MELDELTAPCEN.json"))
    assert cfg.check_model(["chuck", "wail", "snort"], 980) == []
    assert cfg.check_model(["chuck", "wail", "snort"], 992) == []         # same after rounding to stride 32
    assert any("classes" in w for w in cfg.check_model(["target"], 980))
    assert any("imgsz" in w for w in cfg.check_model(["chuck", "wail", "snort"], 640))
    assert cfg.check_model(["chuck", "wail", "snort"], None) == []        # unknown trained size: no complaint


def test_find_settings_next_to_weights():
    with tempfile.TemporaryDirectory() as run:
        os.makedirs(os.path.join(run, "weights"))
        assert find_settings_for_weights(os.path.join(run, "weights", "best.pt")) is None
        open(os.path.join(run, SETTINGS_FILENAME), "w").write("{}")
        assert find_settings_for_weights(os.path.join(run, "weights", "best.pt")).name == SETTINGS_FILENAME
        assert find_settings_for_weights("") is None


# ----------------------------------------------------------------------------------- detector
class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _FakeYolo:
    """Every window contains one box covering x = 0.2 .. 0.4 of the window."""
    names = {0: "chuck", 1: "wail", 2: "snort"}
    overrides = {"imgsz": 980}
    seen = {}

    def predict(self, source, **kw):
        _FakeYolo.seen = dict(kw, n=len(source), sizes={im.size for im in source}, modes={im.mode for im in source})
        res = []
        for _ in source:
            r = MagicMock()
            r.boxes.__len__ = lambda s: 1
            r.boxes.xyxyn, r.boxes.conf, r.boxes.cls = _T([[0.2, 0.1, 0.4, 0.9]]), _T([0.9]), _T([1.0])
            res.append(r)
        return res


def _detector(cfg, batch=2):
    from hyrax_id.detector import YoloDetector
    d = YoloDetector.__new__(YoloDetector)
    d.model, d.names, d.pre, d.batch_size, d.debug_saver = _FakeYolo(), _FakeYolo.names, cfg, batch, None
    d._kw = dict(conf=0.3, iou=0.7, imgsz=cfg.imgsz, verbose=False)
    d.trained_imgsz = 980
    return d


@pytest.mark.parametrize("name,win_s", [("MELDELTAPCEN.json", 5.0), ("SigSpecPCEN_Strat.json", 3.0)])
def test_detector_time_mapping_follows_window_length(name, win_s):
    cfg = PreprocessConfig.from_file(os.path.join(CONFIGS, name))
    sr = cfg.sample_rate
    sig = (np.random.default_rng(0).standard_normal(sr * 12) * 0.05).astype(np.float32)
    labels = _detector(cfg).detect(sig, sr, cfg.hop_samples())
    hop_s = cfg.hop_samples() / sr
    first = labels[0]
    assert first[2] == "wail"
    assert abs(first[0] - 0.2 * win_s) < 1e-6 and abs(first[1] - 0.4 * win_s) < 1e-6
    assert abs(labels[1][0] - (hop_s + 0.2 * win_s)) < 1e-6                 # next window starts one hop later
    assert _FakeYolo.seen["imgsz"] == 980 and _FakeYolo.seen["modes"] == {"RGB"}
    assert _FakeYolo.seen["sizes"] == {(cfg.image.width, cfg.image.height)}


def test_detector_resamples_other_sample_rates_and_clips_tail():
    cfg = PreprocessConfig.from_file(os.path.join(CONFIGS, "MELDELTAPCEN.json"))
    sig = (np.random.default_rng(1).standard_normal(44100 * 7) * 0.05).astype(np.float32)   # 7 s @ 44.1 kHz
    labels = _detector(cfg).detect(sig, 44100, cfg.hop_samples())
    assert all(0 <= a < b <= 7.0 + 1e-6 for a, b, *_ in labels)


# ----------------------------------------------------------------------------------- audio level
def test_read_wave_file_peak_normalize_option():
    from scipy.io import wavfile
    from hyrax_id.audio_io import read_wave_file
    x = (np.sin(np.linspace(0, 200, 48000)) * 8000).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, 48000, x)
    norm, _ = read_wave_file(f.name)
    raw, _ = read_wave_file(f.name, peak_normalize=False)
    assert abs(np.abs(norm).max() - 1.0) < 1e-6
    assert abs(np.abs(raw).max() - 8000 / 32767) < 1e-3


# ----------------------------------------------------------------------------------- pipeline wiring
def _patched_pipeline(tmp, bouts):
    import hyrax_id.pipeline as P
    P.joblib = MagicMock()
    P.joblib.load.return_value = {"model": None, "feature_keys": []}
    P.read_wave_file = lambda p, peak_normalize=True: (np.zeros(48000, dtype=np.float32), 48000)
    P.collect_wav_files = lambda f: {"rec1": os.path.join(tmp, "rec1.wav")}
    P.detect_elements_for_file = lambda *a, **k: ([], bouts)
    P.YoloDetector = lambda *a, **k: MagicMock(names={0: "chuck", 1: "wail", 2: "snort"}, trained_imgsz=980)
    return P


def test_run_batch_resolves_config_next_to_weights_and_writes_empty_label_file(capsys):
    import pipeline_runner as R
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "run"); os.makedirs(os.path.join(run, "weights"))
        open(os.path.join(tmp, "rec1.wav"), "wb").write(b"x")
        with open(os.path.join(CONFIGS, "SigSpecPCEN_Strat.json"), encoding="utf-8") as src, \
                open(os.path.join(run, SETTINGS_FILENAME), "w", encoding="utf-8") as dst:
            dst.write(src.read())
        P = _patched_pipeline(tmp, [])
        R.run_pipeline = P.run_pipeline
        out = os.path.join(tmp, "out")
        R.run_batch(os.path.join(tmp, "rec1.wav"), out, os.path.join(run, "weights", "best.pt"), "gb.joblib",
                    None, None, denoise=False, classify=False, standardize_audio=False,
                    window_size=48000, image_size=800)
        log = capsys.readouterr().out
        assert "no longer a setting" in log and "3 s windows" in log
        assert os.path.getsize(os.path.join(out, "rec1_labels.txt")) == 0
        assert "Detector windows: 3 s, hop 1.5 s (50% overlap)" in log
        assert "waveform channel" in log                                  # level-dependence hint


def test_run_batch_without_any_config_is_a_clear_error():
    import pipeline_runner as R
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "rec1.wav"), "wb").write(b"x")
        with pytest.raises(PreprocessConfigError, match="No preprocessing config"):
            R.run_batch(os.path.join(tmp, "rec1.wav"), tmp, os.path.join(tmp, "best.pt"), "gb", None, None,
                        denoise=False, classify=False)


def test_unknown_detector_class_fails_early():
    import hyrax_id.pipeline as P
    cfg = PreprocessConfig.from_file(os.path.join(CONFIGS, "MELDELTAPCEN.json"))
    P.joblib = MagicMock()
    P.YoloDetector = lambda *a, **k: MagicMock(names={0: "target"}, trained_imgsz=980)
    with pytest.raises(ValueError, match="garbage filter does not know"):
        P.load_models(detector_path="d", garbage_path="g", classifier_path=None, denoiser_path=None, preprocess=cfg,
                      detector_threshold=0.3, detector_iou=0.7, detector_batch_size=4, debug_saver=None,
                      speakers=None, input_dim=64, emb_dim=192, device=MagicMock(type="cpu"),
                      denoise=False, classify=False)
