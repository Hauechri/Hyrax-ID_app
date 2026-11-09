"""Audio standardization: mono selection, resampling, never touching originals, run_batch integration."""
import hashlib
import os
import tempfile
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import CONFIGS, REAL_SOUNDFILE

pytestmark = pytest.mark.skipif(not REAL_SOUNDFILE, reason="needs the real soundfile package")
import soundfile as sf

from hyrax_id.audio_standardize import (resolve_workers, standardize_file, standardize_files, to_mono)


def _tone(sr, seconds=2.0, freq=1000.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_hz(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spec) * sr / len(x)


def _digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------ channel selection
def test_to_mono_cases():
    tone = _tone(48000, 0.1)
    silent = np.zeros_like(tone)
    mono, msg = to_mono(tone)
    assert mono is tone and "Already mono" in msg
    out, msg = to_mono(np.stack([silent, tone], axis=1))                  # only channel 2 has signal
    assert np.array_equal(out, tone) and "channel 2 of 2" in msg
    out, msg = to_mono(np.stack([tone, silent], axis=1))
    assert np.array_equal(out, tone) and "channel 1 of 2" in msg
    out, msg = to_mono(np.stack([tone, 0.5 * tone], axis=1))              # both live: averaged
    assert np.allclose(out, 0.75 * tone) and "Averaged 2" in msg
    out, msg = to_mono(np.stack([silent, silent], axis=1))
    assert np.array_equal(out, silent) and "silent" in msg
    out, _ = to_mono(np.stack([tone, silent, 0.0004 * tone], axis=1))     # 3rd channel below threshold: ignored
    assert np.array_equal(out, tone)
    out, _ = to_mono(np.stack([tone], axis=1))                            # (n, 1)
    assert out.ndim == 1 and np.array_equal(out, tone)


def test_resolve_workers():
    assert resolve_workers(3) == 3 and resolve_workers(1) == 1
    assert resolve_workers(None) >= 1 and resolve_workers(0) >= 1


# ------------------------------------------------------------------------------ files
@pytest.fixture
def recordings(tmp_path):
    d = tmp_path / "in"; d.mkdir()
    tone44 = _tone(44100, 2.0, 1000.0)
    sf.write(d / "stereo_one_live.wav", np.stack([np.zeros_like(tone44), tone44], axis=1), 44100)
    sf.write(d / "mono_48k.wav", _tone(48000, 1.5, 3000.0), 48000)
    sf.write(d / "mono_22k.wav", _tone(22050, 1.0, 2000.0), 22050, subtype="FLOAT")
    sf.write(d / "empty.wav", np.zeros((0,), dtype=np.float32), 48000)
    (d / "corrupt.wav").write_bytes(b"this is not a wav file")
    return d


def _files(d):
    return {p.stem: str(p) for p in sorted(d.glob("*.wav"))}


def test_standardize_files_end_to_end_and_originals_untouched(recordings, tmp_path):
    before = {p: _digest(p) for p in map(str, recordings.glob("*.wav"))}
    out = tmp_path / "out"; lines = []
    s = standardize_files(_files(recordings), out, 48000, workers=3, log=lines.append)
    assert (s.n_ok, s.n_skipped, s.n_failed) == (3, 1, 1) and not s.cancelled
    assert set(s.outputs) == {"stereo_one_live", "mono_48k", "mono_22k"}
    assert {p.name for p in out.glob("*.wav")} == {"stereo_one_live.wav", "mono_48k.wav", "mono_22k.wav"}
    assert before == {p: _digest(p) for p in map(str, recordings.glob("*.wav"))}      # originals byte-identical

    for stem, hz, secs in (("stereo_one_live", 1000, 2.0), ("mono_48k", 3000, 1.5), ("mono_22k", 2000, 1.0)):
        x, sr = sf.read(out / f"{stem}.wav", dtype="float32")
        info = sf.info(out / f"{stem}.wav")
        assert sr == 48000 and x.ndim == 1 and info.subtype == "PCM_16"
        assert abs(len(x) - secs * 48000) <= 2                                        # duration preserved
        assert abs(_dominant_hz(x, sr) - hz) < 2                                      # content preserved
        assert 0.45 < np.abs(x).max() < 0.55                                          # level preserved (0.5 tone)
    text = "\n".join(lines)
    assert "Keeping active channel 2 of 2" in text and "Resampled 44100 Hz -> 48000 Hz" in text
    assert "Empty audio file" in text and "corrupt.wav" in text and "Standardized 3 of 5" in text


def test_worker_count_does_not_change_the_result(recordings, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    standardize_files(_files(recordings), a, 48000, workers=1, log=lambda m: None)
    standardize_files(_files(recordings), b, 48000, workers=4, log=lambda m: None)
    for p in a.glob("*.wav"):
        assert _digest(p) == _digest(b / p.name)


def test_other_target_rate(recordings, tmp_path):
    out = tmp_path / "out16"
    standardize_files({"mono_48k": str(recordings / "mono_48k.wav")}, out, 16000, log=lambda m: None)
    x, sr = sf.read(out / "mono_48k.wav")
    assert sr == 16000 and abs(len(x) - 1.5 * 16000) <= 2 and abs(_dominant_hz(x, sr) - 3000) < 2


def test_cancel_stops_before_work(recordings, tmp_path):
    ev = threading.Event(); ev.set()
    s = standardize_files(_files(recordings), tmp_path / "o", 48000, workers=2, cancel_event=ev, log=lambda m: None)
    assert s.cancelled and s.n_ok == 0 and not list((tmp_path / "o").glob("*.wav"))


def test_standardize_file_never_raises(tmp_path):
    r = standardize_file(tmp_path / "missing.wav", tmp_path / "x.wav", 48000)
    assert r.status == "ERROR" and r.out_path is None


# ------------------------------------------------------------------------------ run_batch integration
def _runner_with_fake_pipeline():
    import pipeline_runner as R
    seen = {}

    def fake(**kw):
        folder = kw["audio_folder"]
        seen["kw"] = kw
        seen["files"] = sorted(os.listdir(folder))
        seen["info"] = {f: sf.info(os.path.join(folder, f)) for f in seen["files"]}
        seen["folder"] = folder

    R.run_pipeline = fake
    return R, seen


def _cfg_path():
    return os.path.join(CONFIGS, "MELDELTAPCEN.json")


def test_run_batch_single_file_runs_on_a_standardized_copy_and_cleans_up(recordings, tmp_path, capsys):
    R, seen = _runner_with_fake_pipeline()
    src = recordings / "stereo_one_live.wav"; digest = _digest(src)
    R.run_batch(str(src), str(tmp_path / "o"), "d.pt", "gb", None, None, denoise=False, classify=False,
                preprocess_cfg_path=_cfg_path())
    assert seen["files"] == ["stereo_one_live.wav"]
    info = seen["info"]["stereo_one_live.wav"]
    assert info.samplerate == 48000 and info.channels == 1
    assert not os.path.exists(seen["folder"]), "temporary folder must be removed"
    assert _digest(src) == digest
    assert "Standardizing 1 file(s)" in capsys.readouterr().out


def test_run_batch_folder_skips_unusable_files_and_passes_settings(recordings, tmp_path):
    R, seen = _runner_with_fake_pipeline()
    R.run_batch(str(recordings), str(tmp_path / "o"), "d.pt", "gb", None, None, denoise=False, classify=False,
                preprocess_cfg_path=_cfg_path(), target_sample_rate=48000, audio_workers=2)
    assert seen["files"] == ["mono_22k.wav", "mono_48k.wav", "stereo_one_live.wav"]
    assert all(i.samplerate == 48000 and i.channels == 1 for i in seen["info"].values())


def test_run_batch_keeps_copies_when_asked(recordings, tmp_path):
    R, seen = _runner_with_fake_pipeline()
    keep = tmp_path / "kept"
    R.run_batch(str(recordings / "mono_22k.wav"), str(tmp_path / "o"), "d.pt", "gb", None, None, denoise=False,
                classify=False, preprocess_cfg_path=_cfg_path(), standardized_dir=str(keep))
    assert (keep / "mono_22k.wav").is_file() and sf.info(keep / "mono_22k.wav").samplerate == 48000


def test_run_batch_without_standardizing_keeps_the_old_behaviour(recordings, tmp_path):
    R, seen = _runner_with_fake_pipeline()
    src = recordings / "mono_22k.wav"
    # sf.info on the copy: still 22.05 kHz because nothing was converted
    R.run_batch(str(src), str(tmp_path / "o"), "d.pt", "gb", None, None, denoise=False, classify=False,
                preprocess_cfg_path=_cfg_path(), standardize_audio=False)
    assert seen["info"]["mono_22k.wav"].samplerate == 22050


def test_run_batch_errors_when_nothing_can_be_standardized(recordings, tmp_path):
    R, seen = _runner_with_fake_pipeline()
    with pytest.raises(RuntimeError, match="None of the audio files"):
        R.run_batch(str(recordings / "corrupt.wav"), str(tmp_path / "o"), "d.pt", "gb", None, None,
                    denoise=False, classify=False, preprocess_cfg_path=_cfg_path())
    assert "kw" not in seen


def test_run_batch_cancel_skips_the_pipeline(recordings, tmp_path):
    R, seen = _runner_with_fake_pipeline()
    ev = threading.Event(); ev.set()
    R.run_batch(str(recordings), str(tmp_path / "o"), "d.pt", "gb", None, None, denoise=False, classify=False,
                preprocess_cfg_path=_cfg_path(), cancel_event=ev)
    assert "kw" not in seen


def test_run_batch_warns_when_target_rate_differs_from_detector(recordings, tmp_path, capsys):
    R, seen = _runner_with_fake_pipeline()
    R.run_batch(str(recordings / "mono_48k.wav"), str(tmp_path / "o"), "d.pt", "gb", None, None, denoise=False,
                classify=False, preprocess_cfg_path=_cfg_path(), target_sample_rate=32000)
    assert "differs from the detector" in capsys.readouterr().out
