"""Spectrogram viewer engine: label reading, measurements, rendering, CSV/PNG output, batch, and parity with
the Spectrogram-Viewer prototype (set SPECTROGRAM_VIEWER_DIR to the prototype's `app` folder to run that part).
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from conftest import REAL_SOUNDFILE, ROOT

pytest.importorskip("parselmouth")
pytest.importorskip("matplotlib")
pytest.importorskip("pandas")
pytestmark = pytest.mark.skipif(not REAL_SOUNDFILE, reason="needs the real soundfile package")

import pandas as pd

import spectro_data as D
from hyrax_id.spectro.library import (MEASUREMENT_COLUMNS, UNKNOWN_ANIMAL, BoutNotFoundError, RenderSettings,
                                      ResultsLibrary, load_audio, safe_name)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return D.make_dataset(str(tmp_path_factory.mktemp("spectro")))


@pytest.fixture
def lib(dataset, tmp_path):
    audio, res = dataset
    return ResultsLibrary(audio, res, output_dir=str(tmp_path / "out"))


# ------------------------------------------------------------------------------ reading the results
def test_library_lists_animals_recordings_bouts(lib):
    assert lib.list_animals() == ["FB", "QY", UNKNOWN_ANIMAL]
    assert lib.list_recordings("FB") == ["rec1"] and lib.list_bouts("FB", "rec1") == [1]
    assert lib.list_recordings("QY") == ["rec1"] and lib.list_bouts("QY", "rec1") == [2]
    assert lib.list_recordings(UNKNOWN_ANIMAL) == ["rec2"] and lib.list_bouts(UNKNOWN_ANIMAL, "rec2") == [1, 2]
    assert lib.n_recordings() == 2 and lib.n_bouts() == 4


def test_elements_are_capitalised_like_the_annotation_file(lib):
    els = lib.bout_elements("FB", "rec1", 1)
    assert [e["element"] for e in els] == ["Chuck", "Wail", "Chuck"]
    assert [(e["time_start"], e["time_end"]) for e in els] == [(1.0, 1.25), (1.5, 2.1), (2.4, 2.6)]
    assert lib.bout_span("FB", "rec1", 1) == (1.0, 2.6)
    assert np.isnan(els[0]["annotated_hzmax"]) and np.isnan(els[0]["annotated_hzmin"])


def test_problem_files_are_reported_not_fatal(lib):
    assert lib.without_audio == ["rec3"]            # labels but no recording
    assert lib.without_bouts == ["rec4"]            # empty label file
    assert "notes" not in lib._label_index          # stray .txt without a matching recording is ignored
    with pytest.raises(BoutNotFoundError, match="Available bouts"):
        lib.bout_elements("FB", "rec1", 99)


def test_other_txt_names_are_accepted_when_a_recording_matches(tmp_path, dataset):
    audio, _ = dataset
    res = tmp_path / "res"; res.mkdir()
    (res / "rec1.txt").write_text(D.labels_text("FB", "QY"), encoding="utf-8")       # e.g. an Audacity export
    lib = ResultsLibrary(audio, str(res), str(tmp_path / "o"))
    assert lib.list_animals() == ["FB", "QY"]


def test_missing_folders_give_an_empty_library(tmp_path):
    lib = ResultsLibrary(str(tmp_path / "nope"), str(tmp_path / "nada"), str(tmp_path / "o"))
    assert lib.list_animals() == [] and lib.n_bouts() == 0


def test_classification_names_with_underscores_and_unicode(tmp_path, dataset):
    audio, _ = dataset
    res = tmp_path / "res"; res.mkdir()
    txt = D.labels_text("My_Own_Animal", "ישראל")
    (res / "rec1_labels.txt").write_text(txt, encoding="utf-8")
    lib = ResultsLibrary(audio, str(res), str(tmp_path / "o"))
    assert lib.list_animals() == sorted(["My_Own_Animal", "ישראל"])


# ------------------------------------------------------------------------------ measurements
def test_measurements_match_the_synthetic_truth(lib):
    m = lib.measure_bout("FB", "rec1", 1, RenderSettings())
    truth = [(1500 + 2200) / 2, (1200 + 2600) / 2, (1800 + 1400) / 2]
    for el, expected in zip(m, truth):
        assert abs(el["f0_mean_hz"] - expected) / expected < 0.05, (el["element"], el["f0_mean_hz"], expected)
        assert el["auto_hzmin"] < el["f0_mean_hz"] < el["auto_hzmax"]
        assert all(np.isfinite(el[k]) for k in ("F1_hz", "F2_hz", "F3_hz"))
        assert el["boundaries"] == "detected" and el["formant_ceiling_hz"] == 7500.0


def test_csv_frame_has_the_prototypes_columns(lib):
    m = lib.measure_bout("QY", "rec1", 2, RenderSettings())
    df = lib.measurements_frame(m, "QY", "rec1", 2)
    assert list(df.columns) == ["singer", "song_id", "bout"] + MEASUREMENT_COLUMNS
    assert list(df.columns) == ["singer", "song_id", "bout", "element", "time_start", "time_end", "boundaries",
                                "annotated_hzmax", "annotated_hzmin", "auto_hzmax", "auto_hzmin", "f0_mean_hz",
                                "f0_median_hz", "F1_hz", "F2_hz", "F3_hz", "formant_ceiling_hz"]
    assert (df["singer"] == "QY").all() and (df["bout"] == 2).all() and df["annotated_hzmax"].isna().all()


def test_stereo_file_with_one_live_channel_is_not_silent(dataset):
    audio, _ = dataset
    fs, sig = load_audio(os.path.join(audio, "rec2.wav"))
    assert fs == 44100 and sig.ndim == 1 and abs(np.abs(sig).max() - 1.0) < 1e-9
    assert np.abs(sig[int(1.0 * fs): int(1.25 * fs)]).max() > 0.2        # the call is audible (channel 0 is silent)


def test_stereo_non_48k_recording_measures_correctly(lib):
    m = lib.measure_bout(UNKNOWN_ANIMAL, "rec2", 1, RenderSettings())
    for el, expected in zip(m, [1850, 1900, 1600]):
        assert abs(el["f0_mean_hz"] - expected) / expected < 0.06, (el["element"], el["f0_mean_hz"])


def test_silent_recording_gives_nan_not_a_crash(tmp_path, dataset):
    audio, _ = dataset
    res = tmp_path / "res"; res.mkdir()
    (res / "rec4_labels.txt").write_text(D.labels_text("X", "Y"), encoding="utf-8")   # bouts over silence
    lib = ResultsLibrary(audio, str(res), str(tmp_path / "o"))
    m = lib.measure_bout("X", "rec4", 1, RenderSettings())
    assert len(m) == 3 and all(np.isnan(e["f0_mean_hz"]) for e in m)


# ------------------------------------------------------------------------------ images and files
def test_render_and_file_names(lib):
    fig, meta = lib.render("FB", "rec1", 1, RenderSettings())
    ax = fig.axes[0]
    assert ax.get_title() == "Bout 1  |  Animal: FB  |  Recording: rec1"
    assert meta["t_start"] == 0.0 and abs(meta["t_end"] - 3.6) < 1e-9          # padding 1 s around 1.0 - 2.6
    png = lib.bout_path("FB", "rec1", 1, ".png")
    assert png == os.path.join(lib.output_dir, "FB", "rec1", "FB_rec1_bout_01.png")
    lib.save_figure(fig, png, 80)
    assert os.path.getsize(png) > 5000
    csv = lib.bout_path("FB", "rec1", 1, "_hz.csv")
    lib.export_measurements(meta["measurements"], csv, "FB", "rec1", 1)
    assert len(pd.read_csv(csv)) == 3


def test_animal_names_are_made_path_safe():
    assert safe_name("Bachelor-cut-in-40") == "Bachelor-cut-in-40" and safe_name("FB") == "FB"
    assert safe_name("a/b:c*d") == "a_b_c_d" and safe_name("x.") == "x" and safe_name("") == "_"


def test_option_validation(lib):
    with pytest.raises(ValueError, match="Freq max"):
        lib.render("FB", "rec1", 1, RenderSettings(freq_min_hz=3000, freq_max_hz=2000))


# ------------------------------------------------------------------------------ batch
def test_batch_single_process(dataset, tmp_path):
    audio, res = dataset
    lib = ResultsLibrary(audio, res, str(tmp_path / "out"))
    recs = [(a, r) for a in lib.list_animals() for r in lib.list_recordings(a)]
    log = []
    out = lib.run_batch(recs, RenderSettings(), workers=1, progress=lambda d, n, m: log.append((d, n)))
    assert out["bouts"] == out["total"] == 4 and out["errors"] == [] and not out["stopped"]
    assert log[-1] == (4, 4)
    rel = sorted(os.path.relpath(f, lib.output_dir) for f in out["files"])
    assert os.path.join("FB", "rec1", "FB_rec1_bout_01.png") in rel
    assert os.path.join("QY", "rec1", "QY_rec1_hz.csv") in rel
    assert os.path.join("unknown", "rec2", "unknown_rec2_hz.csv") in rel
    combined = [f for f in out["files"] if os.path.basename(f).startswith("batch_")]
    assert len(combined) == 1
    df = pd.read_csv(combined[0])
    assert len(df) == 3 + 2 + 3 + 2 and set(df["singer"]) == {"FB", "QY", "unknown"}


def test_batch_cancel_before_start(dataset, tmp_path):
    audio, res = dataset
    lib = ResultsLibrary(audio, res, str(tmp_path / "out"))
    out = lib.run_batch([("FB", "rec1")], RenderSettings(), workers=1, should_stop=lambda: True)
    assert out["stopped"] and out["files"] == [] and not (tmp_path / "out").exists()


def test_batch_csv_only_is_faster_path(dataset, tmp_path):
    audio, res = dataset
    lib = ResultsLibrary(audio, res, str(tmp_path / "out"))
    out = lib.run_batch([("FB", "rec1")], RenderSettings(), save_png=False, combined_csv=False, workers=1)
    assert [os.path.basename(f) for f in out["files"]] == ["FB_rec1_hz.csv"]


def test_batch_with_spawned_worker_processes(dataset, tmp_path):
    """The real thing, as on Windows: a script file, 2 spawned workers. Run in a subprocess so the test does
    not depend on how pytest was started."""
    audio, res = dataset
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {os.path.join(ROOT, 'engine')!r})
        from hyrax_id.spectro.library import ResultsLibrary, RenderSettings
        if __name__ == "__main__":
            lib = ResultsLibrary({audio!r}, {res!r}, {str(tmp_path / 'out_par')!r})
            recs = [(a, r) for a in lib.list_animals() for r in lib.list_recordings(a)]
            out = lib.run_batch(recs, RenderSettings(), workers=2)
            print("RESULT", out["bouts"], out["total"], len(out["errors"]), len(out["files"]), out["stopped"])
    """))
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    line = [l for l in run.stdout.splitlines() if l.startswith("RESULT")][0].split()
    assert line[1:] == ["4", "4", "0", "8", "False"]        # 4 PNG + 3 CSV per recording/animal + 1 combined CSV


# ------------------------------------------------------------------------------ parity with the prototype
@pytest.mark.skipif(not os.environ.get("SPECTROGRAM_VIEWER_DIR"), reason="set SPECTROGRAM_VIEWER_DIR to the prototype's app folder")
def test_identical_to_the_prototype(dataset, tmp_path):
    from PIL import Image
    sys.path.insert(0, os.environ["SPECTROGRAM_VIEWER_DIR"])
    import spectrogram_core as proto
    audio, res = dataset
    rows = [("FB", "rec1", 1, "Chuck", 1.0, 1.25), ("FB", "rec1", 1, "Wail", 1.5, 2.1), ("FB", "rec1", 1, "Chuck", 2.4, 2.6),
            ("QY", "rec1", 2, "Wail", 6.0, 6.7), ("QY", "rec1", 2, "Snort", 7.2, 7.4)]
    ann = pd.DataFrame(rows, columns=["Singer", "Song ID", "Bout number ", "Element", "Time start", "Time end "])
    ann["Hzmax"] = np.nan; ann["Hzmin"] = np.nan
    ann.to_excel(tmp_path / "ann.xlsx", index=False)
    P = proto.SpectrogramProject(audio, annotation_file=str(tmp_path / "ann.xlsx"), output_base_dir=str(tmp_path / "p"))
    L = ResultsLibrary(audio, res, str(tmp_path / "n"))
    sp, sn = proto.RenderSettings(show_annotation_overlay=False, hz_source="auto"), RenderSettings()
    cols = ["time_start", "time_end", "auto_hzmax", "auto_hzmin", "f0_mean_hz", "f0_median_hz", "F1_hz", "F2_hz", "F3_hz"]
    for animal, bout in (("FB", 1), ("QY", 2)):
        a, b = P.measure_bout(animal, "rec1", bout, sp), L.measure_bout(animal, "rec1", bout, sn)
        assert [e["element"] for e in a] == [e["element"] for e in b]
        assert np.array_equal(pd.DataFrame(a)[cols].to_numpy(float), pd.DataFrame(b)[cols].to_numpy(float))
        fa, fb = P.measurements_frame(a, animal, "rec1", bout), L.measurements_frame(b, animal, "rec1", bout)
        assert list(fa.columns) == list(fb.columns)
        figa, _ = P.render(animal, "rec1", bout, sp)
        figb, _ = L.render(animal, "rec1", bout, sn)
        P.save_figure(figa, str(tmp_path / "a.png"), 100)
        L.save_figure(figb, str(tmp_path / "b.png"), 100)
        A, B = np.asarray(Image.open(tmp_path / "a.png")), np.asarray(Image.open(tmp_path / "b.png"))
        assert A.shape == B.shape and np.array_equal(A, B)


def test_praat_measures_is_a_verbatim_copy_when_the_prototype_is_available():
    proto_dir = os.environ.get("SPECTROGRAM_VIEWER_DIR")
    if not proto_dir:
        pytest.skip("set SPECTROGRAM_VIEWER_DIR")
    ours = open(os.path.join(ROOT, "engine", "hyrax_id", "spectro", "praat_measures.py"), encoding="utf-8").read()
    theirs = open(os.path.join(proto_dir, "praat_measures.py"), encoding="utf-8").read()
    assert ours.replace("\r\n", "\n") == theirs.replace("\r\n", "\n")
