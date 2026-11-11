"""Speaker list: loading, validation, wiring into run_batch / load_models / classifier, label round trip."""
import json
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import ROOT
from hyrax_id.speakers import (SpeakerList, SpeakerListError, explain_class_count_mismatch,
                               find_speaker_list_for_weights, SPEAKER_LIST_FILENAME)

LISTS = os.path.join(ROOT, "tests", "speakers")
CURRENT, VLAD = os.path.join(LISTS, "current.json"), os.path.join(LISTS, "VLAD.json")
MEL = os.path.join(ROOT, "tests", "configs", "MELDELTAPCEN.json")

OLD_HARDCODED = ['J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7',
                 'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0']
NEW_SET = {"Bachelor-cut-in-40", "Bachelor-earing-40-tear-1", "Bachelor-tear-20-40", "FB", "H1", "MS",
           "New-resident-in-David-never-caught", "No-ear-resident", "No-ear-sukkot-bachelor", "PA", "PS", "QY",
           "R1", "SN", "Unknow", "WX", "X2", "X3", "Z2"}


# ------------------------------------------------------------------------------ the two shipped lists
def test_vlad_reproduces_the_old_hardcoded_mapping_exactly():
    old_ids = {s: i for i, s in enumerate(sorted(OLD_HARDCODED))}        # what config.py used to build
    v = SpeakerList.from_file(VLAD)
    assert len(v) == 18 and v.name == "VLAD"
    assert v.speaker_to_id == old_ids
    assert v.id_to_speaker == {i: s for s, i in old_ids.items()}


def test_current_holds_the_new_speakers_in_alphabetical_class_order():
    c = SpeakerList.from_file(CURRENT)
    assert len(c) == 19 and c.name == "current" and set(c.speakers) == NEW_SET
    assert list(c.speakers) == sorted(NEW_SET)
    assert c.id_to_speaker[0] == "Bachelor-cut-in-40" and c.id_to_speaker[18] == "Z2"
    assert "19 speakers" in c.describe() and "(+14 more)" in c.describe()


# ------------------------------------------------------------------------------ format and validation
def test_formats_and_whitespace():
    assert SpeakerList.from_dict(["a", " b "]).speakers == ("a", "b")                       # plain list, stripped
    assert SpeakerList.from_dict({"speakers": ["x", "y"], "name": "n", "note": "ignored"}).name == "n"
    assert SpeakerList.from_dict(["x"], source="/p/my_list.json").name == "my_list"
    assert SpeakerList.from_dict(["Ünï", "ישראל"]).speakers == ("Ünï", "ישראל")             # non-ASCII names


@pytest.mark.parametrize("data,msg", [
    ([], "empty"), ({"speakers": []}, "empty"), ({"names": ["a"]}, "JSON list"), ("abc", "JSON list"),
    (["a", "a", "b"], "Duplicate speaker name.*a"), (["a", ""], "Entry 2"), (["a", 5], "Entry 2"),
    (["a", "b\tc"], "tab or line break"), (["a\nb"], "tab or line break"),
])
def test_invalid_lists_give_clear_errors(data, msg):
    with pytest.raises(SpeakerListError, match=msg):
        SpeakerList.from_dict(data)


def test_file_errors(tmp_path):
    with pytest.raises(SpeakerListError, match="not found"):
        SpeakerList.from_file(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"; bad.write_text("{oops")
    with pytest.raises(SpeakerListError, match="Could not read"):
        SpeakerList.from_file(bad)


def test_find_next_to_weights(tmp_path):
    w = tmp_path / "best_model.pt"; w.write_bytes(b"w")
    assert find_speaker_list_for_weights(w) is None and find_speaker_list_for_weights("") is None
    (tmp_path / SPEAKER_LIST_FILENAME).write_text("[\"a\"]")
    assert find_speaker_list_for_weights(w).name == "current.json"


def test_class_count_mismatch_message():
    err = ("Error(s) in loading state_dict: size mismatch for fc.weight: copying a param with shape "
           "torch.Size([19, 192]) from checkpoint, the shape in current model is torch.Size([18, 192]).")
    msg = explain_class_count_mismatch(err, 18)
    assert "trained for 19 speakers" in msg and "speaker list has 18" in msg
    assert "different number" in explain_class_count_mismatch("size mismatch for x", 3)
    assert explain_class_count_mismatch("CUDA out of memory", 3) is None


# ------------------------------------------------------------------------------ classifier + labels
def test_predict_animals_uses_the_given_mapping(monkeypatch):
    import hyrax_id.classifier as C
    monkeypatch.setattr(C, "load_audio_segment", lambda *a, **k: np.zeros(10, dtype=np.float32))
    monkeypatch.setattr(C, "waveform_to_logmel", lambda *a, **k: np.zeros((4, 4)))
    monkeypatch.setattr(C, "run_classifier_on_bout", lambda *a, **k: (3, 0.8))
    bouts = [{"bout_id": 1, "start": 0, "end": 1, "elements": [{"start": 0, "end": 1}]},
             {"bout_id": 2, "start": 2, "end": 3, "elements": []}]
    for path, expected in ((CURRENT, "FB"), (VLAD, "M9")):                   # class index 3 in each list
        out = C.predict_animals_for_file(np.zeros(10), 48000, [dict(b) for b in bouts], None, None,
                                         SpeakerList.from_file(path).id_to_speaker)
        assert out[0]["animal"] == expected and out[0]["animal_confidence"] == 0.8 and out[1]["animal"] == "unknown"
    out = C.predict_animals_for_file(np.zeros(10), 48000, [dict(bouts[0])], None, None, {0: "only"})
    assert out[0]["animal"] == "unknown_id_3"                                # index outside the list


@pytest.mark.parametrize("name", ["Bachelor-cut-in-40", "New-resident-in-David-never-caught", "My_Own_Animal", "ישראל"])
def test_labels_round_trip_with_any_name(tmp_path, name):
    from hyrax_id.labels_io import save_full_audacity_labels
    from pipeline_runner import parse_bout_results
    bouts = [{"bout_id": 7, "start": 1.0, "end": 4.0, "animal": name, "animal_confidence": 0.9123,
              "elements": [{"start": 1.0, "end": 2.0, "type": "chuck", "confidence": 0.8}]}]
    f = tmp_path / "x_labels.txt"
    save_full_audacity_labels(bouts, str(f))
    parsed = parse_bout_results(str(f))
    assert len(parsed) == 1 and parsed[0]["animal"] == name and parsed[0]["bout_id"] == 7
    assert parsed[0]["confidence"] == 0.912 and parsed[0]["elements"][0]["type"] == "chuck"


def test_bare_bout_without_classification_still_parses(tmp_path):
    from pipeline_runner import parse_bout_results
    f = tmp_path / "x.txt"; f.write_text("1.0\t2.0\tBOUT3\n1.0\t1.5\twail_0.700\n", encoding="utf-8")
    b = parse_bout_results(str(f))
    assert b[0]["bout_id"] == 3 and b[0]["animal"] is None and len(b[0]["elements"]) == 1


# ------------------------------------------------------------------------------ run_batch
def _fake_pipeline():
    import pipeline_runner as R
    seen = {}
    R.run_pipeline = lambda **kw: seen.update(kw)
    return R, seen


def _run(R, tmp_path, classify=True, **kw):
    wav = tmp_path / "a.wav"; wav.write_bytes(b"x")
    return R.run_batch(str(wav), str(tmp_path / "o"), "d.pt", "gb", kw.pop("classifier", str(tmp_path / "c" / "best_model.pt")),
                       None, denoise=False, classify=classify, standardize_audio=False, preprocess_cfg_path=MEL, **kw)


def test_run_batch_finds_current_json_next_to_the_classifier_weights(tmp_path, capsys):
    R, seen = _fake_pipeline()
    (tmp_path / "c").mkdir(); (tmp_path / "c" / "current.json").write_text(open(CURRENT, encoding="utf-8").read(), encoding="utf-8")
    _run(R, tmp_path)
    assert len(seen["speakers"]) == 19 and seen["speakers"].speakers[0] == "Bachelor-cut-in-40"
    assert "Speaker list:" in capsys.readouterr().out


def test_explicit_list_wins_over_current_json(tmp_path):
    R, seen = _fake_pipeline()
    (tmp_path / "c").mkdir(); (tmp_path / "c" / "current.json").write_text(open(CURRENT, encoding="utf-8").read(), encoding="utf-8")
    _run(R, tmp_path, speaker_list_path=VLAD)
    assert len(seen["speakers"]) == 18 and seen["speakers"].name == "VLAD"


def test_no_speaker_list_is_a_clear_error_only_when_classifying(tmp_path):
    R, seen = _fake_pipeline()
    with pytest.raises(SpeakerListError, match="No speaker list given"):
        _run(R, tmp_path)
    assert "audio_folder" not in seen                                        # failed before any work
    _run(R, tmp_path, classify=False)
    assert seen["speakers"] is None


def test_bad_list_file_is_reported(tmp_path):
    R, _ = _fake_pipeline()
    bad = tmp_path / "bad.json"; bad.write_text('["a","a"]')
    with pytest.raises(SpeakerListError, match="Duplicate"):
        _run(R, tmp_path, speaker_list_path=str(bad))


# ------------------------------------------------------------------------------ load_models
def _load_models(speakers, state_error=None):
    import hyrax_id.pipeline as P
    cls = MagicMock()
    cls.return_value.load_state_dict.side_effect = state_error
    fake = types.SimpleNamespace(ECAPATDNNBoutModel=cls)
    sys.modules["hyrax_id.model_ECAPATDNN"] = fake      # tried first (the file lives in hyrax_id/)
    sys.modules["model_ECAPATDNN"] = fake               # fallback for the older engine/ layout
    P.joblib = MagicMock(); P.joblib.load.return_value = {"model": None, "feature_keys": []}
    P.YoloDetector = lambda *a, **k: MagicMock(names={0: "chuck", 1: "wail", 2: "snort"}, trained_imgsz=980)
    from hyrax_id.preprocess import PreprocessConfig
    kw = dict(detector_path="d", garbage_path="g", classifier_path="c.pt", denoiser_path=None,
              preprocess=PreprocessConfig.from_file(MEL), detector_threshold=0.3, detector_iou=0.7,
              detector_batch_size=4, debug_saver=None, speakers=speakers, input_dim=64, emb_dim=192,
              device=MagicMock(type="cpu"), denoise=False, classify=True)
    return P.load_models(**kw), cls


def test_classifier_is_built_for_the_size_of_the_list():
    models, cls = _load_models(SpeakerList.from_file(CURRENT))
    assert cls.call_args.kwargs["num_classes"] == 19
    models, cls = _load_models(SpeakerList.from_file(VLAD))
    assert cls.call_args.kwargs["num_classes"] == 18


def test_wrong_list_for_the_weights_gives_an_understandable_error():
    err = RuntimeError("size mismatch for fc.weight: copying a param with shape torch.Size([18, 192]) from "
                       "checkpoint, the shape in current model is torch.Size([19, 192]).")
    with pytest.raises(ValueError, match="trained for 18 speakers, but the speaker list has 19"):
        _load_models(SpeakerList.from_file(CURRENT), state_error=err)
    with pytest.raises(RuntimeError, match="boom"):                          # unrelated errors pass through
        _load_models(SpeakerList.from_file(CURRENT), state_error=RuntimeError("boom"))


def test_run_pipeline_requires_speakers_when_classifying():
    import hyrax_id.pipeline as P
    with pytest.raises(ValueError, match="requires a speaker list"):
        P.run_pipeline(Detector_model_path="d", GarbageFilter_model_path="g", Animal_Classifier_model_path="c",
                       Denoiser_model_path=None, Denoiser_sequence_length=1, Denoiser_num_worker=0, audio_folder="x",
                       output_path="y", preprocess=None, detector_threshold=.3, interarrival_threshold=.7,
                       context_windowsize=5, GB_threshold=.3, GB_scoreweight=.3, denoise=False, classify=True)
