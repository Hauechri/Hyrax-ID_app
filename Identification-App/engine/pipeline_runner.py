"""
Qt-free entry point wrapping run_pipeline() for both single-file and
folder-batch input. Callers (CLI smoke tests, the GUI worker thread) import
this instead of calling run_pipeline() directly, so single-file staging
never needs to be reimplemented.
"""
import os
import shutil
import tempfile
from pathlib import Path

from HYRAX_ID_predict import run_pipeline, SPEAKER_LIST, SR, collect_wav_files  # noqa: F401  (re-exported)

# Calibration constants, matching HYRAX_ID_predict.py's own __main__ defaults.
# These are tied to the trained detector/classifier checkpoints -- change
# only if you know why.
DEFAULT_PARAMS = dict(
    window_size=48000,
    hop_size=24000,
    image_size=800,
    isMulti=True,
    interarrival_threshold=0.724,
    detector_threshold=0.330,
    context_windowsize=5,
    GB_threshold=0.3,
    GB_scoreweight=0.3,
    Denoiser_sequence_length=1,
    Denoiser_num_worker=0,
    input_dim=64,
    emb_dim=192,
)


def list_input_stems(audio_path):
    """
    Stems of the .wav file(s) a run_batch() call with this audio_path will
    actually process. Callers should use this -- not a blind glob of
    output_path -- to decide which *_labels.txt files belong to this run.
    output_path is not necessarily empty: it may already contain unrelated
    files (a previous run, a leftover ground-truth annotation with a
    different label format, etc.), and picking those up would misattribute
    results or crash parse_bout_results on a format it was never meant to
    read.
    """
    audio_path = Path(audio_path)
    if audio_path.is_file():
        return [audio_path.stem]
    return sorted(collect_wav_files(str(audio_path)).keys())


def parse_bout_results(label_path):
    """
    Parse a *_labels.txt file written by run_pipeline() into structured bout
    dicts: [{"bout_id", "start", "end", "animal", "confidence", "elements": [...]}, ...]

    NOTE: HYRAX_ID_predict.py's own parse_annotation_file()/assign_elements_to_bouts()
    target an older ground-truth label format ("BOUT_1", no animal/confidence
    suffix) and raise ValueError on the pipeline's actual output format
    ("BOUT1_{animal}_{conf}") -- confirmed by inspection, not assumption.
    This parses the real output format instead.

    Relies on save_full_audacity_labels() always writing each bout's element
    lines immediately after that bout's own header line (true because a
    bout's start time ties with its first element's start time, and Python's
    stable sort preserves the original append order on ties) -- verified
    against real pipeline output.
    """
    bouts = []
    current = None
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                start_s, end_s, label = line.split("\t")
                start, end = float(start_s), float(end_s)
                if label.startswith("BOUT"):
                    bout_id_s, animal, conf_s = label[len("BOUT"):].split("_", 2)
                    current = {
                        "bout_id": int(bout_id_s),
                        "start": start,
                        "end": end,
                        "animal": animal,
                        "confidence": float(conf_s),
                        "elements": [],
                    }
                    bouts.append(current)
                elif current is not None:
                    elem_type, conf_s = label.rsplit("_", 1)
                    current["elements"].append(
                        {"start": start, "end": end, "type": elem_type, "confidence": float(conf_s)}
                    )
            except ValueError:
                # Defensive only: a line that doesn't match this run's own
                # output format (e.g. this file wasn't actually written by
                # save_full_audacity_labels). list_input_stems() is the real
                # fix for not reading stray files in the first place; this
                # just keeps one bad line from losing the whole file's results.
                continue
    return bouts


def run_batch(
    audio_path,
    output_path,
    detector_model_path,
    garbage_filter_model_path,
    animal_classifier_model_path,
    denoiser_model_path,
    device_override=None,
    cancel_event=None,
    debug_denoised_dir=None,
    **param_overrides,
):
    """
    Run the Hyrax-ID pipeline against either a single .wav file or a folder
    of .wav files -- audio_path may point at either.

    run_pipeline() itself only knows how to scan a folder; a single file is
    staged into a temporary folder-of-one and cleaned up afterwards, so no
    pipeline logic is duplicated or forked.
    """
    params = {**DEFAULT_PARAMS, **param_overrides}
    audio_path = Path(audio_path)

    staging_dir = None
    try:
        if audio_path.is_file():
            staging_dir = tempfile.mkdtemp(prefix="hyraxid_input_")
            shutil.copy2(audio_path, os.path.join(staging_dir, audio_path.name))
            folder = staging_dir
        elif audio_path.is_dir():
            folder = str(audio_path)
        else:
            raise FileNotFoundError(f"No such file or folder: {audio_path}")

        run_pipeline(
            Detector_model_path=str(detector_model_path),
            GarbageFilter_model_path=str(garbage_filter_model_path),
            Animal_Classifier_model_path=str(animal_classifier_model_path),
            Denoiser_model_path=str(denoiser_model_path),
            audio_folder=folder,
            output_path=str(output_path),
            num_classes=len(SPEAKER_LIST),
            debug_denoised_dir=debug_denoised_dir,
            device_override=device_override,
            cancel_event=cancel_event,
            **params,
        )
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
