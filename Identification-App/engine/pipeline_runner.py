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

from hyrax_id.audio_io import collect_wav_files
from hyrax_id.config import SPEAKER_LIST, SR  # noqa: F401  (re-exported)
from hyrax_id.pipeline import run_pipeline
from hyrax_id.preprocess import PreprocessConfig, PreprocessConfigError, find_settings_for_weights

# Calibration constants. These are tied to the trained garbage-filter/classifier checkpoints --
# change only if you know why.
#
# The detector's image geometry (window length, image size, channels, imgsz) is NOT a parameter here:
# it comes from the YOLO app's settings file (see hyrax_id/preprocess.py), so swapping the detector
# means swapping best.pt + its settings file, nothing else.
DEFAULT_PARAMS = dict(
    interarrival_threshold=0.724,
    detector_threshold=0.30,
    context_windowsize=5,
    GB_threshold=0.3,
    GB_scoreweight=0.3,
    Denoiser_sequence_length=1,
    Denoiser_num_worker=0,
    input_dim=64,
    emb_dim=192,
    detector_iou=None,          # None -> the settings file's value, else 0.7
    detector_batch_size=16,
)

# Old-pipeline settings that no longer exist. Callers (e.g. an old .cfg / GUI) get a warning
# instead of a crash, and the value is ignored.
_REMOVED_PARAMS = ("window_size", "hop_size", "image_size", "isMulti")


def resolve_preprocess(preprocess_cfg_path, detector_model_path) -> PreprocessConfig:
    """Load the preprocessing config. No explicit path -> the settings file stored next to the weights
    (animalspot_config.json, as saved by the YOLO app). Raises PreprocessConfigError with a clear message."""
    path = preprocess_cfg_path or find_settings_for_weights(detector_model_path)
    if not path:
        raise PreprocessConfigError(
            "No preprocessing config given and none found next to the detector weights. Choose the settings "
            "file of the YOLO app that trained the detector (animalspot_config.json or a 'Save settings' export).")
    cfg = PreprocessConfig.from_file(path)
    print(f"[INFO] Preprocessing config: {path}")
    return cfg


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

    If the run had classification disabled, bout headers are just "BOUT{n}"
    with no animal/confidence; those bouts come back with animal=None and
    confidence=None.

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
                    parts = label[len("BOUT"):].split("_", 2)
                    if len(parts) == 3:
                        bout_id_s, animal, conf_s = parts
                        confidence = float(conf_s)
                    else:
                        # classification was disabled: bare "BOUT{n}"
                        bout_id_s, animal, confidence = parts[0], None, None
                    current = {
                        "bout_id": int(bout_id_s),
                        "start": start,
                        "end": end,
                        "animal": animal,
                        "confidence": confidence,
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
    denoise=True,
    classify=True,
    preprocess_cfg_path=None,
    preprocess=None,
    window_overlap_percent=None,
    detector_peak_normalize=True,
    debug_image_dir=None,
    debug_max_images=50,
    **param_overrides,
):
    """
    Run the Hyrax-ID pipeline against either a single .wav file or a folder
    of .wav files -- audio_path may point at either.

    denoise / classify switch those two stages on or off. A disabled stage
    does not need its model path (denoiser_model_path /
    animal_classifier_model_path may be None); an enabled one does.

    run_pipeline() itself only knows how to scan a folder; a single file is
    staged into a temporary folder-of-one and cleaned up afterwards, so no
    pipeline logic is duplicated or forked.

    preprocess_cfg_path: the YOLO app's settings file for the detector (animalspot_config.json or a "Save
    settings" export). Omitted -> looked up next to detector_model_path. `preprocess` may be a ready
    PreprocessConfig instead. window_overlap_percent: None -> the settings file's value, else 50.
    detector_peak_normalize: see run_pipeline. debug_image_dir: if set, the first `debug_max_images`
    detector input images are saved there.
    """
    if denoise and not denoiser_model_path:
        raise ValueError("Denoising is enabled but no denoiser model path was given.")
    if classify and not animal_classifier_model_path:
        raise ValueError("Classification is enabled but no animal classifier model path was given.")

    for name in _REMOVED_PARAMS:
        if name in param_overrides:
            print(f"[WARN] run_batch: '{name}' is no longer a setting and is ignored "
                  f"(window/image geometry now comes from the preprocessing config).")
            param_overrides.pop(name)
    if preprocess is None:
        preprocess = resolve_preprocess(preprocess_cfg_path, detector_model_path)
    if window_overlap_percent is not None and not 0 <= window_overlap_percent < 100:
        raise ValueError("window_overlap_percent must be at least 0 and below 100.")

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
            Animal_Classifier_model_path=str(animal_classifier_model_path) if classify else None,
            Denoiser_model_path=str(denoiser_model_path) if denoise else None,
            audio_folder=folder,
            output_path=str(output_path),
            num_classes=len(SPEAKER_LIST),
            debug_denoised_dir=debug_denoised_dir,
            debug_image_dir=debug_image_dir,
            debug_max_images=debug_max_images,
            preprocess=preprocess,
            window_overlap_percent=window_overlap_percent,
            detector_peak_normalize=detector_peak_normalize,
            device_override=device_override,
            cancel_event=cancel_event,
            denoise=denoise,
            classify=classify,
            **params,
        )
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
