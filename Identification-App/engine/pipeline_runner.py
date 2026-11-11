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
from hyrax_id.audio_standardize import DEFAULT_SILENCE_THRESHOLD, standardize_files
from hyrax_id.config import SR
from hyrax_id.labels_io import parse_bout_results  # noqa: F401  (re-exported; the parser lives in labels_io)
from hyrax_id.pipeline import run_pipeline
from hyrax_id.preprocess import PreprocessConfig, PreprocessConfigError, find_settings_for_weights
from hyrax_id.speakers import SpeakerList, SpeakerListError, find_speaker_list_for_weights

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


def resolve_speakers(speaker_list_path, classifier_model_path) -> SpeakerList:
    """Load the classifier's speaker list. No explicit path -> `current.json` next to the classifier weights.
    Raises SpeakerListError with a clear message."""
    path = speaker_list_path or find_speaker_list_for_weights(classifier_model_path)
    if not path:
        raise SpeakerListError(
            "No speaker list given and no current.json found next to the classifier weights. Choose the JSON "
            "list of speaker names the classifier was trained with.")
    speakers = SpeakerList.from_file(path)
    print(f"[INFO] Speaker list: {path}")
    return speakers


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
    speaker_list_path=None,
    speakers=None,
    window_overlap_percent=None,
    detector_peak_normalize=True,
    standardize_audio=True,
    target_sample_rate=SR,
    audio_workers=None,
    silence_threshold=DEFAULT_SILENCE_THRESHOLD,
    standardized_dir=None,
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
    PreprocessConfig instead. speaker_list_path: JSON list of the classifier's speaker names (class index =
    position); omitted -> `current.json` next to the classifier weights; `speakers` may be a ready SpeakerList.
    Only needed when classify=True. window_overlap_percent: None -> the settings file's value, else 50.
    detector_peak_normalize: see run_pipeline. debug_image_dir: if set, the first `debug_max_images`
    detector input images are saved there.

    standardize_audio: first write mono / target_sample_rate / 16-bit copies of the input recordings
    (see hyrax_id/audio_standardize.py) and run the pipeline on those, so 44.1 kHz files or stereo files with
    one live channel just work. The originals are never modified. The copies go to a temporary folder that is
    removed afterwards, or to `standardized_dir` (kept; use a dedicated folder, the pipeline processes every .wav in it) if given.
    audio_workers: parallel files (None/0 = auto).
    silence_threshold: a channel whose peak is below it counts as silent.
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
    if classify and speakers is None:
        speakers = resolve_speakers(speaker_list_path, animal_classifier_model_path)
    if window_overlap_percent is not None and not 0 <= window_overlap_percent < 100:
        raise ValueError("window_overlap_percent must be at least 0 and below 100.")

    params = {**DEFAULT_PARAMS, **param_overrides}
    audio_path = Path(audio_path)

    if audio_path.is_file():
        sources = {audio_path.stem: str(audio_path)}
    elif audio_path.is_dir():
        sources = collect_wav_files(str(audio_path))
    else:
        raise FileNotFoundError(f"No such file or folder: {audio_path}")

    if standardize_audio and target_sample_rate not in (preprocess.sample_rate, SR):
        print(f"[WARN] Target sample rate {target_sample_rate} Hz differs from the detector's "
              f"{preprocess.sample_rate} Hz and the classifier's {SR} Hz; they will resample again.")

    work_dir, cleanup_dir = None, None
    try:
        if standardize_audio:
            if standardized_dir:
                work_dir = str(standardized_dir)
            else:
                work_dir = cleanup_dir = tempfile.mkdtemp(prefix="hyraxid_audio_")
            summary = standardize_files(sources, work_dir, target_sample_rate, audio_workers,
                                        silence_threshold, cancel_event)
            if summary.cancelled:
                print("[INFO] Cancelled.")
                return
            if summary.n_ok == 0:
                raise RuntimeError("None of the audio files could be standardized -- see the log for the reasons.")
            folder = work_dir
        elif audio_path.is_file():
            cleanup_dir = tempfile.mkdtemp(prefix="hyraxid_input_")
            shutil.copy2(audio_path, os.path.join(cleanup_dir, audio_path.name))
            folder = cleanup_dir
        else:
            folder = str(audio_path)

        run_pipeline(
            Detector_model_path=str(detector_model_path),
            GarbageFilter_model_path=str(garbage_filter_model_path),
            Animal_Classifier_model_path=str(animal_classifier_model_path) if classify else None,
            Denoiser_model_path=str(denoiser_model_path) if denoise else None,
            audio_folder=folder,
            output_path=str(output_path),
            speakers=speakers if classify else None,
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
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
