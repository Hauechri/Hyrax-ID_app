"""run_pipeline: denoise (optional) -> YOLO detect -> garbage filter -> bouts -> classify (optional) -> labels.

Each stage lives in its own module; this file only wires them together. Heavy optional stages
(denoiser, classifier) are imported lazily, so they are only needed when enabled.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any

import joblib
import torch

from .audio_io import collect_wav_files, read_wave_file, load_audio
from .config import SR
from .debug import DebugImageSaver
from .detection import detect_elements_for_file
from .detector import YoloDetector
from .garbage_filter import LABEL_MAP
from .labels_io import save_full_audacity_labels
from .postprocess import build_bout_data
from .preprocess import PreprocessConfig
from .speakers import SpeakerList, explain_class_count_mismatch


@dataclass
class Models:
    detector: YoloDetector
    garbagecollector: Any
    feature_keys: list
    denoiser: Any = None
    denoiser_opts: Any = None
    classifier: Any = None


def select_device(device_override=None) -> torch.device:
    if device_override in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_override)


def load_models(*, detector_path, garbage_path, classifier_path, denoiser_path, preprocess, detector_threshold,
                detector_iou, detector_batch_size, debug_saver, speakers, input_dim, emb_dim,
                device, denoise, classify) -> Models:
    if denoise:
        from .denoiser import load_denoiser
        d_model, d_opts = load_denoiser(denoiser_path, device)
        print(f"[INFO] Denoiser loaded: {denoiser_path}")
    else:
        d_model = d_opts = None
        print("[INFO] Denoising disabled -- the detector will run on the original audio.")

    detector = YoloDetector(detector_path, preprocess, conf=detector_threshold, iou=detector_iou,
                            device=None if device.type == "cpu" else str(device),
                            batch_size=detector_batch_size, debug_saver=debug_saver)
    print(f"[INFO] Detector loaded: {detector_path}")
    print(f"[INFO] Classes: {detector.names}")
    print(f"[INFO] Preprocessing: {preprocess.describe()}")
    for warning in preprocess.check_model(detector.names.values(), detector.trained_imgsz):
        print(f"[WARN] {warning}")
    unknown = sorted(str(n) for n in detector.names.values() if str(n).lower() not in LABEL_MAP)
    if unknown:
        raise ValueError(f"The detector has class(es) {unknown} that the garbage filter does not know "
                         f"(it handles {sorted(LABEL_MAP)}). This pipeline needs a detector with those classes.")

    gc_data = joblib.load(garbage_path)
    print(f"[INFO] GarbageCollector loaded: {garbage_path}")

    classifier = None
    if classify:
        try:
            from .model_ECAPATDNN import ECAPATDNNBoutModel   # lives next to this module
        except ImportError:                                   # older layout: engine/model_ECAPATDNN.py
            from model_ECAPATDNN import ECAPATDNNBoutModel
        classifier = ECAPATDNNBoutModel(num_classes=len(speakers), input_dim=input_dim, emb_dim=emb_dim)
        try:
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        except RuntimeError as exc:
            hint = explain_class_count_mismatch(str(exc), len(speakers))
            if hint:
                raise ValueError(hint) from exc
            raise
        classifier.to(device).eval()
        print(f"[INFO] Animal classifier loaded: {classifier_path}")
        print(f"[INFO] Speaker list: {speakers.describe()}")
    else:
        print("[INFO] Classification disabled -- labels will contain bouts and elements only (no animal ID).")

    return Models(detector, gc_data["model"], list(gc_data["feature_keys"]), d_model, d_opts, classifier)


def get_detector_signal(models: Models, path, denoise, denoiser_seq_len, denoiser_workers, device,
                        debug_denoised_dir, stem, peak_normalize=True):
    """Returns (signal, sr) for the detector: denoised or original, mono float32 (peak-normalised if asked)."""
    if denoise:
        from .denoiser import denoise_single_file
        debug_path = os.path.join(debug_denoised_dir, f"{stem}.wav") if debug_denoised_dir else None
        return denoise_single_file(models.denoiser, models.denoiser_opts, path, denoiser_seq_len,
                                   denoiser_workers, device, debug_output_path=debug_path,
                                   peak_normalize=peak_normalize)
    return read_wave_file(path, peak_normalize=peak_normalize)


def run_pipeline(
    Detector_model_path,
    GarbageFilter_model_path,
    Animal_Classifier_model_path,
    Denoiser_model_path,
    Denoiser_sequence_length,
    Denoiser_num_worker,
    audio_folder,
    output_path,
    preprocess: PreprocessConfig,
    detector_threshold,
    interarrival_threshold,
    context_windowsize,
    GB_threshold,
    GB_scoreweight,
    speakers: SpeakerList | None = None,
    input_dim=64,
    emb_dim=192,
    window_overlap_percent=None,
    detector_iou=None,
    detector_batch_size=16,
    debug_denoised_dir=None,
    debug_image_dir=None,
    debug_max_images=50,
    device_override=None,
    cancel_event=None,
    denoise=True,
    classify=True,
    detector_peak_normalize=True,
):
    """
    speakers:      SpeakerList for the classifier (class index -> name); required when classify=True.
    preprocess:    PreprocessConfig loaded from the YOLO app's settings file -- the single source of truth for
                   window length, image size, channels and imgsz.
    window_overlap_percent: overlap of consecutive detector windows (None = the settings file's value, else 50).
    detector_iou:  NMS IoU inside one window (None = the settings file's value, else 0.7).
    detector_peak_normalize: scale the detector's audio to peak 1. Switch off for configs with a 'waveform'
                   channel if the training audio was not peak-normalised (the waveform image depends on level).
    debug_image_dir: if set, the first `debug_max_images` detector input images are saved there.
    debug_denoised_dir: if set, every denoised file is also written there as wav.
    """
    if denoise and not Denoiser_model_path:
        raise ValueError("denoise=True requires Denoiser_model_path.")
    if classify and not Animal_Classifier_model_path:
        raise ValueError("classify=True requires Animal_Classifier_model_path.")
    if classify and speakers is None:
        raise ValueError("classify=True requires a speaker list (speakers=SpeakerList...).")

    device = select_device(device_override)
    print(f"[INFO] Using device: {device}")
    hop_samples = preprocess.hop_samples(window_overlap_percent)
    overlap_used = 100.0 * (1 - hop_samples / preprocess.window_samples)
    print(f"[INFO] Detector windows: {preprocess.window_s:g} s, hop {hop_samples / preprocess.sample_rate:g} s "
          f"({overlap_used:.0f}% overlap)")
    if detector_iou is None:
        detector_iou = preprocess.iou if preprocess.iou is not None else 0.7
    if preprocess.uses_absolute_amplitude and detector_peak_normalize:
        print("[INFO] This preprocessing has a waveform channel, whose image depends on the signal level, and the "
              "detector audio is peak-normalised. If detection is poor, try switching off "
              "'detector_peak_normalize' (the YOLO app used the audio level as it is).")

    debug_saver = DebugImageSaver(debug_image_dir, debug_max_images) if debug_image_dir else None
    models = load_models(
        detector_path=Detector_model_path, garbage_path=GarbageFilter_model_path,
        classifier_path=Animal_Classifier_model_path, denoiser_path=Denoiser_model_path,
        preprocess=preprocess, detector_threshold=detector_threshold, detector_iou=detector_iou,
        detector_batch_size=detector_batch_size, debug_saver=debug_saver, speakers=speakers,
        input_dim=input_dim, emb_dim=emb_dim, device=device, denoise=denoise, classify=classify)

    os.makedirs(output_path, exist_ok=True)
    if debug_denoised_dir is not None:
        os.makedirs(debug_denoised_dir, exist_ok=True)

    originals = sorted(collect_wav_files(audio_folder).items())
    print(f"[INFO] Found {len(originals)} audio file(s) to process.")

    for file_index, (stem, original_path) in enumerate(originals):
        if cancel_event is not None and cancel_event.is_set():
            print("[INFO] Cancelled.")
            break
        print(f"\n[FILE {file_index + 1}/{len(originals)}] Processing: {stem}")

        signal, signal_sr = get_detector_signal(models, original_path, denoise, Denoiser_sequence_length,
                                                Denoiser_num_worker, device, debug_denoised_dir, stem,
                                                peak_normalize=detector_peak_normalize)
        if signal is None or signal.size == 0:
            print(f"  [WARN] {stem}: no audio to process (empty / too short), skipping.")
            continue
        if signal_sr != preprocess.sample_rate:
            print(f"  [WARN] {stem}: sample rate {signal_sr} != {preprocess.sample_rate}; resampling for the detector.")

        elements, bouts = detect_elements_for_file(
            models.detector, models.garbagecollector, models.feature_keys, signal, signal_sr, hop_samples,
            interarrival_threshold, context_windowsize, GB_threshold, GB_scoreweight)

        label_output_path = os.path.join(output_path, f"{stem}_labels.txt")
        if not bouts:
            save_full_audacity_labels([], label_output_path)
            print(f"  [INFO] No bouts detected for {stem}; wrote an empty label file: {label_output_path}")
            continue
        bout_data = build_bout_data(elements, bouts)

        if classify:
            from .classifier import predict_animals_for_file
            original_waveform, sr = load_audio(original_path)       # classifier always uses the ORIGINAL audio
            if sr != SR:
                print(f"  [WARN] {stem}: sample rate {sr} != expected {SR}")
            bout_data = predict_animals_for_file(original_waveform, sr, bout_data, models.classifier, device,
                                                 speakers.id_to_speaker)


        save_full_audacity_labels(bout_data, label_output_path)
        print(f"  ✓ Labels saved to {label_output_path}")
