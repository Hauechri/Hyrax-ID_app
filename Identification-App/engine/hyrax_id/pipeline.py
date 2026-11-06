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
from .audio2image import DebugImageSaver
from .config import DetectorImageConfig, SR
from .detection import detect_elements_for_file
from .detector import YoloDetector
from .labels_io import save_full_audacity_labels
from .postprocess import build_bout_data


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


def load_models(*, detector_path, garbage_path, classifier_path, denoiser_path, img_cfg, detector_threshold,
                detector_iou, detector_batch_size, debug_saver, num_classes, input_dim, emb_dim,
                device, denoise, classify) -> Models:
    if denoise:
        from .denoiser import load_denoiser
        d_model, d_opts = load_denoiser(denoiser_path, device)
        print(f"[INFO] Denoiser loaded: {denoiser_path}")
    else:
        d_model = d_opts = None
        print("[INFO] Denoising disabled -- the detector will run on the original audio.")

    detector = YoloDetector(detector_path, img_cfg, conf=detector_threshold, iou=detector_iou,
                            device=None if device.type == "cpu" else str(device),
                            batch_size=detector_batch_size, debug_saver=debug_saver)
    print(f"[INFO] Detector loaded: {detector_path}  (classes: {detector.names}, imgsz={img_cfg.size}, "
          f"window={img_cfg.window_s:.2f}s)")

    gc_data = joblib.load(garbage_path)
    print(f"[INFO] GarbageCollector loaded: {garbage_path}")

    classifier = None
    if classify:
        from model_ECAPATDNN import ECAPATDNNBoutModel
        classifier = ECAPATDNNBoutModel(num_classes=num_classes, input_dim=input_dim, emb_dim=emb_dim)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.to(device).eval()
        print(f"[INFO] Animal classifier loaded: {classifier_path}")
    else:
        print("[INFO] Classification disabled -- labels will contain bouts and elements only (no animal ID).")

    return Models(detector, gc_data["model"], list(gc_data["feature_keys"]), d_model, d_opts, classifier)


def get_detector_signal(models: Models, path, denoise, denoiser_seq_len, denoiser_workers, device,
                        debug_denoised_dir, stem):
    """Returns (signal, sr) for the detector: denoised or original, peak-normalised mono float32."""
    if denoise:
        from .denoiser import denoise_single_file
        debug_path = os.path.join(debug_denoised_dir, f"{stem}.wav") if debug_denoised_dir else None
        return denoise_single_file(models.denoiser, models.denoiser_opts, path, denoiser_seq_len,
                                   denoiser_workers, device, debug_output_path=debug_path)
    return read_wave_file(path)


def run_pipeline(
    Detector_model_path,
    GarbageFilter_model_path,
    Animal_Classifier_model_path,
    Denoiser_model_path,
    Denoiser_sequence_length,
    Denoiser_num_worker,
    audio_folder,
    output_path,
    img_cfg: DetectorImageConfig,
    hop_samples,
    detector_threshold,
    interarrival_threshold,
    context_windowsize,
    GB_threshold,
    GB_scoreweight,
    num_classes,
    input_dim=64,
    emb_dim=192,
    detector_iou=0.7,
    detector_batch_size=16,
    debug_denoised_dir=None,
    debug_image_dir=None,
    debug_max_images=50,
    device_override=None,
    cancel_event=None,
    denoise=True,
    classify=True,
):
    """
    img_cfg:       DetectorImageConfig -- the single source of truth for window length, image size and
                   imgsz (replaces the old window_size / image_size arguments).
    hop_samples:   window hop in samples at img_cfg.sample_rate (120000 = 50% overlap of 5 s windows).
    debug_image_dir: if set, the first `debug_max_images` detector input images are saved there.
    debug_denoised_dir: if set, every denoised file is also written there as wav.
    """
    if denoise and not Denoiser_model_path:
        raise ValueError("denoise=True requires Denoiser_model_path.")
    if classify and not Animal_Classifier_model_path:
        raise ValueError("classify=True requires Animal_Classifier_model_path.")

    device = select_device(device_override)
    print(f"[INFO] Using device: {device}")

    debug_saver = DebugImageSaver(debug_image_dir, debug_max_images) if debug_image_dir else None
    models = load_models(
        detector_path=Detector_model_path, garbage_path=GarbageFilter_model_path,
        classifier_path=Animal_Classifier_model_path, denoiser_path=Denoiser_model_path,
        img_cfg=img_cfg, detector_threshold=detector_threshold, detector_iou=detector_iou,
        detector_batch_size=detector_batch_size, debug_saver=debug_saver, num_classes=num_classes,
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
                                                Denoiser_num_worker, device, debug_denoised_dir, stem)
        if signal is None or signal.size == 0:
            print(f"  [WARN] {stem}: no audio to process (empty / too short), skipping.")
            continue
        if signal_sr != img_cfg.sample_rate:
            print(f"  [WARN] {stem}: sample rate {signal_sr} != {img_cfg.sample_rate}; resampling for the detector.")

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
            bout_data = predict_animals_for_file(original_waveform, sr, bout_data, models.classifier, device)


        save_full_audacity_labels(bout_data, label_output_path)
        print(f"  ✓ Labels saved to {label_output_path}")
