"""Animal (speaker) classifier stage (extracted unchanged)."""
import numpy as np
import torch
import torchaudio

from .audio_io import load_audio_segment


def waveform_to_logmel(
    waveform,
    sr=48000,
    n_mels=96,
    n_fft=1024,
    hop_length=144,  # 3 ms at 48kHz
    win_length=1024,
    f_min=50,
    f_max=20000,
    eps=1e-6
):
    """Convert waveform (1, T) -> log-mel spectrogram (T_frames, n_mels), used for the classifier."""
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0
    )
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    # Failsafe: torch.stft's default center=True reflect-pads by n_fft//2 on
    # each side, which requires the input to already be longer than that pad.
    # Very short bout elements (a few ms) can be shorter than n_fft and blow
    # this up. Zero-pad up to n_fft samples so STFT always has enough signal.
    min_len = n_fft
    if waveform.shape[-1] < min_len:
        pad_amount = min_len - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    mel_spec = mel_spec_transform(waveform)      # (1, n_mels, T)
    log_mel = torch.log(mel_spec + eps)
    log_mel = log_mel.squeeze(0).transpose(0, 1)  # (T, n_mels)
    return log_mel


def run_classifier_on_bout(model, mel_list, device):
    """
    mel_list: list of (T_i, n_mels) tensors, one per element in the bout
    (variable length, unpadded — matches training in train_tdnn.py, where
    `elements` is a list-of-lists of un-padded per-element tensors and the
    model is always called with batch size 1).

    TDNNBoutModel.forward(self, batch_elements) expects batch_elements to be
    a list of bouts, where each bout is itself a list of (T, F) tensors.
    Since we only ever classify one bout at a time here, we wrap mel_list in
    an outer list of length 1.
    """
    elements = [[e.to(device) for e in mel_list]]  # batch of size 1

    with torch.no_grad():
        logits = model(elements)  # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred].item())

    return pred, confidence


def predict_animals_for_file(original_waveform, sr, bout_data, model, device, id_to_speaker):
    """
    For each bout, extract its elements from the ORIGINAL (non-denoised)
    audio, compute log-mel spectrograms, and classify the bout.

    id_to_speaker maps the classifier's output index to the speaker name (see speakers.SpeakerList).

    Adds "animal" and "animal_confidence" keys to each bout dict in place,
    and returns the (mutated) bout_data list.
    """
    for bout in bout_data:
        if not bout["elements"]:
            bout["animal"] = "unknown"
            bout["animal_confidence"] = 0.0
            continue

        mel_list = []
        for e in bout["elements"]:
            segment = load_audio_segment(original_waveform, sr, e["start"], e["end"])
            mel = waveform_to_logmel(segment, sr=sr)
            mel_list.append(mel)

        pred, confidence = run_classifier_on_bout(model, mel_list, device)

        bout["animal"] = id_to_speaker.get(pred, f"unknown_id_{pred}")
        bout["animal_confidence"] = confidence

    return bout_data
