"""AnimalClean UNet denoiser (extracted unchanged from HYRAX_ID_predict.py)."""
import os
from collections import OrderedDict
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import scipy.io.wavfile as wav

from models.unet_model import UNet
from data.audiodataset import StridedAudioDataset
import data.transforms as T


def pad_to_multiple(x, mult=16):
    """Pad a (B, C, F, T) spectrogram so F and T are both multiples of `mult`."""
    _, _, f, t = x.shape
    pad_f = (mult - f % mult) % mult
    pad_t = (mult - t % mult) % mult
    return torch.nn.functional.pad(x, (0, pad_t, 0, pad_f)), pad_f, pad_t


def load_denoiser(model_path, device):
    """
    Loads the AnimalClean UNet denoiser from a plain model_path (mirrors
    predict.py's non-jit, non-checkpoint loading branch only, since that's
    the only mode the CLI is being run in here).
    """
    model_dict = torch.load(model_path, map_location="cpu")

    # legacy key fix (same fix as predict.py)
    if "outc.conv.0.weight" in model_dict["unet"]:
        w = model_dict["unet"]["outc.conv.0.weight"]
        b = model_dict["unet"]["outc.conv.0.bias"]
        model_dict["unet"]["outc.conv.weight"] = w
        model_dict["unet"]["outc.conv.bias"] = b
        del model_dict["unet"]["outc.conv.0.weight"]
        del model_dict["unet"]["outc.conv.0.bias"]

    model = UNet(1, 1, bilinear=False)
    model.load_state_dict(model_dict["unet"])
    model = nn.Sequential(OrderedDict([("denoiser", model)]))
    dataOpts = model_dict["dataOpts"]

    model = model.to(device)
    model.eval()

    return model, dataOpts


def denoise_single_file(model, dataOpts, audio_path, sequence_len_seconds, num_workers,
                         device, debug_output_path=None):
    """
    Denoises ONE audio file entirely in memory (no intermediate wav on disk
    for the pipeline to read back) and returns (signal, sr) ready to feed
    straight into detect_elements_for_file -- i.e. peak-normalized float32
    mono, matching exactly what read_wave_file used to hand the detector
    from a written-out wav.

    If `debug_output_path` is given, the denoised audio is ALSO written to
    disk there (int16 wav) purely for inspection. The pipeline itself never
    reads that file back -- it's for you to listen to / look at, not part
    of the data flow.

    Returns (None, None) if the file produced no output (e.g. empty or
    shorter than one chunk).
    """
    sr = dataOpts["sr"]
    n_fft = dataOpts["n_fft"]
    hop_length = dataOpts["hop_length"]
    n_freq_bins = dataOpts["n_freq_bins"]
    freq_cmpr = dataOpts["freq_compression"]

    fmin = 500
    fmax = 12000

    sequence_len = int(ceil(sequence_len_seconds * sr))
    hop = sequence_len  # no overlap between chunks, same as predict.py

    t_decompr_f = T.Decompress(
        f_min=fmin,
        f_max=fmax,
        n_fft=n_fft,
        sr=sr
    ).to(device)
    window = torch.hann_window(n_fft)
    if device.type == "cuda":
        window = window.cuda()

    dataset = StridedAudioDataset(
        audio_path,
        sequence_len=sequence_len,
        hop=hop,
        sr=sr,
        fft_size=n_fft,
        fft_hop=hop_length,
        n_freq_bins=n_freq_bins,
        f_min=fmin,
        f_max=fmax,
        freq_compression=freq_cmpr,
        center=True,
        min_max_normalize=False,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_audio = None

    with torch.no_grad():
        for batch in data_loader:
            _, input_spec, spec_cmplx, _ = batch

            if device.type == "cuda":
                input_spec = input_spec.cuda()
                spec_cmplx = spec_cmplx.cuda()

            input_spec, pad_f, pad_t = pad_to_multiple(input_spec, mult=16)
            denoised = model(input_spec)

            if pad_f > 0:
                denoised = denoised[:, :, :-pad_f, :]
            if pad_t > 0:
                denoised = denoised[:, :, :, :-pad_t]

            decompressed = t_decompr_f(denoised)

            spec_cmplx = spec_cmplx.squeeze(0)
            decompressed = decompressed.unsqueeze(-1)

            audio_spec = decompressed * spec_cmplx
            audio_spec = audio_spec.squeeze(0).transpose(0, 1)
            audio_spec = torch.view_as_complex(audio_spec)

            audio_chunk = torch.istft(
                audio_spec,
                n_fft,
                hop_length=hop_length,
                onesided=True,
                center=True,
                window=window,
            )

            total_audio = audio_chunk if total_audio is None else torch.cat((total_audio, audio_chunk), dim=0)

            del denoised, decompressed, audio_spec, audio_chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if total_audio is None:
        return None, None

    total_audio_np = total_audio.cpu().numpy().astype(np.float32)

    if debug_output_path is not None:
        os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)
        debug_int16 = (total_audio_np * np.iinfo(np.int16).max).astype(np.int16)
        wav.write(debug_output_path, sr, debug_int16)

    # Match read_wave_file's normalization exactly (peak -> [-1, 1], mono)
    # so detect_elements_for_file sees the same kind of input it always has.
    if total_audio_np.ndim > 1:
        total_audio_np = total_audio_np.mean(axis=1)
    peak = np.max(np.abs(total_audio_np))
    if peak > 0:
        total_audio_np = total_audio_np / peak

    return total_audio_np, sr
