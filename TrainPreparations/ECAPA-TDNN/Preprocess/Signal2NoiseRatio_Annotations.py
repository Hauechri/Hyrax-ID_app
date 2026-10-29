from pathlib import Path
import numpy as np
import soundfile as sf
import csv

# ============================================================
# Settings
# ============================================================

AUDIO_FOLDER = Path(
    r"C:/PHD/SideProjectsData/IcasspExperiments/ECAPA-TDNNExperiments/Audio/"
)
LABEL_FOLDER = Path(
    r"C:/PHD/SideProjectsData/IcasspExperiments/ECAPA-TDNNExperiments/GTLabels/"
)
OUTPUT_CSV = Path(
    r"C:/PHD/SideProjectsData/IcasspExperiments/ECAPA-TDNNExperiments/SNR_Annotations_Results.csv"
)

# Padding added before/after every annotation (seconds)
PADDING = 0.005  # 5 ms

# Ignore labels containing this string
IGNORE_LABEL = "bout"

EPS = 1e-12

# ============================================================


def load_audacity_labels(label_path):
    """
    Returns a list of (start_time, end_time) intervals,
    excluding labels that contain IGNORE_LABEL.
    """
    intervals = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            start = float(parts[0])
            end = float(parts[1])
            label = " ".join(parts[2:])

            if IGNORE_LABEL.lower() in label.lower():
                continue

            intervals.append((start, end))

    return intervals


# ============================================================
# Process files
# ============================================================

results = []

# Accumulate separately for annotated (target + noise) and
# unannotated (noise-only) regions.
total_mixture_energy = 0.0
total_noise_energy = 0.0
total_mixture_samples = 0
total_noise_samples = 0

total_mixture_duration = 0.0
total_noise_duration = 0.0
total_annotations = 0

wav_files = sorted(AUDIO_FOLDER.glob("*.wav"))

print(f"Found {len(wav_files)} audio files.")

for wav_path in wav_files:
    label_path = LABEL_FOLDER / (wav_path.stem + ".txt")

    if not label_path.exists():
        print(f"[Missing Label] {wav_path.name}")
        continue

    audio, sr = sf.read(wav_path)

    # Stereo / multi-channel -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    n_samples = len(audio)

    # --------------------------------------------------------
    # Build mask for annotated target + noise regions
    # --------------------------------------------------------

    mask = np.zeros(n_samples, dtype=bool)

    intervals = load_audacity_labels(label_path)

    for start, end in intervals:
        start -= PADDING
        end += PADDING

        start = max(start, 0.0)
        end = min(end, n_samples / sr)

        s0 = int(round(start * sr))
        s1 = int(round(end * sr))

        # Ensure indexes stay within the audio bounds
        s0 = max(0, min(s0, n_samples))
        s1 = max(0, min(s1, n_samples))

        if s1 > s0:
            mask[s0:s1] = True

    # --------------------------------------------------------
    # Annotated audio = target + noise
    # Non-annotated audio = noise-only estimate
    # --------------------------------------------------------

    mixture = audio[mask]
    noise = audio[~mask]

    mixture_duration = len(mixture) / sr
    noise_duration = len(noise) / sr

    if len(mixture) == 0:
        print(f"[No Target+Noise Region] {wav_path.name}")
        continue

    if len(noise) == 0:
        print(f"[No Noise-Only Region] {wav_path.name}")
        continue

    # Power in annotated regions: target + noise
    mixture_power = np.mean(mixture ** 2)

    # Noise power estimated from unannotated regions
    noise_power = np.mean(noise ** 2)

    # Under the assumption that target and noise are uncorrelated:
    #
    # P(target + noise) = P(target) + P(noise)
    #
    # Therefore:
    target_power = mixture_power - noise_power

    # Imperfect noise estimates can lead to small negative values.
    target_power = max(target_power, 0.0)

    mixture_rms = np.sqrt(mixture_power)
    noise_rms = np.sqrt(noise_power)
    target_rms = np.sqrt(target_power)

    # Estimated target-to-noise ratio
    snr_db = 10 * np.log10((target_power + EPS) / (noise_power + EPS))

    results.append([
        wav_path.name,
        len(intervals),
        mixture_duration,
        noise_duration,
        mixture_power,
        noise_power,
        target_power,
        mixture_rms,
        noise_rms,
        target_rms,
        snr_db
    ])

    # Accumulate for pooled overall values
    total_mixture_energy += np.sum(mixture ** 2)
    total_noise_energy += np.sum(noise ** 2)
    total_mixture_samples += len(mixture)
    total_noise_samples += len(noise)

    total_mixture_duration += mixture_duration
    total_noise_duration += noise_duration
    total_annotations += len(intervals)

    print(f"{wav_path.name:40s}  {snr_db:7.2f} dB")

# ============================================================
# Compute pooled overall result
# ============================================================

if total_mixture_samples > 0 and total_noise_samples > 0:
    # Average powers across all annotated and unannotated samples
    overall_mixture_power = total_mixture_energy / total_mixture_samples
    overall_noise_power = total_noise_energy / total_noise_samples

    # Estimate target power by subtracting estimated noise power
    # from target+noise power. Do not subtract RMS values.
    overall_target_power = max(
        overall_mixture_power - overall_noise_power,
        0.0
    )

    overall_mixture_rms = np.sqrt(overall_mixture_power)
    overall_noise_rms = np.sqrt(overall_noise_power)
    overall_target_rms = np.sqrt(overall_target_power)

    overall_snr_db = 10 * np.log10(
        (overall_target_power + EPS) / (overall_noise_power + EPS)
    )

    print(f"\nOverall estimated SNR: {overall_snr_db:.2f} dB")
else:
    overall_mixture_power = np.nan
    overall_noise_power = np.nan
    overall_target_power = np.nan

    overall_mixture_rms = np.nan
    overall_noise_rms = np.nan
    overall_target_rms = np.nan

    overall_snr_db = np.nan

    print("\nNo valid audio/annotation pairs were found.")

# ============================================================
# Save CSV
# ============================================================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Filename",
        "NumAnnotations",
        "TargetPlusNoiseDuration_s",
        "NoiseOnlyDuration_s",
        "TargetPlusNoisePower",
        "NoisePower",
        "EstimatedTargetPower",
        "TargetPlusNoiseRMS",
        "NoiseRMS",
        "EstimatedTargetRMS",
        "EstimatedSNR_dB"
    ])

    writer.writerows(results)

    writer.writerow([])

    writer.writerow([
        "OVERALL",
        total_annotations,
        total_mixture_duration,
        total_noise_duration,
        overall_mixture_power,
        overall_noise_power,
        overall_target_power,
        overall_mixture_rms,
        overall_noise_rms,
        overall_target_rms,
        overall_snr_db
    ])

print("\nFinished!")
print(f"Results written to:\n{OUTPUT_CSV}")