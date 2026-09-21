from pathlib import Path
import numpy as np
import soundfile as sf
import csv

# =====================================================
# Settings
# =====================================================

DENOISED_FOLDER = Path(
    r"C:/PHD/SideProjectsData/IcasspExperiments/ECAPA-TDNNExperiments/ACA/"
)
RESIDUAL_FOLDER = Path(
    r"C:/PHD/SideProjectsData/IcasspExperiments/ECAPA-TDNNExperiments/Residual/"
)

OUTPUT_CSV = Path(
    r"C:/PHD/SideProjectsData/IcasspExperiments/ECAPA-TDNNExperiments/SNR_Denoiser_Results.csv"
)

EPS = 1e-12  # Prevent divide-by-zero

# =====================================================

wav_files = sorted(DENOISED_FOLDER.glob("*.wav"))

results = []

# Totals used for duration-weighted overall SNR
total_signal_energy = 0.0
total_noise_energy = 0.0
total_samples = 0

for signal_path in wav_files:
    noise_path = RESIDUAL_FOLDER / signal_path.name

    if not noise_path.exists():
        print(f"Missing residual: {signal_path.name}")
        continue

    signal, sr1 = sf.read(signal_path)
    noise, sr2 = sf.read(noise_path)

    if sr1 != sr2:
        print(f"Sample rate mismatch: {signal_path.name}")
        continue

    # Trim signals to their common length, if needed
    if len(signal) != len(noise):
        L = min(len(signal), len(noise))
        signal = signal[:L]
        noise = noise[:L]

    # Convert stereo/multi-channel audio to mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    # Per-file powers
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    # Per-file SNR
    snr_db = 10 * np.log10((signal_power + EPS) / (noise_power + EPS))

    # Accumulate total energy for the overall SNR
    total_signal_energy += np.sum(signal ** 2)
    total_noise_energy += np.sum(noise ** 2)
    total_samples += len(signal)

    results.append([
        signal_path.name,
        signal_power,
        noise_power,
        snr_db
    ])

    print(f"{signal_path.name:40s}  {snr_db:8.2f} dB")

# Overall SNR across all valid files
if total_samples > 0:
    overall_signal_power = total_signal_energy / total_samples
    overall_noise_power = total_noise_energy / total_samples

    overall_snr_db = 10 * np.log10(
        (total_signal_energy + EPS) / (total_noise_energy + EPS)
    )

    print(f"\nOverall SNR across all files: {overall_snr_db:.2f} dB")
else:
    overall_signal_power = np.nan
    overall_noise_power = np.nan
    overall_snr_db = np.nan
    print("\nNo valid file pairs were found.")

# Save CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Filename",
        "SignalPower",
        "NoisePower",
        "SNR_dB"
    ])

    writer.writerows(results)

    # Blank row followed by the duration-weighted overall result
    writer.writerow([])
    writer.writerow([
        "OVERALL",
        overall_signal_power,
        overall_noise_power,
        overall_snr_db
    ])

print("\nFinished.")
print(f"Saved results to {OUTPUT_CSV}")