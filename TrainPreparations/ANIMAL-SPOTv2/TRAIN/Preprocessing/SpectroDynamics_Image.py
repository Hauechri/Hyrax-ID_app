import os
import numpy as np
import librosa
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.ndimage import zoom
#from stockwell.st import st
import cv2
from scipy.ndimage import median_filter, gaussian_filter
import scipy
from scipy import signal

#OUTPUT_AUDIO_DIR = Path("C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ACASelect/SEGMENTS/")
#OUTPUT_LABEL_DIR = Path("C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ACASelect/LABELS")

INPUT_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSELECT/SEGMENTS/"
OUTPUT_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSELECT/IMAGES/"

IMAGE_SIZE = (800, 800)
SAMPLE_RATE = 48000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def SpectroDynamics(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 60, width: int = 9):
    """
    Compute SpectroDynamics: base, delta, and delta-delta spectrograms.
    Returns:
        spec0: normalized log power spectrogram
        spec1: first-order delta (Δ)
        spec2: second-order delta (ΔΔ)
    """

    # === Base STFT and log power ===
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    S_db = librosa.power_to_db(S, ref=np.max)

    # === Deltas ===
    delta  = librosa.feature.delta(S_db, width=width, order=1)
    delta2 = librosa.feature.delta(S_db, width=width, order=2)

    # === Normalize each to [0, 1] ===
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    spec0 = norm(S_db)
    spec1 = norm(delta)
    spec2 = norm(delta2)

    return spec0, spec1, spec2

def resize(array: np.ndarray, image_size=(800, 800)) -> np.ndarray:
    target_h, target_w = image_size
    src_h, src_w = array.shape

    zoom_factors = (target_h / src_h, target_w / src_w)
    resized = zoom(array, zoom=zoom_factors, order=3)  # cubic interpolation

    return resized

def save_RGB_as_png_fast(R,G,B, output_path):
    """Save a 2D numpy spectrogram as a grayscale PNG compatible with YOLO."""
    # 1. Normalize to [0, 255]
    R = np.clip(R, R.min(), R.max())
    R = ((R - R.min()) / (R.max() - R.min()) * 255).astype(np.uint8)

    G = np.clip(G, G.min(), G.max())
    G = ((G - G.min()) / (G.max() - G.min()) * 255).astype(np.uint8)

    B = np.clip(B, B.min(), B.max())
    B = ((B - B.min()) / (B.max() - B.min()) * 255).astype(np.uint8)


    # 2. Stack into 3 channels (RGB)
    img_rgb = np.dstack([R, G, B])

    # 3. Save with OpenCV (faster than plt)
    cv2.imwrite(output_path, img_rgb)
    #cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

# === Main Processing Function ===
def process_audio_file(file_path: str, file_name: str):
    # Load the audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    file = os.path.splitext(file_name)[0]

    spec0, spec1, spec2 = SpectroDynamics(y,sr)
    # --- Spectrogram ---
    spec0 = resize(spec0, IMAGE_SIZE)
    # --- ΔSpectrogram ---
    spec1 = resize(spec1, IMAGE_SIZE)
    # --- ΔΔSpectrogram ---
    spec2 = resize(spec2, IMAGE_SIZE)



    output_file = file + ".png"
    output = os.path.join(OUTPUT_FOLDER, output_file)
    save_RGB_as_png_fast(spec0, spec1, spec2, output)


def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.wav')]
    print(f"Found {len(files)} audio files.")

    for file_name in tqdm(files, desc="Processing audio files"):
        file_path = os.path.join(INPUT_FOLDER, file_name)
        process_audio_file(file_path, file_name)

if __name__ == "__main__":
    main()