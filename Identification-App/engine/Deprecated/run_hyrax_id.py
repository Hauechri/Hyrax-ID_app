"""Entry point. Edit the SETTINGS block and run:  python run_hyrax_id.py"""
from hyrax_id.config import DetectorImageConfig, SPEAKER_LIST
from hyrax_id.pipeline import run_pipeline

if __name__ == "__main__":
    # === Detector image settings: read straight from the training UI settings file ===
    img_cfg = DetectorImageConfig.from_training_json(
        r"D:\Hyrax-Detect\MELDELTAPCEN.json")          # <- path to MELDELTAPCEN.json
    # (or just DetectorImageConfig() -- the defaults are the same values)

    hop_samples = img_cfg.window_samples // 2          # 50% overlap, as in the old inference

    interarrival_threshold = 0.724
    detector_threshold = 0.330
    context_windowsize = 5
    GB_threshold = 0.3
    GB_scoreweight = 0.3

    Denoiser_sequence_length = 1
    Denoiser_num_worker = 0
    Denoise = True
    Classify = True

    Detector_model_path = r"D:\Hyrax-Detect\MODELS\ACA_MELDELTAPCEN\MELDELTAPCEN_Strat\weights\best.pt"
    GarbageFilter_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/DETECTOR_GB/Boost/XGBRich.joblib"
    Animal_Classifier_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/ECAPA_TDNN/Final_ECAPA_TDNN_ATTENTIVE_ARCFACE/best_model.pt"
    Denoiser_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/01_ACS.pk"

    audio_folder = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/Audio/"
    output_path = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/HyraxID_ECAPA/"

    # Debug: first 50 detector input images -> compare with the training dataset images. None = off.
    debug_image_dir = r"D:\Hyrax-Identify\Base\Heloise_2024-2026\TMP"
    debug_denoised_dir = None

    run_pipeline(
        Detector_model_path=Detector_model_path,
        GarbageFilter_model_path=GarbageFilter_model_path,
        Animal_Classifier_model_path=Animal_Classifier_model_path,
        Denoiser_model_path=Denoiser_model_path,
        Denoiser_sequence_length=Denoiser_sequence_length,
        Denoiser_num_worker=Denoiser_num_worker,
        audio_folder=audio_folder,
        output_path=output_path,
        img_cfg=img_cfg,
        hop_samples=hop_samples,
        detector_threshold=detector_threshold,
        interarrival_threshold=interarrival_threshold,
        context_windowsize=context_windowsize,
        GB_threshold=GB_threshold,
        GB_scoreweight=GB_scoreweight,
        num_classes=len(SPEAKER_LIST),
        debug_image_dir=debug_image_dir,
        debug_denoised_dir=debug_denoised_dir,
        denoise=Denoise,
        classify=Classify,
    )
