import os
from HYRAX_ID_predict import run_pipeline
if __name__ == "__main__":
    #Technically Dependent on Model so we shouldn't give them this information... Maybe I could Encode it as Model Information?
    #TODO Model Meta Information
    SPEAKER_LIST = [
        'J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7',
        'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0'
    ]

    # === SETTINGS ===
    window_size = 48000
    hop_size = 24000
    image_size = 800
    isMulti = True
    interarrival_threshold = 0.724
    detector_threshold = 0.330
    context_windowsize = 5

    GB_threshold = 0.3
    GB_scoreweight = 0.3

    Denoiser_sequence_length = 1
    Denoiser_num_worker = 0

    Detector_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/DETECTOR_GB/ACA_26_M_SpecDyn/weights/best.pt"
    GarbageFilter_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/DETECTOR_GB/Boost/XGBRich.joblib"
    #Animal_Classifier_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/TDNN_MHA/All_MHA/best_model.pt"
    Animal_Classifier_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/ECAPA_TDNN/Final_ECAPA_TDNN_ATTENTIVE_ARCFACE/best_model.pt"

    Denoiser_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/01_ACS.pk"

    audio_folder = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/Audio/"
    output_path = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/HyraxID_ECAPA/"
    debug_denoised_dir = None  # os.path.join(output_path, "denoised_tmp")


    run_pipeline(
        Detector_model_path=Detector_model_path,
        GarbageFilter_model_path=GarbageFilter_model_path,
        Animal_Classifier_model_path=Animal_Classifier_model_path,
        Denoiser_model_path=Denoiser_model_path,
        Denoiser_sequence_length=Denoiser_sequence_length,
        Denoiser_num_worker=Denoiser_num_worker,
        audio_folder=audio_folder,
        debug_denoised_dir=debug_denoised_dir,
        output_path=output_path,
        window_size=window_size,
        hop_size=hop_size,
        image_size=image_size,
        detector_threshold=detector_threshold,
        isMulti=isMulti,
        interarrival_threshold=interarrival_threshold,
        context_windowsize=context_windowsize,
        GB_threshold=GB_threshold,
        GB_scoreweight=GB_scoreweight,
        num_classes=len(SPEAKER_LIST),
        input_dim=64,
        emb_dim=128
    )