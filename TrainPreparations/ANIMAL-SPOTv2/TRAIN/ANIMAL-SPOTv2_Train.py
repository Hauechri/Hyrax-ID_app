from ultralytics import YOLO
import numpy as np
from pathlib import Path

# --- CONFIG ---
data_yaml = 'C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASET/BIODASet/data.yaml'
project_dir = Path("C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/Models/")
run_name = "BIODA_26_M_ICASSP"
max_epochs = 400
MODEL_WEIGHTS = "yolo26n.pt"#"D:/0_PHD/Hyrax/Models/BIODA_26_M_SpecDyn/weights/last.pt"#"yolo26n.pt"
if __name__ == "__main__":
    # --- MAIN TRAINING ---
    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=data_yaml,
        epochs=max_epochs,
        imgsz=800,
        batch=4,
        workers=2,
        name=run_name,
        project=str(project_dir),
        patience=20,
        device=0,
        cache=False,
        exist_ok=True,
        #resume=True
    )