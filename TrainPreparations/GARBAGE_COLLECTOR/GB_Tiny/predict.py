import joblib
from dataset import features_to_matrix


class GarbageCollector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def filter(self, feature_list, detections):
        X = features_to_matrix(feature_list)
        preds = self.model.predict(X)

        # keep only valid detections
        filtered = [
            det for det, p in zip(detections, preds) if p == 1
        ]

        return filtered