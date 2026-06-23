import json
import joblib
import numpy as np
import pandas as pd
import torch
from src.constants import PROJECT_ROOT


class PredictionPipeline:
    def __init__(self):
        self.model_path = PROJECT_ROOT / "artifacts" / "model_trainer" / "model.pkl"
        self.preprocessor_path = PROJECT_ROOT / "artifacts" / "data_transformation" / "preprocessor.pkl"
        self.metadata_path = PROJECT_ROOT / "artifacts" / "model_trainer" / "model_metadata.json"

    def _load(self):
        model = joblib.load(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)
        with open(self.metadata_path) as f:
            metadata = json.load(f)
        return model, preprocessor, metadata

    def _transform(self, preprocessor, recency, frequency, monetary):
        data = pd.DataFrame([[recency, frequency, monetary]],
                            columns=['Recency', 'Frequency', 'Monetary'])
        return preprocessor.transform(np.log1p(data))

    def predict(self, recency, frequency, monetary):
        cluster_id, _ = self.predict_with_distances(recency, frequency, monetary)
        return cluster_id

    def predict_with_distances(self, recency, frequency, monetary):
        """Return (cluster_id, distances, segment_info, model_name).

        Uses the active model chosen during evaluation. Segment info is
        looked up from model_metadata.json (metadata-driven, not hardcoded).
        """
        try:
            model, preprocessor, metadata = self._load()
            active = metadata["active_model"]
            model_info = metadata["models"][active]
            segments = model_info["segments"]

            x = self._transform(preprocessor, recency, frequency, monetary)[0].astype(float)

            if active == "kmeans":
                centroids = np.asarray(model_info["centroids_scaled"], dtype=float)
                distances = np.linalg.norm(centroids - x, axis=1).tolist()
                cluster_id = int(np.argmin(distances))
            elif active == "gmm":
                centroids = np.asarray(model_info["centroids_scaled"], dtype=float)
                distances = np.linalg.norm(centroids - x, axis=1).tolist()
                cluster_id = int(model.predict(x.reshape(1, -1))[0])
            elif active == "dbscan":
                # No native predict; nearest real cluster centroid, noise if too far.
                raw = model_info["centroids_scaled"]
                labels = [int(l) for l in raw.keys() if l != "-1"]
                centroids = np.asarray([raw[str(l)] for l in labels], dtype=float)
                d = np.linalg.norm(centroids - x, axis=1)
                idx = int(np.argmin(d))
                threshold = model_info.get("noise_threshold", float("inf"))
                if float(d[idx]) > float(threshold):
                    cluster_id = -1
                    distances = {str(labels[i]): float(d[i]) for i in range(len(labels))}
                    distances["-1"] = float(d[idx])
                else:
                    cluster_id = labels[idx]
                    distances = {str(labels[i]): float(d[i]) for i in range(len(labels))}
                # convert to a list-like for downstream (caller handles dict too)
                seg = segments.get(str(cluster_id), {})
                return cluster_id, distances, seg, active
            else:
                raise Exception(f"Unknown active model: {active}")

            seg = segments.get(str(cluster_id), {})
            # distances as dict keyed by cluster label
            dist_dict = {str(i): float(distances[i]) for i in range(len(distances))}
            return cluster_id, dist_dict, seg, active

        except Exception as e:
            raise e
