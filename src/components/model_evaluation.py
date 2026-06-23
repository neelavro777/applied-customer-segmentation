import os
import json
import shutil
import numpy as np
import logging
import joblib
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.entity.config_entity import ModelEvaluationConfig


def _safe_metrics(X, labels):
    """Silhouette & Davies-Bouldin. DBSCAN noise points are excluded."""
    labels = np.asarray(labels)
    mask = labels != -1
    # Need at least 2 real clusters and >1 sample to compute.
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return None, None
    sil = silhouette_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    return float(sil), float(db)


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def log_into_mlflow(self):
        try:
            logging.info("Starting Model Evaluation (K-Means + DBSCAN + GMM)")

            with open(self.config.metadata_path) as f:
                metadata = json.load(f)

            test = np.load(self.config.test_data_path)
            X = test[:, 1:]
            # DBSCAN has no predict(); score it on the training set it was fit on.
            train = np.load(self.config.root_dir.parent / "data_transformation" / "train.npy")
            X_train = train[:, 1:]

            model_metrics = {}
            for name, info in metadata["models"].items():
                model = joblib.load(info["model_path"])
                if name == "kmeans":
                    labels = model.predict(X)
                    sil, db = _safe_metrics(X, labels)
                elif name == "gmm":
                    labels = model.predict(X)
                    sil, db = _safe_metrics(X, labels)
                elif name == "dbscan":
                    # No native predict; refit on train and score there.
                    labels = model.fit_predict(X_train)
                    sil, db = _safe_metrics(X_train, labels)
                else:
                    continue
                model_metrics[name] = {
                    "silhouette_score": sil,
                    "davies_bouldin_index": db,
                    "n_clusters": info["n_clusters"],
                }
                logging.info(f"[{name}] silhouette={sil}, davies_bouldin={db}, k={info['n_clusters']}")

            # Pick the best model by silhouette score (higher = better), but
            # require at least 3 real clusters so segments stay actionable for
            # marketing. If no model meets that, fall back to best silhouette.
            scored = {n: m for n, m in model_metrics.items() if m["silhouette_score"] is not None}
            if not scored:
                raise Exception("No model could be evaluated with silhouette score.")
            granular = {n: m for n, m in scored.items() if m["n_clusters"] >= 3}
            pool = granular if granular else scored
            best = max(pool, key=lambda n: pool[n]["silhouette_score"])
            logging.info(f"Best model: {best} (silhouette={scored[best]['silhouette_score']}, k={scored[best]['n_clusters']})")

            # Copy best model to the serving path.
            best_path = metadata["models"][best]["model_path"]
            shutil.copyfile(best_path, self.config.model_path)
            metadata["active_model"] = best
            metadata["models"][best]["copied_to"] = str(self.config.model_path)
            for n, m in model_metrics.items():
                metadata["models"][n]["metrics"] = {
                    "silhouette_score": m["silhouette_score"],
                    "davies_bouldin_index": m["davies_bouldin_index"],
                }
            metadata["metrics"] = model_metrics  # keep a flat copy too

            with open(self.config.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Local metrics file (best model only, for quick reference).
            with open(self.config.metric_file_name, "w") as f:
                json.dump({
                    "best_model": best,
                    "silhouette_score": scored[best]["silhouette_score"],
                    "davies_bouldin_index": scored[best]["davies_bouldin_index"],
                }, f, indent=2)

            # ---- MLflow logging for all models ----
            mlflow.set_registry_uri("sqlite:///mlflow.db")
            urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name="multi-model-comparison"):
                mlflow.log_param("best_model", best)
                for name, m in model_metrics.items():
                    prefix = f"{name}."
                    mlflow.log_metric(f"{prefix}silhouette_score", m["silhouette_score"] or float("nan"))
                    mlflow.log_metric(f"{prefix}davies_bouldin_index", m["davies_bouldin_index"] or float("nan"))
                    mlflow.log_param(f"{prefix}n_clusters", m["n_clusters"])
                mlflow.log_metric("best.silhouette_score", scored[best]["silhouette_score"])
                mlflow.log_metric("best.davies_bouldin_index", scored[best]["davies_bouldin_index"])

            logging.info("Successfully evaluated & logged all models to MLflow")

        except Exception as e:
            logging.error(f"Error in model evaluation: {e}")
            raise e
