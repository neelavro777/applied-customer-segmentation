import os
import json
import numpy as np
import logging
import joblib
import torch
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from src.entity.config_entity import ModelTrainerConfig
from src.components.segment_naming import assign_segments


class PyTorchKMeans:
    def __init__(self, n_clusters, max_iters=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        # cluster centroids in scaled feature space (numpy), set after fit
        self.centroids_scaled = None

    def fit(self, X: np.ndarray):
        torch.manual_seed(self.random_state)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device} for PyTorch K-Means")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        indices = torch.randperm(X_tensor.shape[0])[:self.n_clusters]
        self.centroids = X_tensor[indices].clone()

        for i in range(self.max_iters):
            distances = torch.cdist(X_tensor, self.centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([
                X_tensor[labels == k].mean(dim=0) if (labels == k).sum() > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            if torch.allclose(self.centroids, new_centroids, rtol=1e-4):
                logging.info(f"K-Means converged at iteration {i+1}")
                break
            self.centroids = new_centroids

        self.centroids_scaled = self.centroids.detach().cpu().numpy()
        self.labels_ = labels.cpu().numpy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise Exception("Model is not fitted yet.")
        device = self.centroids.device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        distances = torch.cdist(X_tensor, self.centroids)
        labels = torch.argmin(distances, dim=1)
        return labels.cpu().numpy()


def _centroids_rfm_from_scaled(X_scaled, labels, scaler, ignore_noise=False):
    """Mean RFM (original scale) per cluster."""
    unique = sorted(int(l) for l in np.unique(labels))
    if ignore_noise:
        unique = [l for l in unique if l != -1]
    centroids = {}
    for l in unique:
        m = labels == l
        if m.sum() == 0:
            centroids[l] = [0.0, 0.0, 0.0]
            continue
        mean_scaled = X_scaled[m].mean(axis=0)
        # Inverse transform expects 3 features; scaler was fit on 3 RFM features.
        mean_log = scaler.inverse_transform(mean_scaled.reshape(1, -1))[0]
        centroids[l] = np.expm1(mean_log).tolist()
    return centroids


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self):
        try:
            import joblib as _joblib
            scaler = _joblib.load(self.config.root_dir.parent / "data_transformation" / "preprocessor.pkl")

            logging.info("Starting Model Training (K-Means + DBSCAN + GMM)")
            transformed = np.load(self.config.train_data_path)
            X = transformed[:, 1:]  # drop CustomerID

            results = {}

            # ---- K-Means (custom PyTorch) ----
            logging.info("Training PyTorch K-Means ...")
            kmeans = PyTorchKMeans(
                n_clusters=self.config.n_clusters,
                max_iters=self.config.max_iters,
                random_state=self.config.random_state,
            )
            kmeans.fit(X)
            km_labels = kmeans.labels_
            joblib.dump(kmeans, self.config.kmeans_model_path)
            km_centroids_rfm = _centroids_rfm_from_scaled(X, km_labels, scaler)
            results["kmeans"] = {
                "model_path": str(self.config.kmeans_model_path),
                "labels": km_labels,
                "centroids_scaled": kmeans.centroids_scaled.tolist(),
                "centroids_rfm": km_centroids_rfm,
                "n_clusters": int(len(np.unique(km_labels))),
            }
            logging.info(f"K-Means saved -> {self.config.kmeans_model_path}")

            # ---- DBSCAN ----
            logging.info("Training DBSCAN ...")
            db = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples, n_jobs=-1)
            db.fit(X)
            db_labels = db.labels_
            joblib.dump(db, self.config.dbscan_model_path)
            db_centroids_rfm = _centroids_rfm_from_scaled(X, db_labels, scaler, ignore_noise=True)
            # noise centroid placeholder
            db_centroids_rfm[-1] = [0.0, 0.0, 0.0]
            # scaled centroids: mean of each real cluster; noise -> far point
            db_centroids_scaled = {}
            for l in sorted(int(x) for x in np.unique(db_labels)):
                if l == -1:
                    db_centroids_scaled[l] = [1e9, 1e9, 1e9]
                else:
                    db_centroids_scaled[l] = X[db_labels == l].mean(axis=0).tolist()
            results["dbscan"] = {
                "model_path": str(self.config.dbscan_model_path),
                "labels": db_labels,
                "centroids_scaled": {str(k): v for k, v in db_centroids_scaled.items()},
                "centroids_rfm": db_centroids_rfm,
                "n_clusters": int(len([l for l in np.unique(db_labels) if l != -1])),
                "n_noise": int((db_labels == -1).sum()),
                "eps": float(self.config.eps),
                "min_samples": int(self.config.min_samples),
            }
            logging.info(f"DBSCAN saved -> {self.config.dbscan_model_path} (clusters={results['dbscan']['n_clusters']}, noise={results['dbscan']['n_noise']})")

            # ---- GMM ----
            logging.info("Training Gaussian Mixture Model ...")
            gmm = GaussianMixture(
                n_components=self.config.n_components,
                covariance_type=self.config.covariance_type,
                random_state=self.config.random_state,
            )
            gmm.fit(X)
            gmm_labels = gmm.predict(X)
            joblib.dump(gmm, self.config.gmm_model_path)
            gmm_centroids_scaled = gmm.means_
            gmm_centroids_rfm = {}
            for l in range(gmm.n_components):
                mean_log = scaler.inverse_transform(gmm_centroids_scaled[l].reshape(1, -1))[0]
                gmm_centroids_rfm[l] = np.expm1(mean_log).tolist()
            results["gmm"] = {
                "model_path": str(self.config.gmm_model_path),
                "labels": gmm_labels,
                "centroids_scaled": gmm_centroids_scaled.tolist(),
                "centroids_rfm": gmm_centroids_rfm,
                "n_clusters": int(gmm.n_components),
            }
            logging.info(f"GMM saved -> {self.config.gmm_model_path}")

            # ---- Segment naming + metadata ----
            metadata = {"models": {}}
            for name, res in results.items():
                segments = assign_segments(np.asarray(res["labels"]), res["centroids_rfm"])
                # Distance threshold for DBSCAN noise detection at inference.
                max_intra = None
                if name == "dbscan":
                    dists = []
                    for l in sorted(int(x) for x in np.unique(res["labels"])):
                        if l == -1:
                            continue
                        m = res["labels"] == l
                        if m.sum() == 0:
                            continue
                        c = np.asarray(res["centroids_scaled"][str(l)])
                        d = np.linalg.norm(X[m] - c, axis=1)
                        dists.append(float(d.max()) if len(d) else 0.0)
                    max_intra = max(dists) if dists else float(self.config.eps)
                metadata["models"][name] = {
                    "model_path": res["model_path"],
                    "n_clusters": res["n_clusters"],
                    "centroids_scaled": res["centroids_scaled"] if name != "dbscan" else res["centroids_scaled"],
                    "segments": segments,
                    **({"n_noise": res["n_noise"], "noise_threshold": max_intra} if name == "dbscan" else {}),
                }

            # Default active model = kmeans until evaluation overrides it.
            metadata["active_model"] = "kmeans"
            metadata["models"]["kmeans"]["copied_to"] = str(self.config.model_path)
            # Copy kmeans to the serving path initially.
            joblib.dump(kmeans, self.config.model_path)

            with open(self.config.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Model metadata written -> {self.config.metadata_path}")

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
