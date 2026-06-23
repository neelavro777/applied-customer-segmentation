from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    data_path: Path
    ingested_data_path: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    preprocessor_path: Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    kmeans_model_path: Path
    dbscan_model_path: Path
    gmm_model_path: Path
    model_path: Path
    metadata_path: Path
    # KMeans
    n_clusters: int
    init: str
    random_state: int
    max_iters: int
    # DBSCAN
    eps: float
    min_samples: int
    # GMM
    n_components: int
    covariance_type: str

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    metadata_path: Path
    test_data_path: Path
    val_data_path: Path
    metric_file_name: Path
