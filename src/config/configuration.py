from src.constants import *
from src.utils.common import read_yaml, create_directories, resolve_path
from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([resolve_path(self.config['artifacts_root'])])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']

        create_directories([resolve_path(config['root_dir'])])

        data_ingestion_config = DataIngestionConfig(
            root_dir=resolve_path(config['root_dir']),
            data_path=resolve_path(config['data_path']),
            ingested_data_path=resolve_path(config['ingested_data_path'])
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        create_directories([resolve_path(config['root_dir'])])

        return DataValidationConfig(
            root_dir=resolve_path(config['root_dir']),
            STATUS_FILE=str(resolve_path(config['STATUS_FILE']))
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        create_directories([resolve_path(config['root_dir'])])

        return DataTransformationConfig(
            root_dir=resolve_path(config['root_dir']),
            data_path=resolve_path(config['data_path']),
            preprocessor_path=resolve_path(config['preprocessor_path']),
            train_data_path=resolve_path(config['train_data_path']),
            test_data_path=resolve_path(config['test_data_path']),
            val_data_path=resolve_path(config['val_data_path'])
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config['model_trainer']
        km = self.params['KMeans']
        db = self.params['DBSCAN']
        gmm = self.params['GMM']

        create_directories([resolve_path(config['root_dir'])])

        return ModelTrainerConfig(
            root_dir=resolve_path(config['root_dir']),
            train_data_path=resolve_path(config['train_data_path']),
            kmeans_model_path=resolve_path(config['kmeans_model_path']),
            dbscan_model_path=resolve_path(config['dbscan_model_path']),
            gmm_model_path=resolve_path(config['gmm_model_path']),
            model_path=resolve_path(config['model_path']),
            metadata_path=resolve_path(config['metadata_path']),
            n_clusters=km['n_clusters'],
            init=km['init'],
            random_state=km['random_state'],
            max_iters=km['max_iters'],
            eps=db['eps'],
            min_samples=db['min_samples'],
            n_components=gmm['n_components'],
            covariance_type=gmm['covariance_type'],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config['model_evaluation']
        create_directories([resolve_path(config['root_dir'])])

        return ModelEvaluationConfig(
            root_dir=resolve_path(config['root_dir']),
            model_path=resolve_path(config['model_path']),
            metadata_path=resolve_path(config['metadata_path']),
            test_data_path=resolve_path(config['test_data_path']),
            val_data_path=resolve_path(config['val_data_path']),
            metric_file_name=resolve_path(config['metric_file_name'])
        )
