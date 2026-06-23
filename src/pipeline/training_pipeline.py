import logging
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()

        # Data Ingestion
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.initiate_data_ingestion()

        # Data Validation
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        is_valid = data_validation.validate_all_columns()

        if not is_valid:
            raise Exception("Data validation failed! Check the raw data schema.")

        # Data Transformation
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        data_transformation.initiate_data_transformation()

        # Model Trainer
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.initiate_model_trainer()

        # Model Evaluation
        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(model_evaluation_config)
        model_evaluation.log_into_mlflow()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
    try:
        logging.info(">>>>>> Training Pipeline Started <<<<<<")
        pipeline = TrainingPipeline()
        pipeline.main()
        logging.info(">>>>>> Training Pipeline Completed Successfully <<<<<<\n\x1b[32m")
    except Exception as e:
        logging.exception(e)
        raise e
