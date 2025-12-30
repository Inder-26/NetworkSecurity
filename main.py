from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer,ModelTrainerConfig

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        traningpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(traningpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed {dataingestionartifact}")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(traningpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info(f"Data validation completed {data_validation_artifact}")
        print(data_validation_artifact)
        data_transformation_config=DataTransformationConfig(traningpipelineconfig)
        data_transformation=DataTransformation(data_transformation_config,data_validation_artifact)
        logging.info("Initiate the data transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation completed {data_transformation_artifact}")
        print(data_transformation_artifact)

        logging.info("Model Trainer Started")
        model_trainer_config=ModelTrainerConfig(traningpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,
                                   data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logging.info(f"Model Trainer completed {model_trainer_artifact}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)