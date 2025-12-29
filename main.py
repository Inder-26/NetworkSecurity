from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig
from networksecurity.entity.config_entity import TraningPipelineConfig
import sys

if __name__ == "__main__":
    try:
        traningpipelineconfig=TraningPipelineConfig()
        dataingestionconfig=DataIngestionConfig(traningpipelineconfig)
        dataingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(traningpipelineconfig)
        datavalidation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact=datavalidation.initiate_data_validation()
        logging.info(f"Data validation completed {data_validation_artifact}")
        print(data_validation_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)