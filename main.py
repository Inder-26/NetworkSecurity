from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
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
 
    except Exception as e:
        raise NetworkSecurityException(e, sys)