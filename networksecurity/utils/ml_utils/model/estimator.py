from networksecurity.exception.exception import NetworkSecurityException
import sys,os
from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR_NAME,MODEL_FILE_NAME
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor,model):
        """
        Initialize the NetworkModel with preprocessor and model.

        Args:
            preprocessor: The preprocessing object.
            model: The trained model.
        """
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self, X):
        """
        Make predictions using the preprocessor and model.

        Args:
            X: The input data for prediction.
        Returns:
            The predictions made by the model.
        """
        try:
            X_transform = self.preprocessor.transform(X)
            y_hat = self.model.predict(X_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)