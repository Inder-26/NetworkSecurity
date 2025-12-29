import os,sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classfication_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def train_model(self,X_train,X_test,y_train,y_test):
        model = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }
        params = {
            "Decision Tree": {
                'criterion':['gini','entropy','log_loss'],
                #'splitter':['best','random'],
                #'max_features':['sqrt','log2']
            },
            "Random Forest": {
                #'criterion':['gini','entropy','log_loss'],
                #'max_features':['sqrt','log2'],
                'n_estimators':[8,16,32,64,128,256]
            },
            "Gradient Boosting": {
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                'n_estimators':[8,16,32,64,128,256]
            },
            "AdaBoost": {
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators':[8,16,32,64,128,256]
            },
            "Logistic Regression": {},
        }

        model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,
                                             X_test=X_test,y_test=y_test,models=model,params=params)
        
        ## To get the best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get the best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)]
        best_model = model[best_model_name]
        logging.info(f"Best model found , Model Name : {best_model_name} , R2 Score : {best_model_score}")

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        classification_train_metric=get_classification_score(y_true=y_train, y_pred=y_train_pred)
        classification_test_metric=get_classification_score(y_true=y_test, y_pred=y_test_pred)
        

        ## Track with mlflow


        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_model)
        logging.info(f"Trained model saved at : {self.model_trainer_config.trained_model_file_path}")
    
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                            train_metric_artifact=classification_train_metric,
                            test_metric_artifact=classification_test_metric)
        logging.info(f"Model Trainer Artifact : {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training array and transformed test array")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## Load numpy array
            train_array = load_numpy_array_data(file_path=train_file_path)
            test_array = load_numpy_array_data(file_path=test_file_path)
            logging.info("Splitting training and test input and target feature")
            X_train,y_train = train_array[:,:-1],train_array[:,-1]
            X_test,y_test = test_array[:,:-1],test_array[:,-1]

            model_trainer_artifact = self.train_model(X_train=X_train, X_test=X_test,
                                                      y_train=y_train, y_test=y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)