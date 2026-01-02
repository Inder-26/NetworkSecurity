import os
import sys
import mlflow
import dagshub
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)

from networksecurity.utils.ml_utils.metric.classfication_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

import os

# ---------------- Dagshub + MLflow ----------------
if os.getenv("MLFLOW_TRACKING_URI"):
    print("info: MLflow tracking URI is already set, skipping DagsHub init")
elif os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
    dagshub.init(repo_owner="Inder-26", repo_name="NetworkSecurity", mlflow=True)
else:
    print("Warning: DagsHub credentials not found. Tracking might rely on local configs or fail.")



# ---------------- Helper: log visual artifacts ----------------
def log_classification_artifacts(y_true, y_pred, y_proba):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")
    plt.close()


# ---------------- Model Trainer ----------------
class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, X_test, y_train, y_test):

        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
        }

        # ---------- MLflow logging ----------
        best_f1 = -1
        best_model = None
        best_model_name = None
        best_run_id = None


        for model_name, model in models.items():

            with mlflow.start_run(run_name=model_name) as run:

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                y_test_proba = model.predict_proba(X_test)[:, 1]

                train_metric = get_classification_score(y_train, y_train_pred)
                test_metric = get_classification_score(y_test, y_test_pred)

                # Params & tags
                mlflow.log_params(model.get_params())
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("stage", "experiment")

                # Metrics (decision metric = test_f1)
                mlflow.log_metric("train_f1", train_metric.f1_score)
                mlflow.log_metric("test_f1", test_metric.f1_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)

                # Visual evaluation (artifacts)
                log_classification_artifacts(
                    y_true=y_test,
                    y_pred=y_test_pred,
                    y_proba=y_test_proba,
                )

                if test_metric.f1_score > best_f1:
                    best_f1 = test_metric.f1_score
                    best_model = model
                    best_model_name = model_name
                    best_run_id = run.info.run_id


        logging.info(
            f"Best Model: {best_model_name} | "
            f"Test F1: {best_f1}"
        )


        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("best_model", "true")

        # ---------- Save final model for deployment ----------
        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )

        final_model_dir = os.path.join(os.getcwd(), "final_model")
        os.makedirs(final_model_dir, exist_ok=True)

        save_object(
            os.path.join(final_model_dir, "model.pkl"),
            best_model,
        )
        save_object(
            os.path.join(final_model_dir, "preprocessor.pkl"),
            preprocessor,
        )

        logging.info(f"Final model and preprocessor saved in final_model")
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        best_train_metric = get_classification_score(y_train, y_train_pred)
        best_test_metric = get_classification_score(y_test, y_test_pred)

        return ModelTrainerArtifact(
            trained_model_file_path=os.path.join(final_model_dir, "model.pkl"),
            train_metric_artifact=best_train_metric,
            test_metric_artifact=best_test_metric,
        )


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_array = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_array = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            return self.train_model(X_train, X_test, y_train, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
