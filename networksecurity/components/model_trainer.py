import os
import sys
import mlflow
import dagshub
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

# ---------------- Dagshub + MLflow ----------------
dagshub.init(
    repo_owner="Inder-26",
    repo_name="NetworkSecurity",
    mlflow=True,
)


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

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"]
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression": {},
        }

        # ---------- Hyperparameter search ----------
        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            params=params,
        )

        # ---------- MLflow logging ----------
        model_scores = {}
        run_id_map = {}

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

                model_scores[model_name] = test_metric.f1_score
                run_id_map[model_name] = run.info.run_id

        # ---------- Best model selection ----------
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = model_report[best_model_name]["model"]

        logging.info(
            f"Best Model: {best_model_name} | "
            f"Test F1: {model_scores[best_model_name]}"
        )

        # ---------- Tag best model ----------
        mlflow.start_run(run_id=run_id_map[best_model_name])
        mlflow.set_tag("best_model", "true")
        mlflow.end_run()

        # ---------- Save final model for deployment ----------
        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )

        final_model_dir = os.path.join(os.getcwd(), "final_models")
        os.makedirs(final_model_dir, exist_ok=True)

        save_object(
            os.path.join(final_model_dir, "model.pkl"),
            best_model,
        )
        save_object(
            os.path.join(final_model_dir, "preprocessor.pkl"),
            preprocessor,
        )

        logging.info("Final model and preprocessor saved in final_model/")

        return ModelTrainerArtifact(
            trained_model_file_path=os.path.join(
                final_model_dir, "model.pkl"
            ),
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
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
