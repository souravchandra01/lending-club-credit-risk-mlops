import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.utils.logger import logger
from src.utils.exception import LendingClubException
from config.constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.transformation_artifact = transformation_artifact

    def _evaluate(self, y_true, y_pred, y_prob) -> ClassificationMetricArtifact:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
            roc_auc_score=roc_auc_score(y_true, y_prob)
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Starting model training")
        try:
            train_arr = np.load(self.transformation_artifact.transformed_train_path, allow_pickle=True)
            test_arr = np.load(self.transformation_artifact.transformed_test_path, allow_pickle=True)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)

            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            model = LGBMClassifier(
                n_estimators=self.config.n_estimators,
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                verbose=-1
            )

            # MLflow tracking
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run():
                logger.info("Model is being trained...")
                model.fit(X_train, y_train)
                logger.info("Model training done")

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                metrics = self._evaluate(y_test, y_pred, y_prob)

        
                mlflow.log_params({
                    "model": "LGBMClassifier",
                    "n_estimators": self.config.n_estimators,
                    "class_weight": self.config.class_weight,
                    "random_state": self.config.random_state
                })
                mlflow.log_metrics({
                    "f1_score": metrics.f1_score,
                    "precision": metrics.precision_score,
                    "recall": metrics.recall_score,
                    "roc_auc": metrics.roc_auc_score
                })
                mlflow.sklearn.log_model(model, name="model")

                logger.info(f"ROC AUC: {metrics.roc_auc_score:.4f} | F1: {metrics.f1_score:.4f}")

            
            if metrics.roc_auc_score < self.config.expected_accuracy:
                raise Exception(
                    f"Model AUC {metrics.roc_auc_score:.4f} below threshold {self.config.expected_accuracy}"
                )

            # Save model
            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            joblib.dump(model, self.config.trained_model_path)
            logger.info(f"Model saved at: {self.config.trained_model_path}")

            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                metric_artifact=metrics
            )

        except Exception as e:
            raise LendingClubException(e)