import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (ModelTrainerArtifact, ModelEvaluationArtifact, DataTransformationArtifact)
from src.utils.logger import logger
from src.utils.exception import LendingClubException
from src.cloud.s3_handler import S3Handler


class ModelEvaluation:
    def __init__(
        self,
        config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact
    ):
        self.config = config
        self.trainer_artifact = trainer_artifact
        self.transformation_artifact = transformation_artifact
        self.s3 = S3Handler()

    def _get_test_data(self):
        test_arr = np.load(
            self.transformation_artifact.transformed_test_path,
            allow_pickle=True
        )
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)
        return X_test, y_test

    def _evaluate_model(self, model, X_test, y_test) -> float:
        y_prob = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_prob)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logger.info("Starting model evaluation")
        try:
            X_test, y_test = self._get_test_data()

            # Load newly trained model
            new_model = joblib.load(self.trainer_artifact.trained_model_path)
            new_model_auc = self._evaluate_model(new_model, X_test, y_test)
            logger.info(f"New model AUC: {new_model_auc:.4f}")

            # Check if a model already exists in S3
            production_model = self.s3.load_model(
                bucket_name=self.config.bucket_name,
                s3_key=self.config.s3_model_key
            )

            # If no production model exists — accept new model directly
            if production_model is None:
                logger.info("No production model found in S3 — accepting new model")
                return ModelEvaluationArtifact(
                    is_model_accepted=True,
                    changed_accuracy=new_model_auc,
                    s3_model_path=self.config.s3_model_key,
                    trained_model_path=self.trainer_artifact.trained_model_path,
                    preprocessor_path=self.transformation_artifact.preprocessor_path
                )

            # If a model exists — compare
            prod_model_auc = self._evaluate_model(production_model, X_test, y_test)
            logger.info(f"Production model AUC: {prod_model_auc:.4f}")

            improvement = new_model_auc - prod_model_auc
            logger.info(f"Improvement: {improvement:.4f} (threshold: {self.config.changed_threshold})")

            is_accepted = improvement >= self.config.changed_threshold

            if is_accepted:
                logger.info("New model accepted — better than production")
            else:
                logger.info("New model rejected — not better enough than production")

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                changed_accuracy=improvement,
                s3_model_path=self.config.s3_model_key,
                trained_model_path=self.trainer_artifact.trained_model_path,
                preprocessor_path=self.transformation_artifact.preprocessor_path
            )

        except Exception as e:
            raise LendingClubException(e)