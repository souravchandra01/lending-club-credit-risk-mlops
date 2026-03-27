from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.utils.logger import logger
from src.utils.exception import LendingClubException
from src.cloud.s3_handler import S3Handler


class ModelPusher:
    def __init__(self, config: ModelPusherConfig, evaluation_artifact: ModelEvaluationArtifact):
        self.config = config
        self.evaluation_artifact = evaluation_artifact
        self.s3 = S3Handler()

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logger.info("Starting model pusher")
        try:
            if not self.evaluation_artifact.is_model_accepted:
                logger.info("Model not accepted — skipping push to S3")
            else:
                self.s3.upload_model(
                    file_path=self.evaluation_artifact.trained_model_path,
                    bucket_name=self.config.bucket_name,
                    s3_key=self.config.s3_model_key
                )
                logger.info(f"Model pushed to S3: s3://{self.config.bucket_name}/{self.config.s3_model_key}")

            self.s3.upload_model(
                file_path=self.evaluation_artifact.preprocessor_path,
                bucket_name=self.config.bucket_name,
                s3_key=self.config.s3_preprocessor_key
            )

            logger.info(f"Preprocessor pushed to S3: s3://{self.config.bucket_name}/{self.config.s3_preprocessor_key}")

            return ModelPusherArtifact(
                bucket_name=self.config.bucket_name,
                s3_model_path=self.config.s3_model_key,
                s3_preprocessor_path=self.config.s3_preprocessor_key
            )

        except Exception as e:
            raise LendingClubException(e)