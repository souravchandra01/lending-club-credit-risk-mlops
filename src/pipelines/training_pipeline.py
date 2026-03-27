from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import (DataIngestionConfig, 
                                      DataValidationConfig, 
                                      DataTransformationConfig, 
                                      ModelTrainerConfig, 
                                      ModelEvaluationConfig, 
                                      ModelPusherConfig
                                      )
from src.utils.logger import logger
from src.utils.exception import LendingClubException

from dotenv import load_dotenv
load_dotenv()


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self):
        
        data_ingestion = DataIngestion(config=self.data_ingestion_config)
        return data_ingestion.initiate_data_ingestion()

    def start_data_validation(self, ingestion_artifact):
        
        data_validation = DataValidation(
            config=self.data_validation_config,
            ingestion_artifact=ingestion_artifact
        )
        return data_validation.initiate_data_validation()
    
    def start_data_tranformation(self, ingestion_artifact):

        data_transformation = DataTransformation(
            config = self.data_transformation_config,
            ingestion_artifact=ingestion_artifact
        )
        return data_transformation.initiate_data_transformation()

    def start_model_trainer(self, transformation_artifact):
        model_trainer = ModelTrainer(
            config=self.model_trainer_config,
            transformation_artifact=transformation_artifact
        )
        return model_trainer.initiate_model_trainer()
    
    def start_model_evaluation(self, trainer_artifact, transformation_artifact):
        model_evaluation = ModelEvaluation(
        config=self.model_evaluation_config,
        trainer_artifact=trainer_artifact,
        transformation_artifact=transformation_artifact
        )
        return model_evaluation.initiate_model_evaluation()

    def start_model_pusher(self, evaluation_artifact):
        model_pusher = ModelPusher(
        config=self.model_pusher_config,
        evaluation_artifact=evaluation_artifact
        )
        return model_pusher.initiate_model_pusher()

    def run_pipeline(self):
        try:
            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)
            logger.info(f"Validation status: {validation_artifact.validation_status}")
            transformation_artifact = self.start_data_tranformation(ingestion_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            logger.info(f"ROC AUC: {trainer_artifact.metric_artifact.roc_auc_score:.4f}")
            evaluation_artifact = self.start_model_evaluation(trainer_artifact, transformation_artifact)
            self.start_model_pusher(evaluation_artifact)
            logger.info(f"Model accepted: {evaluation_artifact.is_model_accepted}")
            logger.info("Training Pipeline completed successfully")
        except Exception as e:
            raise LendingClubException(e)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()