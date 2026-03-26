import os
from dataclasses import dataclass
from datetime import datetime
 
from config.constants import (
    ARTIFACT_DIR,
    RAW_DATA_PATH,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
    TARGET_COLUMN,
    PREPROCESSING_FILE_NAME,
    MODEL_FILE_NAME,
    MODEL_BUCKET_NAME,
    S3_MODEL_KEY,
    EXPECTED_ACCURACY,
    MODEL_EVALUATION_THRESHOLD,
)
 
TIMESTAMP: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
 
 
@dataclass
class TrainingPipelineConfig:
    timestamp: str = TIMESTAMP
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
 
 
training_pipeline_config = TrainingPipelineConfig()
 
 
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_ingestion"
    )
    raw_data_path: str = RAW_DATA_PATH
    train_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_ingestion", "train.csv"
    )
    test_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_ingestion", "test.csv"
    )
    train_test_split_ratio: float = TRAIN_TEST_SPLIT_RATIO
    random_state: int = RANDOM_STATE
    target_column: str = TARGET_COLUMN


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_validation"
    )
    validation_report_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_validation", "report.json"
    )

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_transformation"
    )
    transformed_train_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_transformation", "train.npy"
    )
    transformed_test_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_transformation", "test.npy"
    )
    preprocessor_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_transformation", PREPROCESSING_FILE_NAME
    )

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, "model_trainer"
    )
    trained_model_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "model_trainer", MODEL_FILE_NAME
    )
    expected_accuracy: float = EXPECTED_ACCURACY
    random_state: int = RANDOM_STATE
    n_estimators: int = 100
    class_weight: str = "balanced"

