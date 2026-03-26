import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Root
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Artifacts
ARTIFACT_DIR: str = str(ROOT_DIR / "artifacts")

# Data
RAW_DATA_PATH: str = str(ROOT_DIR / "data" / "raw" / "lending_club_loan_two.csv")
TARGET_COLUMN: str = "default"
TRAIN_TEST_SPLIT_RATIO: float = 0.2
RANDOM_STATE: int = 42

# Schema
SCHEMA_FILE_PATH: str = str(ROOT_DIR / "config" / "schema.yaml")

# Preprocessing
PREPROCESSING_FILE_NAME: str = "preprocessor.pkl"

# Model
MODEL_FILE_NAME: str = "model.pkl"
EXPECTED_ACCURACY: float = 0.60
MODEL_EVALUATION_THRESHOLD: float = 0.02

# AWS
AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL_BUCKET_NAME: str = os.getenv("MODEL_BUCKET_NAME", "lending-club-mlops-models")
S3_MODEL_KEY: str = "model-registry/model.pkl"

# MLflow
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = "lending-club-credit-risk"

# App
APP_HOST: str = "0.0.0.0"
APP_PORT: int = 8000