import json
import os
import pandas as pd
import yaml

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.utils.logger import logger
from src.utils.exception import LendingClubException
from config.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

        with open(SCHEMA_FILE_PATH, "r") as f:
            self.schema = yaml.safe_load(f)

    def _check_schema(self, df: pd.DataFrame) -> tuple[bool, str]:
        required = (
            self.schema["required_columns"]["numerical"]
            + self.schema["required_columns"]["categorical"]
        )
        missing = [col for col in required if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"
        return True, "Schema check passed"

    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("Starting data validation")
        try:
            df = pd.read_csv(self.ingestion_artifact.train_file_path)

            status, message = self._check_schema(df)

            report = {
                "schema_check": {"status": status, "message": message},
                "overall_status": status
            }

            # Save report
            os.makedirs(self.config.data_validation_dir, exist_ok=True)
            with open(self.config.validation_report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info(f"Validation report saved at: {self.config.validation_report_path}")

            if not status:
                raise Exception(f"Data validation failed. Check report at {self.config.validation_report_path}")

            logger.info("Data validation passed")

            return DataValidationArtifact(
                validation_status=status,
                message=message,
                validation_report_path=self.config.validation_report_path
            )

        except Exception as e:
            raise LendingClubException(e)