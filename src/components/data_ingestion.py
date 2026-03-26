import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.logger import logger
from src.utils.exception import LendingClubException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting data ingestion")
        try:
            # Load raw data
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Loaded dataset: {df.shape}")

            # Create target column from loan_status
            df["default"] = df["loan_status"].apply(
                lambda x: 1 if x == "Charged Off" else 0
            )
            logger.info(f"Default rate: {df['default'].mean():.2%}")

            # Train test split
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=self.config.random_state,
                stratify=df[self.config.target_column]
            )
            logger.info(f"Train: {train_df.shape} | Test: {test_df.shape}")

            # Save splits
            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)
            train_df.to_csv(self.config.train_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)
            logger.info("Train and test files saved")

            return DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path
            )

        except Exception as e:
            raise LendingClubException(e)