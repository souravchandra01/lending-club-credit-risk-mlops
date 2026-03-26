import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from src.utils.logger import logger
from src.utils.exception import LendingClubException


NUMERICAL_COLUMNS = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "pub_rec_bankruptcies", "emp_length", "term_months",
    "income_to_loan", "total_interest", "loan_to_installment", "revol_to_income"
]

CATEGORICAL_COLUMNS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "initial_list_status", "application_type"
]

TARGET_COLUMN = "default"


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _clean_emp_length(self, df: pd.DataFrame) -> pd.DataFrame:
        df["emp_length"] = (
            df["emp_length"]
            .astype(str)
            .str.replace("+", "", regex=False)
            .str.replace("< 1", "0", regex=False)
            .str.replace(" years", "", regex=False)
            .str.replace(" year", "", regex=False)
        )
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce").fillna(0)
        return df

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0)
        df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)
        df["income_to_loan"] = df["annual_inc"] / (df["loan_amnt"] + 1)
        df["total_interest"] = (df["int_rate"] / 100) * df["loan_amnt"]
        df["loan_to_installment"] = df["loan_amnt"] / (df["installment"] + 1)
        df["revol_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["loan_status", "term"], errors="ignore")

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        le = LabelEncoder()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    def _get_preprocessor(self) -> ColumnTransformer:
        numerical_pipeline = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLUMNS)
        ], remainder="passthrough")
        return preprocessor

    def _transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean_emp_length(df)
        df = self._fill_nulls(df)
        df = self._engineer_features(df)
        df = self._drop_columns(df)
        df = self._encode_categoricals(df)
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info("Starting data transformation")
        try:
            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            train_df = self._transform_dataframe(train_df)
            test_df = self._transform_dataframe(test_df)


            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]


            preprocessor = self._get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save processed data
            os.makedirs(self.config.data_transformation_dir, exist_ok=True)
            np.save(self.config.transformed_train_path, train_arr)
            np.save(self.config.transformed_test_path, test_arr)
            joblib.dump(preprocessor, self.config.preprocessor_path)

            logger.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessor_path=self.config.preprocessor_path
            )

        except Exception as e:
            raise LendingClubException(e)