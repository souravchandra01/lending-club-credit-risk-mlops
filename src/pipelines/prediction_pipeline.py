import io
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

from src.utils.logger import logger
from src.utils.exception import LendingClubException
from src.cloud.s3_handler import S3Handler
from config.constants import MODEL_BUCKET_NAME, S3_MODEL_KEY, S3_PREPROCESSOR_KEY


class PredictionPipeline:
    def __init__(self):
        self.s3 = S3Handler()
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self):
        logger.info("Loading model and preprocessor from S3")
        self.model = self.s3.load_model(
            bucket_name=MODEL_BUCKET_NAME,
            s3_key=S3_MODEL_KEY
        )
        self.preprocessor = self.s3.load_model(
            bucket_name=MODEL_BUCKET_NAME,
            s3_key=S3_PREPROCESSOR_KEY
        )
        if self.model is None or self.preprocessor is None:
            raise Exception("Model or preprocessor not found in S3. Run training pipeline first.")
        logger.info("Model and preprocessor loaded successfully")

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

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)
        df["income_to_loan"] = df["annual_inc"] / (df["loan_amnt"] + 1)
        df["total_interest"] = (df["int_rate"] / 100) * df["loan_amnt"]
        df["loan_to_installment"] = df["loan_amnt"] / (df["installment"] + 1)
        df["revol_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        le = LabelEncoder()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["term"], errors="ignore")

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._clean_emp_length(df)
        df = self._engineer_features(df)
        df = self._drop_columns(df)
        df = self._encode_categoricals(df)
        return self.preprocessor.transform(df)

    def predict(self, df: pd.DataFrame) -> dict:
        try:
            logger.info("Running prediction")
            transformed = self._transform(df)
            prediction = self.model.predict(transformed)[0]
            probability = self.model.predict_proba(transformed)[0][1]

            THRESHOLD = 0.3

            prediction = 1 if probability >= THRESHOLD else 0

            result = {
                "prediction": "Default" if prediction == 1 else "No Default",
                "default_probability": round(float(probability), 4)
            }
            logger.info(f"Prediction: {result['prediction']} | Probability: {result['default_probability']}")
            return result

        except Exception as e:
            raise LendingClubException(e)