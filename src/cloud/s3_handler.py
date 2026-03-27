import io
import joblib
import boto3
from botocore.exceptions import ClientError

from src.utils.logger import logger
from src.utils.exception import LendingClubException
from config.constants import AWS_REGION


class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client("s3", region_name=AWS_REGION)

    def upload_model(self, file_path: str, bucket_name: str, s3_key: str) -> None:
        try:
            logger.info(f"Uploading model to s3://{bucket_name}/{s3_key}")
            self.s3_client.upload_file(file_path, bucket_name, s3_key)
            logger.info("Model uploaded successfully")
        except ClientError as e:
            raise LendingClubException(e)

    def load_model(self, bucket_name: str, s3_key: str):
        try:
            logger.info(f"Loading model from s3://{bucket_name}/{s3_key}")
            response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            model = joblib.load(io.BytesIO(response["Body"].read()))
            logger.info("Model loaded from S3 successfully")
            return model
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.info("No model found in S3")
                return None
            raise LendingClubException(e)