from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import threading

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.logger import logger
from src.utils.exception import LendingClubException
from config.constants import APP_HOST, APP_PORT


app = FastAPI(
    title="Lending Club Credit Risk API",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class LoanInput(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    open_acc: int
    pub_rec: int
    revol_bal: float
    revol_util: float
    total_acc: int
    pub_rec_bankruptcies: int
    emp_length: str
    term: str
    grade: str
    sub_grade: str
    home_ownership: str
    verification_status: str
    purpose: str
    initial_list_status: str
    application_type: str


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(data: LoanInput):
    try:
        df = pd.DataFrame([data.model_dump()])
        pipeline = PredictionPipeline()
        result = pipeline.predict(df)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)