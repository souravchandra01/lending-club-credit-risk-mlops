# 🏦 Lending Club Credit Risk MLOps

> A production-grade MLOps pipeline for predicting loan default risk — covering the complete ML lifecycle from data ingestion to cloud deployment with CI/CD automation.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.2-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.10.1-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-yellow.svg)](https://aws.amazon.com/)

---

## ✨ Key Features

- **🔄 End-to-End ML Pipeline** - Ingestion → Validation → Transformation → Training → Evaluation → Deployment
- **📊 Experiment Tracking** - MLflow tracks every training run with metrics, params and model artifacts
- **🔁 Model Versioning** - Best model stored in AWS S3 with threshold-based promotion
- **🚀 REST API** - FastAPI serves predictions with a clean dark-themed frontend
- **🐳 Containerized** - Docker image pushed to AWS ECR
- **⚙️ CI/CD** - GitHub Actions auto-deploys on every push to main
- **☁️ AWS Deployed** - EC2 + ECR with self-hosted GitHub runner

---

## 🛠️ Tech Stack

### ML & Backend
- **LightGBM** - Gradient boosting classifier for credit risk
- **Scikit-learn** - Preprocessing pipeline (StandardScaler)
- **MLflow** - Experiment tracking and model logging
- **FastAPI** - REST API for predictions
- **Pydantic** - Request validation

### Infrastructure
- **Docker** - Containerization
- **AWS S3** - Model and preprocessor storage
- **AWS ECR** - Container registry
- **AWS EC2** - t3.small deployment instance
- **GitHub Actions** - CI/CD with self-hosted runner

### Frontend
- **Vanilla JS** - No framework overhead
- **HTML5/CSS3** - Dark themed responsive UI

---

## 🏗️ Architecture

```
Raw Data (Lending Club CSV)
          ↓
   Data Ingestion
   (train/test split)
          ↓
   Data Validation
   (schema checks)
          ↓
  Data Transformation
  (feature engineering,
   encoding, scaling)
          ↓
   Model Trainer ──────→ MLflow
   (LightGBM)           (metrics, params,
          ↓              model artifacts)
   Model Evaluation
   (compare vs S3 model)
          ↓
   Model Pusher
   (push to AWS S3)
          ↓
   FastAPI + Frontend
   (load from S3 → predict)
          ↓
   Docker → ECR → EC2
   (GitHub Actions CI/CD)
```

---

## 📁 Project Structure

```
lending-club-credit-risk-mlops/
│
├── .github/workflows/
│   └── ci-cd.yml               # CI/CD pipeline
│
├── config/
│   ├── constants.py            # all paths and constants
│   └── schema.yaml             # data validation schema
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   │
│   ├── cloud/
│   │   └── s3_handler.py
│   │
│   └── utils/
│       ├── logger.py
│       └── exception.py
│
├── static/                     # frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── notebooks/
│   └── Experiment_Notebook.ipynb
│
├── app.py                      # FastAPI application
├── Dockerfile
├── requirements.txt
├── setup.py
└── template.py                 # auto-generates project structure
```

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone repository
git clone https://github.com/souravchandra01/lending-club-credit-risk-mlops.git
cd lending-club-credit-risk-mlops

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate        # Windows
source myenv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Add environment variables
cp .env.example .env
# Fill in your AWS credentials in .env
```

### Run Training Pipeline

```bash
# Terminal 1 — Start MLflow UI
mlflow ui

# Terminal 2 — Run pipeline
python src/pipelines/training_pipeline.py
```

Open MLflow at `http://localhost:5000` to track experiments.

### Run FastAPI App Locally

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000`

---

## ☁️ AWS Deployment

**Infrastructure:**
- EC2 t3.small (Ubuntu 24.04)
- ECR repository
- S3 bucket for model storage
- GitHub Actions self-hosted runner on EC2

**Deploy:**
```bash
git push origin main
# GitHub Actions automatically builds and deploys!
```

**CI/CD Steps:**
1. Build Docker image on EC2
2. Push to AWS ECR
3. Stop old container
4. Pull latest image from ECR
5. Run new container with AWS credentials
6. Verify health check at `/health`

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC AUC | 0.72 |
| Recall | 0.67 |
| Precision | 0.32 |
| F1 Score | 0.43 |

> Recall is prioritized over precision — in credit risk, missing a default is more costly than a false alarm. The model catches 67% of actual defaults.

**Dataset:** [Lending Club Loan Data](https://www.kaggle.com/datasets/epsilon22/lending-club-loan-two?select=lending_club_loan_two.csv) — 396,030 records, 27 features.

---

## 🔧 Environment Variables

Create a `.env` file in the project root:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1
MODEL_BUCKET_NAME=lending-club-mlops-models
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 💰 Cost Considerations

| Service | Cost/Month |
|---|---|
| EC2 t3.small | ~$15 |
| S3 storage | ~$1 |
| ECR storage | ~$1 |
| **Total** | **~$17/month** |

Instance can be stopped when not in use to avoid compute charges.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 📧 Contact

**Sourav Chandra**

**GitHub:** [souravchandra01](https://github.com/souravchandra01)

**LinkedIn:** [sourav-chandra-5a3112265](https://linkedin.com/in/sourav-chandra-5a3112265)

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Built with ❤️ for learning MLOps

</div>