import os
from pathlib import Path

files = [
    # GitHub Actions
    ".github/workflows/ci-cd.yml",

    # Config
    "config/__init__.py",
    "config/config.py",

    # Source - Components
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/components/model_pusher.py",

    # Source - Pipelines
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",

    # Source - Entity
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",

    # Source - Cloud
    "src/cloud/__init__.py",
    "src/cloud/s3_handler.py",

    # Source - Utils
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/exception.py",
    "src/utils/common.py",

    # Notebooks
    "notebooks/01_eda.ipynb",

    # Data
    "data/raw/.gitkeep",

    # Artifacts (gitignored, just create folder)
    "artifacts/.gitkeep",

    # Models (gitignored)
    "models/.gitkeep",

    # Tests
    "tests/__init__.py",
    "tests/test_data_ingestion.py",
    "tests/test_transformation.py",
    "tests/test_prediction.py",

    # Static frontend
    "static/index.html",
    "static/style.css",
    "static/script.js",

    # Root files
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    ".env.example",
    ".gitignore",
    "README.md",
]

def create_project():
    for filepath in files:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            path.touch()
            print(f"Created: {path}")
        else:
            print(f"Exists:  {path}")

    print("\n✅ Project structure created successfully!")

if __name__ == "__main__":
    create_project()