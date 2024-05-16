import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Template"
    API_V1_STR: str = "/api/v1"

    PROJECT_ROOT_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    STATIC_FILES_PATH: str = PROJECT_ROOT_PATH + "/static"
    MISC_FILES_PATH: str = STATIC_FILES_PATH + "/misc"
    MODEL_PATH: str = STATIC_FILES_PATH + "/models"

    MODEL_PATH_FILE: str = MODEL_PATH + "/random_forest_model.pickle"
    LABEL_ENCODER_FILE: str = MISC_FILES_PATH + "/label_encoder.pickle"
    SCALER_FILE: str = MISC_FILES_PATH + "/min_max_scaler.pickle"
    VECTORIZER_FILE: str = MISC_FILES_PATH + "/tfidf_vectorizer.pickle"
    ARTIFACTS_PATH: str = PROJECT_ROOT_PATH + "/var"
    ARTIFACTS_PATH_FILE: str = ARTIFACTS_PATH + "/artifacts.zip"
    LOG_LEVEL: str = 'debug'
    DOTENV_PATH: str | None = None
    GITHUB_ARTIFACT_ID: str | None = None
    GITHUB_USER: str | None = None
    GITHUB_REPO: str | None = None
    GITHUB_TOKEN: str | None = None

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT_PATH + "/.env", case_sensitive=True)

settings = Settings()
