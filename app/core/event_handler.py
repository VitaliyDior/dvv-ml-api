import os
import pickle
import zipfile
import ssl
import nltk
from typing import Callable

import requests
from fastapi import FastAPI
from fastapi.logger import logger
from gensim.models import Doc2Vec

from app.core.config import Settings


def _startup_model(app: FastAPI, settings: Settings) -> None:
    file = open(settings.MODEL_PATH_FILE, 'rb')
    model_instance = pickle.load(file)
    app.state.model = model_instance

    file = open(settings.VECTORIZER_FILE, 'rb')
    app.state.vectorizer = pickle.load(file)
    file.close()

    file = open(settings.SCALER_FILE, 'rb')
    app.state.scaler = pickle.load(file)
    file.close()

    file = open(settings.LABEL_ENCODER_FILE, 'rb')
    app.state.label_encoder = pickle.load(file)
    file.close()

    # file = open(settings.DOC2VEC_MODEL_PATH_FILE, 'rb')
    app.state.d2v_model = Doc2Vec.load(settings.DOC2VEC_MODEL_PATH_FILE)
    # file.close()


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None
    app.state.vectorizer = None
    app.state.scaler = None
    app.state.label_encoder = None


def start_app_handler(app: FastAPI, settings: Settings) -> Callable:
    def startup() -> None:
        logger.info("Staring up!!!!")
        _download_nltk()
        _download_model(settings)
        _startup_model(app, settings)

    return startup


def _download_nltk() -> None:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    nltk.download('stopwords')


def _download_model(settings) -> None:
    logger.info('Checking model existence')
    if os.path.exists(settings.MODEL_PATH_FILE):
        logger.info('Model Exists')
        return
    logger.info('No model exists ... downloading')
    artifact_url = 'https://api.github.com/repos/{user}/{repo}/actions/artifacts/{art_id}/zip'.format(
        user=settings.GITHUB_USER,
        repo=settings.GITHUB_REPO,
        art_id=settings.GITHUB_ARTIFACT_ID,
    )

    if not os.path.exists(settings.ARTIFACTS_PATH):
        os.mkdir(settings.ARTIFACTS_PATH)

    logger.info('Downloading artifact')
    headers = {'Authorization': 'Bearer ' + settings.GITHUB_TOKEN}
    r = requests.get(artifact_url, allow_redirects=True, headers=headers)
    logger.info('Saving artifact to file')
    with open(settings.ARTIFACTS_PATH_FILE, 'wb') as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    print('UnZip archive')
    logger.info('UnZip artifact to file')
    with zipfile.ZipFile(settings.ARTIFACTS_PATH_FILE, 'r') as zip_ref:
        zip_ref.extractall(settings.STATIC_FILES_PATH)
    logger.info('Done!')


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown
