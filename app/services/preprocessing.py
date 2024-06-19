from typing import List

from fastapi import FastAPI

from app.core.preprocessing import text_preprocessing, TextPreprocessingStep
from app.models.preprocessing import PreprocessingMethod
from app.services.text_preprocessors.nltk_text_preprocessor import NltkTextPreprocessor
from app.services.text_preprocessors.spacy_text_preprocessor import SpacyTextPreprocessor


class TextPreprocessingService:
    nltk_preprocessor: NltkTextPreprocessor
    spacy_preprocessor: SpacyTextPreprocessor

    def __init__(self, app: FastAPI):
        self.nltk_preprocessor = NltkTextPreprocessor(app)
        self.spacy_preprocessor = SpacyTextPreprocessor(app)

    def preprocess(self, text: str, method: PreprocessingMethod, pipeline: List[TextPreprocessingStep]) -> str:
        if method == PreprocessingMethod.nltk:
            return self.nltk_preprocessor.preprocess(text, pipeline)
        if method == PreprocessingMethod.spacy:
            return self.spacy_preprocessor.preprocess(text, pipeline)
        raise ValueError()
