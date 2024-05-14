from typing import Any, List

from app.core.preprocessing import text_preprocessing
from app.models.classification import TextClassificationResponse


class TextClassificationService:

    def preprocess(self, text: str) -> str:
        tokens = text_preprocessing(text, ('clear', 'tokenize', 'stop-words', 'stemming'), {})
        return ' '.join(tokens)

    def classify(self, text: str) -> TextClassificationResponse:
        resp = TextClassificationResponse
        resp.label = 'positive'
        resp.probability = 0.95
        print(resp)
        return resp
