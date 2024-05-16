import operator

from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

from app.core.preprocessing import text_preprocessing
from app.models.classification import TextClassificationResponse


class TextClassificationService:
    model: RandomForestClassifier
    vectorizer: TfidfVectorizer
    scaler: MaxAbsScaler
    label_encoder: LabelEncoder

    def __init__(self, app: FastAPI):
        self.model = app.state.model
        self.vectorizer = app.state.vectorizer
        self.scaler = app.state.scaler
        self.label_encoder = app.state.label_encoder

    def classify(self, text: str) -> TextClassificationResponse:
        tokens = text_preprocessing(text, ('clear', 'tokenize', 'stop-words', 'stemming'), {})
        text_preprocessed = ' '.join(tokens)
        vectorized_text = self.vectorizer.transform([text_preprocessed])
        x = self.scaler.transform(vectorized_text)
        y = self.model.predict_proba(x)
        result = dict(zip(self.label_encoder.classes_, y[0]))
        result_key = max(result.items(), key=operator.itemgetter(1))[0]
        resp = TextClassificationResponse
        resp.label = result_key
        resp.probability = round(result[result_key], 2)
        return resp
