import operator

from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from transformers import BertTokenizer, BertForSequenceClassification

from app.core.preprocessing import text_preprocessing
from app.models.classification import TextClassificationResponse, TextClassificationBertResponse
import torch
import numpy as np

class TextClassificationService:
    model: RandomForestClassifier
    vectorizer: TfidfVectorizer
    scaler: MaxAbsScaler
    label_encoder: LabelEncoder
    bert_tokenizer: BertTokenizer
    bert_classifier: BertForSequenceClassification

    bert_id2label = {0: 'Books & Literature', 1: 'Arts & Entertainment', 2: 'Business & Industrial', 3: 'Online Communities', 4: 'Adult', 5: 'People & Society', 6: 'Sensitive Subjects', 7: 'News', 8: 'Health', 9: 'Shopping', 10: 'Beauty & Fitness', 11: 'Hobbies & Leisure', 12: 'Autos & Vehicles', 13: 'Food & Drink', 14: 'Computers & Electronics', 15: 'Games', 16: 'Law & Government', 17: 'Reference', 18: 'Sports', 19: 'Jobs & Education', 20: 'Real Estate', 21: 'Internet & Telecom', 22: 'Travel & Transportation', 23: 'Pets & Animals', 24: 'Home & Garden', 25: 'Science', 26: 'Finance'}

    def __init__(self, app: FastAPI):
        self.model = app.state.model
        self.vectorizer = app.state.vectorizer
        self.scaler = app.state.scaler
        self.label_encoder = app.state.label_encoder,
        self.bert_tokenizer = app.state.bert_tokenizer
        self.bert_classifier =  app.state.bert_model

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


    def classify_bert(self, text: str) -> TextClassificationBertResponse:
        encoding = self.bert_tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
        outputs = self.bert_classifier(**encoding)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        label_ids = np.where(probs >= 0.5)
        resp = TextClassificationBertResponse
        labels = []
        for i in list(label_ids[0]):
            item = {"label": self.bert_id2label[i], "probability": float(probs[i])}
            labels.append(item)
        resp.labels = labels
        return resp


