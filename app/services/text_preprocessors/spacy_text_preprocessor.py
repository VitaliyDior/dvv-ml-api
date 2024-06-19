import re
from typing import List
import spacy
from fastapi import FastAPI
from spacy.tokens import Token

from app.core.preprocessing import TextPreprocessingStep


class SpacyTextPreprocessor:
    nlp: spacy

    def __init__(self, app: FastAPI):
        self.nlp = app.state.spacy
        pass

    def preprocess(self, text: str, pipeline: List[TextPreprocessingStep]) -> str:

        if TextPreprocessingStep.Clear.value in pipeline:
            text = re.sub(r"[^a-zA-Z\']", " ", text)
            text = re.sub(r"[^\x00-\x7F]+", "", text)
            tokens = self.nlp(text)
            tokens = self._text_cleanup(tokens)
        else:
            tokens = self.nlp(text)

        if TextPreprocessingStep.Tokenize.value in pipeline:
            tokens = self._word_tokenize(tokens)

        if TextPreprocessingStep.StopWords.value in pipeline:
            tokens = self._stopwords_removal(tokens)

        if TextPreprocessingStep.Stemming.value in pipeline:
            tokens = self._text_stemmer(tokens)
            return " ".join(tokens).lower()

        return " ".join([token.text for token in tokens]).lower()

    def _text_cleanup(self, tokens: List[Token]) -> List[Token]:
        return [token for token in tokens if not (token.text.strip() == "" or token.like_num or token.is_currency)]

    def _word_tokenize(self, tokens: List[Token]) -> List[Token]:
        return tokens

    def _stopwords_removal(self, tokens: List[Token]) -> List[Token]:
        return [token for token in tokens if not token.is_stop]

    def _text_stemmer(self, tokens: List[Token]) -> List[str]:
        return [token.lemma_ for token in tokens]
