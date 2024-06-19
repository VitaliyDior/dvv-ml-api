import re
from typing import List

from fastapi import FastAPI
from nltk.tokenize import word_tokenize
from app.core.preprocessing import TextPreprocessingStep
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class NltkTextPreprocessor:
    def __init__(self, app: FastAPI):

        pass

    def preprocess(self, text: str, pipeline: List[TextPreprocessingStep]) -> str:
        local_text = text
        if TextPreprocessingStep.Clear.value in pipeline:
            local_text = self.text_cleanup(local_text)

        if TextPreprocessingStep.Tokenize.value in pipeline:
            local_text = self.word_tokenize(local_text)
        if TextPreprocessingStep.StopWords.value in pipeline:
            local_text = self.stopwords_removal(local_text)
        if TextPreprocessingStep.Stemming.value in pipeline:
            local_text = self.text_stemmer(local_text)

        if type(local_text) == list:
            return ' '.join(local_text)
        return  local_text

    def text_cleanup(self, text: str) -> str:
        no_tags_text = re.sub(r"<.*?>", ' ', text.lower())
        no_symbols_text = re.sub(r"[^\w\s]", ' ', no_tags_text)
        no_numbers_text = re.sub(r"\d", ' ', no_symbols_text)
        no_dup_spaces_text = re.sub(r"\s\s+", ' ', no_numbers_text)
        return no_dup_spaces_text

    def word_tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def stopwords_removal(self, words: List[str]) -> List[str]:
        return [word for word in words if word.lower() not in stopwords.words('english')]

    def text_stemmer(self, words: List[str]) -> List[str]:
        stemmer = PorterStemmer()
        stemmed = []
        for word in words:
            stem_word = stemmer.stem(word)
            stemmed.append(stem_word)
        return stemmed