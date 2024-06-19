from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, StrictStr
from fastapi import Depends, FastAPI, Query

from app.core.preprocessing import TextPreprocessingStep


class PreprocessingMethod(str, Enum):
    nltk = "nltk"
    spacy = "spacy"

class PreprocessingQueryParams:
    def __init__(
            self,
            method: PreprocessingMethod = Query(..., description="Cool Description for foo"),
            pipeline: List[TextPreprocessingStep] = Query(('clear', 'tokenize', 'stop-words', 'stemming'), description="Cool Description for bar"),
    ):
        self.method = method
        self.pipeline = pipeline

class TextPreprocessingRequest(BaseModel):
    input_text: StrictStr = Field(..., title="input_text", description="Input text", json_schema_extra={'example': "Input text for preprocessing"})

class TextPreprocessingResponse(BaseModel):
    preprocessed_text: str = Field(..., title="preprocessed_text", description="Text preprocessed by selected method", json_schema_extra={'preprocessed_text': 'any text'})
