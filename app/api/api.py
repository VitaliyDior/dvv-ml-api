from typing import Any, List

from fastapi import APIRouter, Request, Depends

from app.models.classification import TextClassificationRequest, TextClassificationResponse
from app.models.predict import PredictResponse, PredictRequest
from app.models.preprocessing import TextPreprocessingRequest, TextPreprocessingResponse, \
    PreprocessingQueryParams
from app.models.similarity import TextSimilarityRequest, TextSimilarityResponse
from app.models.topics import TextTopicsRequest
from app.services.preprocessing import TextPreprocessingService
from app.services.similarity import SimilarityService
from app.services.classification import TextClassificationService
from app.services.text_topics import TextTopicsService

api_router = APIRouter()


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_text = payload.input_text
    model = request.app.state.model

    predict_value = model.predict(input_text)
    return PredictResponse(result=predict_value)


@api_router.post("/similarity", response_model=TextSimilarityResponse)
async def similarity(payload: TextSimilarityRequest, service: SimilarityService = Depends(SimilarityService)) \
        -> TextSimilarityResponse:
    score = service.score(first_text=payload.first_text,
                          second_text=payload.second_text,
                          algo=payload.algo)

    return TextSimilarityResponse(first_text=payload.first_text,
                                  second_text=payload.second_text,
                                  algo=payload.algo,
                                  score=score)


@api_router.post("/classification", response_model=TextClassificationResponse)
async def classification(request: Request, payload: TextClassificationRequest) \
        -> TextClassificationResponse:
    return TextClassificationService(request.app).classify(payload.text)


@api_router.post("/topics")
async def topics(request: Request, payload: TextTopicsRequest) -> List[List[str]]:
    return TextTopicsService(request.app).split_to_topics(payload.text)


@api_router.post("/preprocess", response_model=TextPreprocessingResponse)
async def preprocess(request: Request, payload: TextPreprocessingRequest,
                     query: PreprocessingQueryParams = Depends()) -> TextPreprocessingResponse:
    response = TextPreprocessingResponse(preprocessed_text='')
    response.preprocessed_text = TextPreprocessingService(request.app).preprocess(payload.input_text, query.method,
                                                                                  query.pipeline)
    return response
