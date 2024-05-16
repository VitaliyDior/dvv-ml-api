from typing import Any

from fastapi import APIRouter, Request, Depends

from app.models.classification import TextClassificationRequest, TextClassificationResponse
from app.models.predict import PredictResponse, PredictRequest
from app.models.similarity import TextSimilarityRequest, TextSimilarityResponse
from app.services.similarity import SimilarityService
from app.services.classification import TextClassificationService

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