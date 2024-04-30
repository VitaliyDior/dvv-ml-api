from http import HTTPStatus

from fastapi.testclient import TestClient

from app.core.config import settings
from app.models.similarity import SimilarityAlgorithm

first_text = "The bottle is empty."
second_text = "There is nothing in the bottle."


def test_similarity_empty_first_text(client: TestClient) -> None:
    r = client.post(f"{settings.API_V1_STR}/similarity", json={"first_text": '',
                                                               "second_text": second_text,
                                                               "algo": SimilarityAlgorithm.Hamming.value})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_similarity_empty_second_text(client: TestClient) -> None:
    r = client.post(f"{settings.API_V1_STR}/similarity", json={"first_text": first_text,
                                                               "second_text": '',
                                                               "algo": SimilarityAlgorithm.Hamming.value})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_similarity_none_first_text(client: TestClient) -> None:
    r = client.post(f"{settings.API_V1_STR}/similarity", json={"first_text": None,
                                                               "second_text": second_text,
                                                               "algo": SimilarityAlgorithm.Hamming.value})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_similarity_none_second_text(client: TestClient) -> None:
    r = client.post(f"{settings.API_V1_STR}/similarity", json={"first_text": first_text,
                                                               "second_text": None,
                                                               "algo": SimilarityAlgorithm.Hamming.value})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_similarity_invalid_algo(client: TestClient) -> None:
    r = client.post(f"{settings.API_V1_STR}/similarity", json={"first_text": first_text,
                                                               "second_text": second_text,
                                                               "algo": 'foobar'})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
