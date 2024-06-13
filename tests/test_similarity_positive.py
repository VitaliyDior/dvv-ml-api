from http import HTTPStatus

from fastapi.testclient import TestClient

from app import settings
from app.models.similarity import SimilarityAlgorithm

first_text = "The bottle is empty."
second_text = "There is nothing in the bottle."


def test_healthz(client: TestClient) -> None:
    for algo in SimilarityAlgorithm:
        r = client.post(f"{settings.API_V1_STR}/similarity", json={"first_text": first_text,
                                                                   "second_text": second_text,
                                                                   "algo": algo.value})
        assert r.status_code == HTTPStatus.OK
        assert type(r.json()['score']) is float
        assert r.json()['first_text'] == first_text
        assert r.json()['second_text'] == second_text
        assert r.json()['algo'] == algo.value
