from app.models.similarity import SimilarityAlgorithm
from textdistance import (Hamming, MLIPNS, Levenshtein, DamerauLevenshtein, JaroWinkler,
                          StrCmp95, NeedlemanWunsch, Gotoh, SmithWaterman)


class SimilarityService:
    Hamming = Hamming
    MLIPNS = MLIPNS
    Levenshtein = Levenshtein
    DamerauLevenshtein = DamerauLevenshtein
    JaroWinkler = JaroWinkler
    Strcmp95 = StrCmp95
    NeedlemanWunsch = NeedlemanWunsch
    Gotoh = Gotoh
    SmithWaterman = SmithWaterman

    def score(self, first_text: str, second_text: str, algo: SimilarityAlgorithm) -> float:
        scoring = getattr(self, algo.name)()
        return float(scoring(first_text, second_text))
