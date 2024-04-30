from enum import Enum
from pydantic import BaseModel, Field, StrictStr


class SimilarityAlgorithm(str, Enum):
    Hamming = "hamming"
    MLIPNS = "mlipns"
    Levenshtein = "levenshtein"
    DamerauLevenshtein = "damerau-levenshtein"
    JaroWinkler = "jaro"
    Strcmp95 = "strcmp95"
    NeedlemanWunsch = "needleman-wunsch"
    Gotoh = "gotohh"
    SmithWaterman = "smith-waterman"


class TextSimilarityRequest(BaseModel):
    first_text: StrictStr = Field(...,
                                  title="first_text",
                                  min_length=1,
                                  description="First text to compare",
                                  json_schema_extra={'example': "First text to calculate similarity"}
                                  )

    second_text: StrictStr = Field(...,
                                   title="second_text",
                                   min_length=1,
                                   description="Second text to compare",
                                   json_schema_extra={'example': "Second text to compare"}
                                   )

    algo: SimilarityAlgorithm = Field(...,
                                      title="Similarity Algorithm",
                                      description="One of similarity algorithms name",
                                      json_schema_extra={'example': "hamming"}
                                      )


class TextSimilarityResponse(BaseModel):
    first_text: StrictStr = Field(...,
                                  title="first_text",
                                  description="First text passed to similarity algorithm",
                                  json_schema_extra={'example': "First text to calculate similarity"}
                                  )

    second_text: StrictStr = Field(...,
                                   title="second_text",
                                   description="Second text passed to similarity algorithm",
                                   json_schema_extra={'example': "Second text to compare"}
                                   )

    algo: SimilarityAlgorithm = Field(...,
                                      title="Similarity Algorithm",
                                      description="Name of similarity algorithms applied",
                                      json_schema_extra={'example': "hamming"}
                                      )
    score: float = Field(...,
                         title="Similarity Score",
                         description="Similarity Score calculated by chosen algorithm",
                         json_schema_extra={'example': "1.0"}
                         )
