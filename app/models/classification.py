from typing import Optional, SupportsFloat

from pydantic import BaseModel, StrictStr, Field, PositiveFloat


class TextClassificationRequest(BaseModel):
    text: StrictStr = Field(...,
                            title="text",
                            min_length=1,
                            description="Text for classification",
                            json_schema_extra={'example': "First text to classify"}
                            )


class TextClassificationResponse(BaseModel):
    label: StrictStr = Field(...,
                             title="label",
                             min_length=1,
                             description="Label assigned to text by classificator",
                             json_schema_extra={'example': "positive"}
                             )
    probability: Optional[PositiveFloat] = Field(...,
                             title="probability",
                             description="Probability of the label assigned to the classified text",
                             json_schema_extra={'example': "0.95"}
                             )
