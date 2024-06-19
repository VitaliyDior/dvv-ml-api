from typing import Optional, SupportsFloat

from pydantic import BaseModel, StrictStr, Field, PositiveFloat


class TextTopicsRequest(BaseModel):
    text: StrictStr = Field(...,
                            title="text",
                            min_length=1,
                            description="Text to break by topics",
                            json_schema_extra={'example': "First text to classify"}
                            )

