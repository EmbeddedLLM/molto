from typing import Annotated

from pydantic import BaseModel, Field


class OpenAIChatCompletionRequest(BaseModel):
    model: Annotated[
        str,
        Field(description="The model to use for generating the chat completion."),
    ]
