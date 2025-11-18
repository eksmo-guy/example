from pydantic import BaseModel

from eksmo_src.eksmo_types import Usage


class AiAnswer(BaseModel):
    message: str
    usage: Usage
