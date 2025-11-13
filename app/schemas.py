from pydantic import BaseModel

class AskResponse(BaseModel):
    answer: str

class AskQuery(BaseModel):
    question: str
