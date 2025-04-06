from pydantic import BaseModel


class QueryRequest(BaseModel):
    input: str
    top_k: int = 5
