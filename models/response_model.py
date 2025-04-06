from pydantic import BaseModel
from typing import Optional as optional


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: optional[str] = None


class QueryResult(BaseModel):
    match: int
    page_number: int
    chunk_text: str
    document_name: str
    similarity_score: float


class QueryResponse(BaseModel):
    status: str
    results: list[QueryResult]


class LLMResponse(BaseModel):
    status: str
    response: str
    query: str
    vector_results: list[QueryResult]
