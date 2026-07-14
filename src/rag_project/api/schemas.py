from pydantic import BaseModel, Field

from rag_project.retrieval.retriever import RetrievedChunk


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
