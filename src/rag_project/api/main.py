from functools import lru_cache

from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph

from rag_project.api.schemas import QueryRequest, QueryResponse
from rag_project.config import get_settings
from rag_project.graph.qa_graph import QAState, build_qa_graph
from rag_project.retrieval.retriever import Retriever

app = FastAPI(title="FastAPI Docs RAG")


@lru_cache
def get_qa_graph() -> CompiledStateGraph[QAState, None, QAState, QAState]:
    """Cached so the embedding model and Chroma client load once per process
    instead of once per request (see Retriever)."""
    settings = get_settings()
    retriever = Retriever(settings.chroma_persist_dir, settings.embedding_model)
    return build_qa_graph(retriever, settings.generation_model, settings.anthropic_api_key)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    graph = get_qa_graph()
    result = graph.invoke(QAState(question=request.question))
    return QueryResponse(answer=result["answer"], sources=result["chunks"])
