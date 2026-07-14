from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from rag_project.retrieval.retriever import RetrievedChunk, Retriever

SYSTEM_PROMPT = (
    "You are a documentation assistant answering questions about FastAPI. "
    "Answer only using the provided context. If the context does not contain "
    "the answer, say so explicitly instead of guessing."
)


class QAState(BaseModel):
    question: str
    chunks: list[RetrievedChunk] = []
    answer: str = ""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.source_path}]\n{c.content}" for c in chunks)


def build_qa_graph(
    retriever: Retriever, generation_model: str, api_key: str, k: int = 4
) -> CompiledStateGraph[QAState, None, QAState, QAState]:
    llm = ChatAnthropic(model=generation_model, api_key=api_key)

    def retrieve(state: QAState) -> dict[str, list[RetrievedChunk]]:
        return {"chunks": retriever.similarity_search(state.question, k=k)}

    def generate(state: QAState) -> dict[str, str]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{_format_context(state.chunks)}\n\nQuestion: {state.question}"
            ),
        ]
        response = llm.invoke(messages)
        return {"answer": response.text}

    graph: StateGraph[QAState] = StateGraph(QAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
