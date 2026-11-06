"""Frozen baseline: Gemini embeddings + Pinecone + Gemini agent.

Kept only so experiment E0 can measure the original system.
`fixed=False` reproduces the original behaviour exactly (including its bugs);
`fixed=True` applies the bug fixes so they can be reported as a separate row.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from baseline.config import KNOWLEDGE_BASE_SUBJECT

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536,
    task_type="RETRIEVAL_DOCUMENT",
)
vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME", "document-assistant"), embedding=embeddings
)
llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)

SYSTEM_PROMPT = (
    f"You are a helpful AI assistant that answers questions about {KNOWLEDGE_BASE_SUBJECT} documents"
    "You have access to a tool that retrieves relevant documents to form context"
    "Use the tool to find relevant information and context before answering the question"
    "Always mention the sources you use in your answers"
    "If there you can not find the answer for the question in the retrieved documents, say so"
)


def retrieve(query: str, fixed: bool = False, k: int = 5):
    """Retrieval step on its own, so the eval harness can score it without the LLM."""
    if fixed:
        # Bug 1 fix: k must go through search_kwargs; as a call kwarg it is ignored (default 4).
        return vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)
    return vector_store.as_retriever().invoke(query, k=k)


def _source(doc, fixed: bool) -> str:
    # Bug 2 fix: ingestion stores the URL under "url", not "source".
    if fixed:
        return doc.metadata.get("url") or doc.metadata.get("source") or "unknown"
    return doc.metadata.get("source", "unknown")


def _text(content) -> str:
    # Bug 3 fix: content may be a plain string or a list of content blocks.
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def run_llm(query: str, fixed: bool = False):
    """Answer a query with the original agent. Returns {"answer", "context"}."""

    @tool(response_format="content_and_artifact")
    def retrieval_context(query: str):
        """Tool for retrieving useful documentation context to help answer user queries"""
        docs = retrieve(query, fixed=fixed)
        serialized = "\n\n".join(
            f"Source: {_source(d, fixed)} \nContext: {d.page_content}" for d in docs
        )
        return serialized, docs

    agent = create_agent(model=llm, tools=[retrieval_context], system_prompt=SYSTEM_PROMPT)
    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    context = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and isinstance(getattr(message, "artifact", None), list):
            context.extend(message.artifact)

    content = response["messages"][-1].content
    answer = _text(content) if fixed else content[0]["text"]
    return {"answer": answer, "context": context}


if __name__ == "__main__":
    print(run_llm("Hello", fixed=True))
