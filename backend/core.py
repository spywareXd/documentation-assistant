from config import KNOWLEDGE_BASE_SUBJECT
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage, SystemMessage, HumanMessage
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",
                                        output_dimensionality=1536,
                                        task_type="RETRIEVAL_DOCUMENT")

vector_store=PineconeVectorStore(index_name="document-assistant", embedding=embeddings)
llm=ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)

@tool(response_format= "content_and_artifact")
def retrieval_context(query: str):
    """Tool for retrieving useful documentation context to help answer user queries"""

    retrieved_docs=vector_store.as_retriever().invoke(query, k=5)

    #join all retrieved docs to form context
    serialized="\n\n".join(f"Source: {doc.metadata.get('source', 'unknown')} \nContext: {doc.page_content}"
                                                                        for doc in retrieved_docs)

    #serialized-> content to LLM,  retrieved_docs -> artifacts to Application
    return serialized, retrieved_docs


def run_llm(query: str):
    """
    Run the RAG pipeline to answer a user query using context from retrieved documents

    Args:
        query: The question that user asked

    Returns:
        Dictionary containing:
            -answer: The generated answer to the question
            -context: The list of retrieved documents used for context
    """

    #defense prompting
    system_prompt=(
                   f"You are a helpful AI assistant that answers questions about {KNOWLEDGE_BASE_SUBJECT} documents"
                   "You have access to a tool that retrieves relevant documents to form context"
                   "Use the tool to find relevant information and context before answering the question"
                   "Always mention the sources you use in your answers"
                   "If there you can not find the answer for the question in the retrieved documents, say so")

    tools=[retrieval_context]


    agent=create_agent(model=llm, tools=[retrieval_context], system_prompt=system_prompt)
    messages=[HumanMessage(content=query)]
    response=agent.invoke({"messages": messages})
    answer=response["messages"][-1].content #LLM may call function multiple time until it arrives at final answer. We only extract the final message of AI
    #extract context documents:
    context=[]
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):  #if message has ToolMessage with artifact attribute
            if isinstance(message.artifact, list):
                context.extend(message.artifact)

    return {"answer": answer[0]['text'], "context": context}


if __name__ == "__main__":
    result=run_llm(query="Hello")
    print(result)
