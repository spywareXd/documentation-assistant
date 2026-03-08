import asyncio
import ssl
import os
from dotenv import load_dotenv
from sqlalchemy.testing.suite.test_reflection import metadata

load_dotenv()
from typing import Dict, List, Any
#certifi verifies that the HTML request is certified and trusted
import certifi

from langchain_chroma import Chroma
#chroma -> local vector store if not using PineCone
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
#This splitter tries to split text intelligently by gradually using smaller separators.
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)



ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=1536
)

vector_store=PineconeVectorStore(index_name="document-assistant", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()









async def main():
    """Main async function to execute the entire process"""
    log_header("DOCUMENT INGESTION PIPELINE")
    log_info("Tavily Crawl: Starting to Crawl documentation", Colors.PURPLE)

    #tavily_crawl is a tool when used via the langchain package hence it can be invoked
    res = tavily_crawl.invoke(
        {
            "url": "https://docs.blender.org/manual/en/latest/",
            "max_depth": 2,
            "extract_depth": "advanced",
            "limit" : 35
        }
    )
    all_docs = [Document(page_content=results["raw_content"], metadata={"source": results["url"]}) for results in res["results"]]
    log_success(f"Crawling Successful: Crawled {len(all_docs)} URLS")








if __name__ == "__main__":
    asyncio.run(main())