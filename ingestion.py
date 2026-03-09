import asyncio
import re
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


semaphore = asyncio.Semaphore(2)  #to avoid rate limits
MAIN_URL="https://darksouls.fandom.com/wiki/Lore"

"""URL Filtering"""
BAD_URLS = [
    "https://auth.fandom.com/"
    "auth.fandom.com/",
    "veaction=edit",
    "action=edit",
    "section=",
    "oldid=",
    "diff=",
    "curid=",
    "replyId=",
    "commentId=",
    "/f/p/",  # forum/discussion style pages if you don't want them
    "/wiki/Special:",
    "/wiki/File:",
    "/wiki/Category:",
    "/wiki/Template:",
    "/wiki/Help:",
    "/wiki/User:",
    "/wiki/User_blog:",
    "/wiki/Message_Wall:",
    r"/Gallery",
    r"/Images",
]

GOOD_URLS = [
    "https://darksouls.fandom.com/wiki/"
]


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=1536,

)

vector_store=PineconeVectorStore(index_name="document-assistant", embedding=embeddings)
#vector_store.delete(delete_all=True, namespace="__default__") #to reset VectorStore
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=2, max_breadth=60, max_pages=1000)
tavily_crawl = TavilyCrawl()


#eg: Map->200 URLs   input: [200URLS]
def chunk_batches(sites: List[str], batch_size: int = 20) -> List[List[str]]:
    """ To Split URLS into batches and return a list containing batches of URLS"""
    chunks=[]
    for i in range (0, len(sites), batch_size):
        chunks.append(sites[i:i+batch_size])
    return chunks
#eg: Output: [[20urls] ,[20urls], [20urls]...]


#input -> One batch; [20urls]
async def extract_batch(urls: List[str], batch_num : int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs"""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs.",
            Colors.BLUE,
        )
        #multiple batches can run concurrently
        site_extract=await tavily_extract.ainvoke(input={"urls": urls, "extract_depth": "basic",
                                                                            }) #async invoke
        extract_count=len(site_extract.get("results",[]))
        #some urls may not get extracted because they may not have any content, blocked or timeout etc

        # Normalize weird wrapper responses
        if isinstance(site_extract, dict):
            results = site_extract.get("results", [])
            if not isinstance(results, list):
                log_error(
                    f"TavilyExtract: Batch {batch_num} returned dict but results was not a list: {site_extract}"
                )
                return {"results": []}

            if results:
                log_success(
                    f"TavilyExtract: Extracted {len(results)} URLs from batch {batch_num}."
                )
            else:
                log_error(
                    f"TavilyExtract: Extraction failed in batch {batch_num}. Response: {site_extract}"
                )
            return {"results": results}

        # Sometimes wrappers return a list directly
        elif isinstance(site_extract, list):
            log_warning(
                f"TavilyExtract: Batch {batch_num} returned a list instead of dict. Normalizing response."
            )
            if site_extract:
                log_success(
                    f"TavilyExtract: Extracted {len(site_extract)} documents from batch {batch_num}."
                )
            else:
                log_error(
                    f"TavilyExtract: Extraction failed in batch {batch_num}. Empty list returned."
                )
            return {"results": site_extract}

        # Sometimes wrappers return a string / error message
        elif isinstance(site_extract, str):
            log_error(
                f"TavilyExtract: Batch {batch_num} returned a string instead of dict: {site_extract}"
            )
            return {"results": []}

        else:
            log_error(
                f"TavilyExtract: Batch {batch_num} returned unexpected type {type(site_extract).__name__}: {site_extract}"
            )
            return {"results": []}

    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return {"results": []}
#output-> dict output of every url in the batch as a list: [20 dicts]


#helper function for extraction to feed batches;
async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches.", Colors.DARKCYAN)

    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    #tasks -> List[List[Dict[..]], List[Dict[..]], List[Dict[..]]....20 batches] -> async coroutines

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # travers url by url and convert it to document and append to one list
    all_pages=[]
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
            continue

        for extracted_page in result.get("results", []):
            if not isinstance(extracted_page, dict):
                log_warning(f"TavilyExtract: Skipping malformed extracted page: {extracted_page}")
                continue

            raw_content = extracted_page.get("raw_content", "")
            url = extracted_page.get("url", "unknown")

            if raw_content:
                document = Document(
                    page_content=raw_content,
                    metadata={"url": url}
                )
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages




async def batch_indexing(documents : List[Document], batch_size: int = 50):

    log_header("VECTOR INDEXING PHASE")
    log_info(f"VectorIndexing: Starting concurrent indexing of {len(documents)} documents.", Colors.DARKCYAN)

    #split into batches List[List[Doc]]
    docs_batches=[documents[i : i+batch_size] for i in range(0, len(documents), batch_size)]

    log_info(
        f"VectorIndexing: Split into {len(docs_batches)} batches of {batch_size} documents each"
    )

    """Only use this if Embeddings Model support Input Chunk Limitters(Like OpenAI). Otherwise it will cause rate limit exceeded errors"""
    #another async routine to process each batch
    # async def add_batch(batch: List[Document], batch_num: int):
    #
    #         try:
    #             await vector_store.aadd_documents(batch)  #async add
    #             log_success(f"VectorStore: Successfully added batch {batch_num} with {len(batch)} documents to Vector Store")
    #
    #         except Exception as e:
    #             log_error(f"VectorStore: Failed to add batch {batch_num} - {e}")
    #             return False
    #         return True
    #
    # task=[add_batch(b,i+1) for i, b in enumerate(docs_batches)]
    # results=await asyncio.gather(*task, return_exceptions=True)
    #
    # # Count successful batches
    # successful= sum([1 for i in results if i==True])
    #
    # if successful==len(docs_batches):
    #     log_success(f"VectorStore: Successfully index all {len(docs_batches)} batches.")
    # else:
    #     log_warning(f"VectorStore: Processed {successful}/{len(docs_batches)} batches")

    """Gemini Implementation:"""

    successful = 0
    for i, batch in enumerate(docs_batches, start=1):
        try:
            vector_store.add_documents(batch)
            log_success(
                f"VectorStore: Successfully added batch {i} with {len(batch)} documents to Vector Store"
            )
            successful += 1

            await asyncio.sleep(1)

        except Exception as e:
            log_error(f"VectorStore: Failed to add batch {i} - {e}")

    if successful == len(docs_batches):
        log_success(f"VectorStore: Successfully indexed all {len(docs_batches)} batches.")
    else:
        log_warning(f"VectorStore: Processed {successful}/{len(docs_batches)} batches")











async def main():
    """Main async function to execute the entire process"""
    log_header("DOCUMENT INGESTION PIPELINE")
    #log_info("Tavily Crawl: Starting to Crawl documentation", Colors.PURPLE)

    #tavily_crawl is a tool when used via the langchain package hence it can be invoked
    """Tavily Crawl Implementation: """
    # res = tavily_crawl.invoke(
    #     {
    #         "url": "https://docs.blender.org/manual/en/latest/",
    #         "max_depth": 2,
    #         "extract_depth": "advanced",
    #         "limit" : 35
    #     }
    # )
    # all_docs = [Document(page_content=results["raw_content"], metadata={"source": results["url"]}) for results in res["results"]]
    # log_success(f"Crawling Successful: Crawled {len(all_docs)} URLS")

    """ Tavily Map and Extract Implementation: """

    log_info("Generating URL Map for the site")
    #Apply instruction and regex filtering
    site_map=tavily_map.invoke({"url": MAIN_URL,
                                "limit":200,
                                "instructions": "Map only actual Dark Souls story lore and character lore. Ignore Everything else",
                                "exclude_paths": BAD_URLS})



    urls = list(site_map.get("results", []))

    log_success(f"Generated URL Map for {len(site_map.get('results'))} URLs")
    log_info("Mapped URLs: ", Colors.YELLOW)
    for u in urls:
        print(u)


    url_batches = chunk_batches(urls, batch_size=20)

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)

    #Chunk all_docs
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_split=text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter:Created {len(docs_split)} chunks from {len(all_docs)} documents"
    )
    """To optimize chunking and avoid rate limits: """
    print("\n" + "=" * 60)
    print("CHUNK DEBUG REPORT")
    print("=" * 60)

    # Per-document chunk counts
    doc_chunk_stats = []

    for doc in all_docs:
        single_chunks = text_splitter.split_documents([doc])
        content_len = len(doc.page_content)
        approx_tokens = content_len / 4  # rough estimate
        chunk_count = len(single_chunks)

        doc_chunk_stats.append({
            "url": doc.metadata.get("url", "unknown"),
            "char_length": content_len,
            "approx_tokens": int(approx_tokens),
            "chunk_count": chunk_count,
            "avg_chars_per_chunk": int(content_len / chunk_count) if chunk_count > 0 else 0,
        })

    # Sort by chunk_count descending
    doc_chunk_stats.sort(key=lambda x: x["chunk_count"], reverse=True)

    print(f"\nTotal documents: {len(all_docs)}")
    print(f"Total chunks: {len(docs_split)}")
    print(f"Average chunks per document: {len(docs_split)} / {len(all_docs):.2f}")

    print("\nTop 10 documents producing the most chunks:\n")
    for i, stat in enumerate(doc_chunk_stats[:10], start=1):
        print(f"{i}. URL: {stat['url']}")
        print(f"   Characters: {stat['char_length']}")
        print(f"   Approx tokens: {stat['approx_tokens']}")
        print(f"   Chunks: {stat['chunk_count']}")
        print(f"   Avg chars/chunk: {stat['avg_chars_per_chunk']}")
        print("-" * 60)


    #index coroutines for batch indexing
    await batch_indexing(docs_split, batch_size=25)

    #success log
    log_header("INGESTION PIPELINE COMPLETED")
    log_success("Documentation has been indexed to PineCone vector store")
    log_info(f"No. of URLs scraped : {len(urls)}")
    log_info(f"No. of Documents indexed:{len(docs_split)} ")




if __name__ == "__main__":
    asyncio.run(main())