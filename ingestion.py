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
tavily_map = TavilyMap(max_depth=2, max_breadth=30, max_pages=1000)
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
        site_extract=await tavily_extract.ainvoke(input={"urls": urls, "extract_depth": "advanced"})
        extract_count=len(site_extract.get("results",[]))
        #some urls may not get extracted because theey may not have any content, blocked or timeout etc

        if extract_count>0:
            log_success(f"TavilyExtract: Extracted {extract_count} documents from batch {batch_num}.")
        else:
            log_error(f"TavilyExtract: Extraction failed in batch {batch_num}.")
            #no url extracted in a batch
        return site_extract
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []
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
    for result in results:  #batch by batch
        if isinstance(result,Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:   #url by url in a batch
                document=Document(page_content=extracted_page["raw_content"], metadata={"url": extracted_page["url"]})
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")
    return all_pages


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
    site_map=tavily_map.invoke({"url":"https://docs.blender.org/manual/en/latest/",
                                "limit":100})
    log_success(f"Generated URL Map for {len(site_map)} URLs")
    log_success(
        f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs")

    url_batches = chunk_batches(list(site_map["results"]), batch_size=20)

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)

    log_success("break")








if __name__ == "__main__":
    asyncio.run(main())