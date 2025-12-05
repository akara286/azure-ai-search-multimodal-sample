import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rich.logging import RichHandler
from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.core.pipeline.policies import UserAgentPolicy

from azure.storage.blob.aio import BlobServiceClient

from backend.search_grounding import SearchGroundingRetriever
from backend.knowledge_agent import KnowledgeAgentGrounding
from backend.constants import USER_AGENT
from backend.multimodalrag import MultimodalRag
from backend.data_model import DocumentPerChunkDataModel
from backend.citation_file_handler import CitationFilesHandler


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Global variables to hold app state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    tokenCredential = DefaultAzureCredential()
    tokenProvider = get_bearer_token_provider(
        tokenCredential,
        "https://cognitiveservices.azure.com/.default",
    )

    chatcompletions_model_name = os.environ["AZURE_OPENAI_MODEL_NAME"]
    openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    search_endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
    search_index_name = os.environ["SEARCH_INDEX_NAME"]
    knowledge_agent_name = os.environ["KNOWLEDGE_AGENT_NAME"]
    openai_deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=search_index_name,
        credential=tokenCredential,
        user_agent_policy=UserAgentPolicy(base_user_agent=USER_AGENT),
    )
    data_model = DocumentPerChunkDataModel()

    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=tokenCredential,
        user_agent_policy=UserAgentPolicy(base_user_agent=USER_AGENT),
    )

    ka_retrieval_client = KnowledgeAgentRetrievalClient(
        agent_name=knowledge_agent_name,
        endpoint=search_endpoint,
        credential=tokenCredential,
    )

    knowledge_agent = KnowledgeAgentGrounding(
        ka_retrieval_client,
        search_client,
        index_client,
        data_model,
        search_index_name,
        knowledge_agent_name,
        openai_endpoint,
        openai_deployment_name,
        chatcompletions_model_name,
    )

    openai_client = AsyncAzureOpenAI(
        azure_ad_token_provider=tokenProvider,
        api_version="2024-08-01-preview",
        azure_endpoint=openai_endpoint,
        timeout=30,
    )

    search_grounding = SearchGroundingRetriever(
        search_client,
        openai_client,
        data_model,
        chatcompletions_model_name,
    )

    blob_service_client = BlobServiceClient(
        account_url=os.environ["ARTIFACTS_STORAGE_ACCOUNT_URL"],
        credential=tokenCredential,
    )
    artifacts_container_client = blob_service_client.get_container_client(
        os.environ["ARTIFACTS_STORAGE_CONTAINER"]
    )
    samples_container_client = blob_service_client.get_container_client(
        os.environ["SAMPLES_STORAGE_CONTAINER"]
    )

    mmrag = MultimodalRag(
        knowledge_agent,
        search_grounding,
        openai_client,
        chatcompletions_model_name,
        artifacts_container_client,
    )

    citation_files_handler = CitationFilesHandler(
        blob_service_client, samples_container_client
    )

    # Store in app state
    app_state["index_client"] = index_client
    app_state["citation_files_handler"] = citation_files_handler
    app_state["mmrag"] = mmrag

    yield

    # Shutdown
    await tokenCredential.close()
    await search_client.close()
    await index_client.close()
    await ka_retrieval_client.close()
    await openai_client.close()
    await blob_service_client.close()


app = FastAPI(lifespan=lifespan)


class CitationRequest(BaseModel):
    fileName: str


@app.get("/list_indexes")
async def list_indexes():
    index_client = app_state["index_client"]
    indexes = []
    async for index in index_client.list_indexes():
        indexes.append(index.name)
    return indexes


@app.post("/get_citation_doc")
async def get_citation_doc(request: Request):
    data = await request.json()
    handler = app_state["citation_files_handler"]
    return await handler.get_citation_doc(data["fileName"])


@app.post("/chat")
async def chat(request: Request):
    mmrag = app_state["mmrag"]
    return await mmrag._handle_request(request)


# Serve static files
current_directory = Path(__file__).parent
static_dir = current_directory / "static"

# Mount static directory
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host=host, port=port)
