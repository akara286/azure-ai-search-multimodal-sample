"""
FastAPI Application - V2 Modernized RAG Backend

Entry point for the modernized multimodal RAG API using:
- Azure AI Search Knowledge Base API (2025-11-01-preview)
- GPT-5-mini for query planning and answer generation
- Hybrid search with semantic reranking and query rewrite
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from azure.identity import DefaultAzureCredential
from azure.identity.aio import get_bearer_token_provider
from azure.storage.blob.aio import BlobServiceClient
from openai import AsyncAzureOpenAI

from models import SearchConfigV2
from query_planner import QueryPlanner
from knowledge_base import KnowledgeBaseClient
from search_retriever import HybridSearchRetriever
from answer_generator import AnswerGenerator
from citation_handler import CitationHandler, BlobHelper
from pipeline import RAGPipelineV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backend_v2.app")

# Application state
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    logger.info("Initializing V2 RAG Backend...")

    # Azure credentials
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )

    # Configuration from environment
    openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-5-mini")

    search_endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
    search_index = os.environ["SEARCH_INDEX_NAME"]
    knowledge_base_name = os.environ.get("KNOWLEDGE_BASE_NAME", os.environ.get("KNOWLEDGE_AGENT_NAME"))

    storage_url = os.environ["ARTIFACTS_STORAGE_ACCOUNT_URL"]
    artifacts_container = os.environ["ARTIFACTS_STORAGE_CONTAINER"]

    # Initialize OpenAI client
    openai_client = AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2025-01-01-preview",  # Latest API version for GPT-5
        azure_endpoint=openai_endpoint,
        timeout=60,
    )

    # Initialize Query Planner
    query_planner = QueryPlanner(
        openai_client=openai_client,
        model_name=model_name,
    )

    # Initialize Knowledge Base Client (optional, for agentic retrieval)
    knowledge_base_client: Optional[KnowledgeBaseClient] = None
    if knowledge_base_name:
        try:
            knowledge_base_client = KnowledgeBaseClient(
                endpoint=search_endpoint,
                credential=credential,
                knowledge_base_name=knowledge_base_name,
                index_name=search_index,
                azure_openai_endpoint=openai_endpoint,
                model_deployment=openai_deployment,
                model_name=model_name,
            )
            await knowledge_base_client.initialize()
            logger.info(f"Knowledge Base '{knowledge_base_name}' initialized")
        except Exception as e:
            logger.warning(f"Knowledge Base initialization failed: {e}. Falling back to hybrid search.")
            knowledge_base_client = None

    # Initialize Hybrid Search Retriever
    search_retriever = HybridSearchRetriever(
        endpoint=search_endpoint,
        index_name=search_index,
        credential=credential,
        query_planner=query_planner,
    )
    await search_retriever.initialize()

    # Initialize Blob Storage
    blob_service_client = BlobServiceClient(
        account_url=storage_url,
        credential=credential,
    )
    artifacts_container_client = blob_service_client.get_container_client(artifacts_container)

    # Initialize Citation Handler
    citation_handler = CitationHandler(
        blob_service_client=blob_service_client,
        artifacts_container=artifacts_container_client,
    )

    # Initialize Answer Generator
    blob_helper = BlobHelper(citation_handler)
    answer_generator = AnswerGenerator(
        openai_client=openai_client,
        model_name=model_name,
        blob_helper=blob_helper,
    )

    # Initialize Pipeline
    pipeline = RAGPipelineV2(
        query_planner=query_planner,
        knowledge_base_client=knowledge_base_client,
        search_retriever=search_retriever,
        answer_generator=answer_generator,
        citation_handler=citation_handler,
    )

    # Store in app state
    app_state["pipeline"] = pipeline
    app_state["search_retriever"] = search_retriever
    app_state["citation_handler"] = citation_handler
    app_state["blob_service_client"] = blob_service_client

    logger.info("V2 RAG Backend initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down V2 RAG Backend...")
    if knowledge_base_client:
        await knowledge_base_client.close()
    await search_retriever.close()
    await blob_service_client.close()


# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG API V2",
    description="Modernized RAG API with Azure AI Search 2025 features",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v2/chat")
async def chat(request: Request):
    """
    Chat endpoint with streaming SSE response.

    Request body:
    {
        "query": "user question",
        "chatThread": [...previous messages...],
        "config": {
            "chunk_count": 10,
            "use_semantic_ranker": true,
            "use_query_rewrite": true,
            "use_knowledge_base": false,
            "use_streaming": true,
            "query_rewrite_count": 5,
            "scoring_profile": null
        }
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = body.get("query", "")
    chat_thread = body.get("chatThread", [])
    config_dict = body.get("config", {})

    config: SearchConfigV2 = {
        "chunk_count": config_dict.get("chunk_count", 10),
        "use_semantic_ranker": config_dict.get("use_semantic_ranker", True),
        "use_query_rewrite": config_dict.get("use_query_rewrite", True),
        "use_knowledge_base": config_dict.get("use_knowledge_base", False),
        "use_streaming": config_dict.get("use_streaming", True),
        "query_rewrite_count": config_dict.get("query_rewrite_count", 5),
        "scoring_profile": config_dict.get("scoring_profile"),
        "search_mode": config_dict.get("search_mode", "hybrid"),
    }

    pipeline: RAGPipelineV2 = app_state["pipeline"]

    # Return streaming response
    return StreamingResponse(
        pipeline.execute_streaming(query, chat_thread, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )


@app.post("/v2/chat/sync")
async def chat_sync(request: Request):
    """
    Synchronous chat endpoint (non-streaming).

    Returns complete response as JSON.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = body.get("query", "")
    chat_thread = body.get("chatThread", [])
    config_dict = body.get("config", {})

    config: SearchConfigV2 = {
        "chunk_count": config_dict.get("chunk_count", 10),
        "use_semantic_ranker": config_dict.get("use_semantic_ranker", True),
        "use_query_rewrite": config_dict.get("use_query_rewrite", True),
        "use_knowledge_base": config_dict.get("use_knowledge_base", False),
        "use_streaming": False,
        "query_rewrite_count": config_dict.get("query_rewrite_count", 5),
        "scoring_profile": config_dict.get("scoring_profile"),
        "search_mode": config_dict.get("search_mode", "hybrid"),
    }

    pipeline: RAGPipelineV2 = app_state["pipeline"]

    try:
        result = await pipeline.execute(query, chat_thread, config)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Chat sync failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/v2/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/v2/config")
async def get_config():
    """Get current configuration."""
    return {
        "model": os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-5-mini"),
        "search_index": os.environ.get("SEARCH_INDEX_NAME"),
        "knowledge_base": os.environ.get("KNOWLEDGE_BASE_NAME"),
        "features": {
            "knowledge_base_api": app_state.get("knowledge_base_client") is not None,
            "query_rewrite": True,
            "semantic_ranking": True,
            "hybrid_search": True,
        },
    }


# Mount static files for frontend (if running standalone)
static_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
