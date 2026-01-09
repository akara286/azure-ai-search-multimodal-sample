"""
FastAPI Application - V2 Modernized RAG Backend

Entry point for the modernized multimodal RAG API using:
- Azure AI Search Knowledge Base API (2025-11-01-preview)
- gpt-5-mini for Knowledge Base retrieval (default)
- gpt-oss-120b for answer generation (default)
"""

import os
import logging
import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Request ID context variable for distributed tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")


class RequestIDFilter(logging.Filter):
    """Logging filter that adds request_id to log records."""

    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


from azure.identity import DefaultAzureCredential
from azure.identity.aio import get_bearer_token_provider
from azure.storage.blob.aio import BlobServiceClient
from openai import AsyncAzureOpenAI

from models import SearchConfigV2
from knowledge_base import KnowledgeBaseClient
from answer_generator import AnswerGenerator
from citation_handler import CitationHandler, BlobHelper
from pipeline import RAGPipelineV2

# Configure logging with request ID
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s",
)

# Add request ID filter to all handlers
for handler in logging.root.handlers:
    handler.addFilter(RequestIDFilter())

logger = logging.getLogger("backend_v2.app")


# Request validation models
class ChatConfigRequest(BaseModel):
    """Configuration options for chat requests."""

    chunk_count: int = Field(
        default=10, ge=1, le=100, description="Number of chunks to retrieve"
    )
    use_semantic_ranker: bool = Field(
        default=True, description="Enable semantic ranking"
    )
    use_query_rewrite: bool = Field(default=True, description="Enable query rewriting")
    use_knowledge_base: bool = Field(default=True, description="Use Knowledge Base API")
    use_knowledge_agent: bool = Field(
        default=False, description="Alias for use_knowledge_base"
    )
    use_streaming: bool = Field(default=True, description="Enable streaming responses")
    query_rewrite_count: int = Field(
        default=5, ge=1, le=10, description="Number of query rewrites"
    )
    scoring_profile: Optional[str] = Field(
        default=None, description="Scoring profile name"
    )
    search_mode: str = Field(default="knowledge_base", description="Search mode")


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""

    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    chatThread: List[dict] = Field(default=[], description="Previous chat messages")
    config: ChatConfigRequest = Field(
        default_factory=ChatConfigRequest, description="Search configuration"
    )


# Rate limiter - configurable via RATE_LIMIT environment variable (default: 30/minute)
rate_limit = os.environ.get("RATE_LIMIT", "30/minute")
limiter = Limiter(key_func=get_remote_address)

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
    # Answer generation uses gpt-oss by default (override via AZURE_OPENAI_* env vars)
    openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-oss-120b")
    model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-oss-120b")

    # Separate model config for Knowledge Base (must be a supported model)
    # Supported: gpt-4o, gpt-4o-mini, gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, gpt-5, gpt-5-mini, gpt-5-nano
    # Retrieval (Knowledge Base API) uses gpt-5-mini by default.
    kb_deployment = os.environ.get("AZURE_OPENAI_KB_DEPLOYMENT", "gpt-5-mini")
    kb_model_name = os.environ.get("AZURE_OPENAI_KB_MODEL_NAME", "gpt-5-mini")

    search_endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
    search_index = os.environ["SEARCH_INDEX_NAME"]
    knowledge_base_name = os.environ.get(
        "KNOWLEDGE_BASE_NAME", os.environ.get("KNOWLEDGE_AGENT_NAME")
    )
    knowledge_source_name = os.environ.get("KNOWLEDGE_SOURCE_NAME")
    semantic_configuration_name = os.environ.get("SEMANTIC_CONFIGURATION_NAME")

    storage_url = os.environ["ARTIFACTS_STORAGE_ACCOUNT_URL"]
    artifacts_container = os.environ["ARTIFACTS_STORAGE_CONTAINER"]

    # Initialize OpenAI client
    openai_client = AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2025-01-01-preview",  # Latest API version for GPT-5
        azure_endpoint=openai_endpoint,
        timeout=60,
    )

    if not knowledge_base_name:
        raise RuntimeError(
            "Knowledge Base is required. Set KNOWLEDGE_BASE_NAME or KNOWLEDGE_AGENT_NAME."
        )

    # Initialize Knowledge Base Client (required)
    knowledge_base_client = KnowledgeBaseClient(
        endpoint=search_endpoint,
        credential=credential,
        knowledge_base_name=knowledge_base_name,
        index_name=search_index,
        azure_openai_endpoint=openai_endpoint,
        model_deployment=kb_deployment,
        model_name=kb_model_name,
        semantic_configuration_name=semantic_configuration_name,
        knowledge_source_name=knowledge_source_name,
    )
    await knowledge_base_client.initialize()
    logger.info(
        f"Knowledge Base '{knowledge_base_name}' initialized with model {kb_model_name}"
    )

    # Initialize Blob Storage
    blob_service_client = BlobServiceClient(
        account_url=storage_url,
        credential=credential,
    )
    artifacts_container_client = blob_service_client.get_container_client(
        artifacts_container
    )

    # Initialize Citation Handler (with KB client for fetching missing locationMetadata)
    citation_handler = CitationHandler(
        blob_service_client=blob_service_client,
        artifacts_container=artifacts_container_client,
        search_client=knowledge_base_client,
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
        knowledge_base_client=knowledge_base_client,
        answer_generator=answer_generator,
        citation_handler=citation_handler,
    )

    # Store in app state
    app_state["pipeline"] = pipeline
    app_state["citation_handler"] = citation_handler
    app_state["blob_service_client"] = blob_service_client
    app_state["knowledge_base_client"] = knowledge_base_client

    logger.info("V2 RAG Backend initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down V2 RAG Backend...")
    await knowledge_base_client.close()
    await blob_service_client.close()


# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG API V2",
    description="Modernized RAG API with Azure AI Search 2025 features",
    version="2.0.0",
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Middleware to add request ID for distributed tracing."""
    # Use existing request ID from header or generate new one
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request_id_var.set(req_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# CORS middleware - restrict origins for security
# Set ALLOWED_ORIGINS environment variable for production (comma-separated)
allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://localhost:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,  # Don't expose credentials
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.post("/v2/chat")
@limiter.limit(rate_limit)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Chat endpoint with streaming SSE response.

    Request body validated by ChatRequest model.
    Rate limited to prevent abuse.
    """
    config: SearchConfigV2 = {
        "chunk_count": chat_request.config.chunk_count,
        "use_semantic_ranker": chat_request.config.use_semantic_ranker,
        "use_query_rewrite": chat_request.config.use_query_rewrite,
        "use_knowledge_base": True,
        "use_streaming": chat_request.config.use_streaming,
        "query_rewrite_count": chat_request.config.query_rewrite_count,
        "scoring_profile": chat_request.config.scoring_profile,
        "search_mode": "knowledge_base",
    }

    pipeline: RAGPipelineV2 = app_state["pipeline"]

    # Return streaming response
    return StreamingResponse(
        pipeline.execute_streaming(chat_request.query, chat_request.chatThread, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )


@app.post("/v2/chat/sync")
@limiter.limit(rate_limit)
async def chat_sync(request: Request, chat_request: ChatRequest):
    """
    Synchronous chat endpoint (non-streaming).

    Returns complete response as JSON.
    Rate limited to prevent abuse.
    """
    config: SearchConfigV2 = {
        "chunk_count": chat_request.config.chunk_count,
        "use_semantic_ranker": chat_request.config.use_semantic_ranker,
        "use_query_rewrite": chat_request.config.use_query_rewrite,
        "use_knowledge_base": True,
        "use_streaming": False,
        "query_rewrite_count": chat_request.config.query_rewrite_count,
        "scoring_profile": chat_request.config.scoring_profile,
        "search_mode": "knowledge_base",
    }

    pipeline: RAGPipelineV2 = app_state["pipeline"]

    try:
        result = await pipeline.execute(
            chat_request.query, chat_request.chatThread, config
        )
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
    kb_name = os.environ.get(
        "KNOWLEDGE_BASE_NAME", os.environ.get("KNOWLEDGE_AGENT_NAME")
    )
    return {
        "model": os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-oss-120b"),
        "search_index": os.environ.get("SEARCH_INDEX_NAME"),
        "knowledge_base": kb_name,
        "features": {
            "knowledge_base_api": True,
            "query_rewrite": True,
            "semantic_ranking": True,
            "hybrid_search": False,
        },
    }


@app.get("/v2/citation/{file_name:path}")
async def get_citation_document(
    file_name: str, page: int = Query(default=1, ge=1, description="Page number")
):
    """
    Get a SAS URL for accessing a citation document (PDF).

    Args:
        file_name: Name/path of the file to retrieve
        page: Optional page number to link to

    Returns:
        JSON with the signed URL for accessing the document
    """
    citation_handler: CitationHandler = app_state.get("citation_handler")

    if not citation_handler:
        return JSONResponse(
            status_code=500,
            content={"error": "Citation handler not initialized"},
        )

    url = await citation_handler.get_document_url(file_name, page_number=page)

    if isinstance(url, dict) and "error" in url:
        return JSONResponse(
            status_code=500,
            content={"error": url["error"]},
        )

    if url:
        return {
            "url": url,
            "fileName": file_name,
            "page": page,
        }
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"Document not found (Generic): {file_name}"},
        )


# Mount static files for frontend (if running standalone)
# Mount static files for frontend (if running standalone)
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
