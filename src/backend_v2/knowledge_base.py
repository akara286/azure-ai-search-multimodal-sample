"""
Knowledge Base client for Azure AI Search 2025-11-01-preview API.

This module implements the new Knowledge Base API which replaces the older
Knowledge Agent API, providing agentic retrieval with automatic query
decomposition, subquery generation, and result synthesis.
"""

import asyncio
import logging
import time
from typing import List, Optional
from azure.core.credentials import TokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)

from models import (
    Message,
    RetrievalResult,
    RetrievalResults,
    SearchConfigV2,
)

logger = logging.getLogger("backend_v2.knowledge_base")


class KnowledgeBaseClient:
    """
    Client for Azure AI Search Knowledge Base API (2025-11-01-preview).

    The Knowledge Base API provides:
    - Automatic query decomposition into subqueries
    - LLM-powered query planning
    - Multi-source retrieval
    - Built-in semantic reranking
    - Citation tracking
    """

    def __init__(
        self,
        endpoint: str,
        credential: TokenCredential,
        knowledge_base_name: str,
        index_name: str,
        azure_openai_endpoint: str,
        model_deployment: str,
        model_name: str = "gpt-5-mini",
        semantic_configuration_name: Optional[str] = None,
        knowledge_source_name: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.credential = credential
        self.knowledge_base_name = knowledge_base_name
        self.index_name = index_name
        self.azure_openai_endpoint = azure_openai_endpoint
        self.model_deployment = model_deployment
        self.model_name = model_name
        self.semantic_configuration_name = semantic_configuration_name
        # Use provided source name or generate from index name
        self.knowledge_source_name = knowledge_source_name or f"{index_name}-source"

        self._index_client: Optional[SearchIndexClient] = None
        self._search_client: Optional[SearchClient] = None
        self._kb_initialized = False

    async def initialize(self):
        """Initialize the Knowledge Base and ensure it exists."""
        if self._kb_initialized:
            return

        self._index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential,
        )

        self._search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

        await self._ensure_knowledge_base_exists()
        self._kb_initialized = True
        logger.info(f"Knowledge Base '{self.knowledge_base_name}' initialized")

    async def _ensure_knowledge_base_exists(self):
        """
        Create or update the Knowledge Base configuration.

        Note: The 2025-11-01-preview API uses KnowledgeBase instead of KnowledgeAgent.
        Knowledge sources are now separate reusable objects.

        Schema based on:
        https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-create-knowledge-base
        """
        try:
            # Use the configured knowledge source name
            knowledge_source_name = self.knowledge_source_name

            # Create knowledge source pointing to our index
            # 2025-11-01-preview schema for knowledge sources
            # See: https://learn.microsoft.com/en-us/azure/search/agentic-knowledge-source-how-to-search-index
            # Fields based on actual index schema:
            # content_id, text_document_id, document_title, image_document_id,
            # content_text, content_embedding, content_path
            search_index_params = {
                "searchIndexName": self.index_name,
                "sourceDataFields": [
                    {"name": "id"},
                    {"name": "text_document_id"},
                    {"name": "document_title"},
                    {"name": "image_document_id"},
                    {"name": "content_text"},
                    {"name": "content_path"},
                    {"name": "locationMetadata"},
                ],
                "searchFields": [
                    {"name": "content_text"},
                ],
            }

            # Add semantic configuration if provided
            if self.semantic_configuration_name:
                search_index_params["semanticConfigurationName"] = (
                    self.semantic_configuration_name
                )

            knowledge_source_definition = {
                "name": knowledge_source_name,
                "kind": "searchIndex",  # Required: searchIndex, azureBlob, etc.
                "description": f"Knowledge source for {self.index_name}",
                "searchIndexParameters": search_index_params,
            }

            # Create the knowledge base that references the source
            # 2025-11-01-preview schema for knowledge bases
            # See: https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-create-knowledge-base
            # outputMode: null (default) = raw data extraction, "answerSynthesis" = LLM-generated answers
            knowledge_base_definition = {
                "name": self.knowledge_base_name,
                "description": "Multimodal RAG knowledge base for document Q&A",
                "retrievalInstructions": "Use the user's question directly as the search query. Search for content that answers the user's specific question.",
                "knowledgeSources": [{"name": knowledge_source_name}],
                "models": [
                    {
                        "kind": "azureOpenAI",
                        "azureOpenAIParameters": {
                            "resourceUri": self.azure_openai_endpoint,
                            "deploymentId": self.model_deployment,
                            "modelName": self.model_name,
                        },
                    }
                ],
                "retrievalReasoningEffort": {
                    "kind": "medium"  # Options: minimal, low, medium (high not supported)
                },
            }

            # Use the REST API directly for Knowledge Base operations
            # as the SDK may not have full support yet
            await self._create_or_update_knowledge_base(
                knowledge_source_definition, knowledge_base_definition
            )

        except Exception as e:
            logger.error(f"Failed to ensure Knowledge Base exists: {e}")
            raise

    async def _create_or_update_knowledge_base(
        self, knowledge_source: dict, knowledge_base: dict
    ):
        """
        Create or update Knowledge Base via REST API.

        This uses the 2025-11-01-preview API endpoints directly.
        """
        import httpx
        from azure.core.credentials import AccessToken

        # Get access token
        token: AccessToken = self.credential.get_token(
            "https://search.azure.com/.default"
        )

        headers = {
            "Authorization": f"Bearer {token.token}",
            "Content-Type": "application/json",
            "api-version": "2025-11-01-preview",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create/update knowledge source
            source_url = f"{self.endpoint}/knowledgesources/{knowledge_source['name']}?api-version=2025-11-01-preview"
            resp = await client.put(source_url, json=knowledge_source, headers=headers)
            if resp.status_code not in [200, 201]:
                logger.warning(
                    f"Knowledge source creation response: {resp.status_code} - {resp.text}"
                )

            # Create/update knowledge base
            kb_url = f"{self.endpoint}/knowledgebases/{knowledge_base['name']}?api-version=2025-11-01-preview"
            resp = await client.put(kb_url, json=knowledge_base, headers=headers)
            if resp.status_code not in [200, 201]:
                logger.warning(
                    f"Knowledge base creation response: {resp.status_code} - {resp.text}"
                )

    async def retrieve(
        self,
        user_message: str,
        chat_history: List[Message],
        config: SearchConfigV2,
    ) -> RetrievalResults:
        """
        Execute agentic retrieval using the Knowledge Base.

        The Knowledge Base automatically:
        1. Analyzes the query and chat history
        2. Decomposes complex queries into subqueries
        3. Executes subqueries with query rewriting
        4. Applies semantic reranking
        5. Returns consolidated results with citations

        Schema based on:
        https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-retrieve

        Args:
            user_message: The current user query
            chat_history: Previous conversation messages
            config: Search configuration options

        Returns:
            RetrievalResults with references, subqueries, and metadata
        """
        if not self._kb_initialized:
            await self.initialize()

        # Build messages array in the correct format
        messages = []
        for msg in chat_history:
            messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", []),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}],
            }
        )

        # Use configured knowledge source name
        knowledge_source_name = self.knowledge_source_name

        # Build the retrieval request per 2025-11-01-preview schema
        # See: https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-retrieve
        # Note: outputMode is omitted to use default (raw data extraction)
        retrieval_request = {
            "messages": messages,
            "includeActivity": True,  # Include activity log for debugging/timing
            "knowledgeSourceParams": [
                {
                    "knowledgeSourceName": knowledge_source_name,
                    "kind": "searchIndex",
                    "includeReferences": True,
                    "includeReferenceSourceData": True,
                }
            ],
        }

        # Add reasoning effort setting - default to medium (max supported)
        reasoning_effort = config.get("reasoning_effort", "medium")
        # Map 'high' to 'medium' effectively if passed
        if reasoning_effort == "high":
            reasoning_effort = "medium"

        if reasoning_effort in ["low", "medium"]:
            retrieval_request["retrievalReasoningEffort"] = {"kind": reasoning_effort}

        # Add optional output size limit
        if config.get("max_output_size"):
            retrieval_request["maxOutputSize"] = config["max_output_size"]

        # Execute retrieval
        from azure.core.credentials import AccessToken
        import httpx

        token: AccessToken = self.credential.get_token(
            "https://search.azure.com/.default"
        )

        headers = {
            "Authorization": f"Bearer {token.token}",
            "Content-Type": "application/json",
        }

        retrieve_url = f"{self.endpoint}/knowledgebases/{self.knowledge_base_name}/retrieve?api-version=2025-11-01-preview"

        # Set timeout for KB API calls (90 seconds)
        logger.info(f"Sending retrieval request to: {retrieve_url}")
        logger.debug(f"Retrieval request body: {retrieval_request}")

        kb_start = time.time()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
                resp = await client.post(
                    retrieve_url, json=retrieval_request, headers=headers
                )

                if resp.status_code not in [200, 206]:  # 206 = Partial Content
                    raise Exception(
                        f"Knowledge Base retrieval failed: {resp.status_code} - {resp.text}"
                    )

                result = resp.json()
                kb_time = (time.time() - kb_start) * 1000
                logger.info(f"[TIMING] Knowledge Base API call: {kb_time:.0f}ms")
                logger.info(f"Knowledge Base response status: {resp.status_code}")
                logger.info(
                    f"Knowledge Base response keys: {result.keys() if result else 'None'}"
                )
                logger.info(
                    f"Knowledge Base references count: {len(result.get('references', []))}"
                )

                # Log detailed timing from activity log
                for activity in result.get("activity", []):
                    activity_type = activity.get("type", "")
                    elapsed = activity.get("elapsedMs", 0)
                    if activity_type == "modelQueryPlanning":
                        logger.info(
                            f"[TIMING] KB - Query Planning: {elapsed}ms (in:{activity.get('inputTokens', 0)} out:{activity.get('outputTokens', 0)})"
                        )
                    elif activity_type == "searchIndex":
                        logger.info(
                            f"[TIMING] KB - Search Index: {elapsed}ms (count:{activity.get('count', 0)})"
                        )
                    elif activity_type == "agenticReasoning":
                        logger.info(
                            f"[TIMING] KB - Reasoning: {elapsed}ms (tokens:{activity.get('reasoningTokens', 0)})"
                        )

        except httpx.TimeoutException:
            kb_time = (time.time() - kb_start) * 1000
            logger.error(f"Knowledge Base API call timed out after {kb_time:.0f}ms")
            raise Exception(f"Knowledge Base retrieval timed out after {kb_time/1000:.1f}s")

        # Parse the response
        return self._parse_retrieval_response(result)

    def _parse_retrieval_response(self, response: dict) -> RetrievalResults:
        """
        Parse the Knowledge Base retrieval response into our format.

        Response structure per 2025-11-01-preview:
        {
            "response": [{"role": "assistant", "content": [{"type": "text", "text": "..."}]}],
            "references": [{"type": "AzureSearchDoc", "id": "...", "docKey": "...", "sourceData": {...}}],
            "activity": [{"type": "modelQueryPlanning|searchIndex|...", ...}]
        }
        """
        import json

        references: List[RetrievalResult] = []
        subqueries: List[str] = []
        query_rewrites: List[str] = []

        # Extract activity log for subqueries and activity info
        for activity in response.get("activity", []):
            activity_type = activity.get("type", "")
            if activity_type == "modelQueryPlanning":
                # Query planning activity
                logger.debug(f"Query planning: {activity.get('elapsedMs')}ms")
            elif activity_type == "searchIndex":
                # Search activity
                logger.debug(
                    f"Search executed: {activity.get('count')} results in {activity.get('elapsedMs')}ms"
                )
            elif activity_type == "agenticReasoning":
                # Reasoning activity
                logger.debug(f"Reasoning: {activity.get('elapsedMs')}ms")

        # Build a lookup map from references - the references array contains the actual document data
        reference_map = {}
        for ref in response.get("references", []):
            ref_id = str(ref.get("id", ""))
            doc_key = ref.get("docKey", ref_id)
            source_data = ref.get("sourceData", {})
            reference_map[ref_id] = {
                "doc_key": doc_key,
                "source_data": source_data,
                "activity_source": ref.get("activitySource"),
            }
            logger.debug(f"Reference {ref_id}: docKey={doc_key}, sourceData keys={list(source_data.keys()) if source_data else 'None'}")

        # Extract content from response
        response_had_content = False
        for resp_item in response.get("response", []):
            for content in resp_item.get("content", []):
                if content.get("type") == "text":
                    text_content = content.get("text", "")
                    logger.debug(f"Response text content (first 200 chars): {text_content[:200] if text_content else 'EMPTY'}")

                    if not text_content or not text_content.strip():
                        logger.warning("Knowledge Base response content is empty")
                        continue

                    response_had_content = True

                    # Try to parse as JSON array of results
                    try:
                        parsed_content = json.loads(text_content)
                        if isinstance(parsed_content, list):
                            for item in parsed_content:
                                ref = self._build_reference_from_item(
                                    item, reference_map
                                )
                                if ref:
                                    references.append(ref)
                        elif isinstance(parsed_content, dict):
                            ref = self._build_reference_from_item(
                                parsed_content, reference_map
                            )
                            if ref:
                                references.append(ref)
                    except json.JSONDecodeError:
                        # Plain text response, create a single reference
                        references.append(
                            {
                                "ref_id": "response",
                                "content": {"text": text_content},
                                "content_type": "text",
                                "score": 1.0,
                                "reranker_score": None,
                                "source_subquery": None,
                            }
                        )

        # FALLBACK: If response content is empty but we have references, use sourceData directly
        # This handles cases where the KB API returns references but empty response content
        if not response_had_content and reference_map:
            logger.warning(f"No response content but {len(reference_map)} references found. Using sourceData directly.")
            for ref_id, ref_info in reference_map.items():
                source_data = ref_info.get("source_data", {})
                doc_key = ref_info.get("doc_key", ref_id)

                # Extract text from sourceData
                content_text = source_data.get("content_text", "")
                if not content_text:
                    # Try alternative field names
                    content_text = source_data.get("text", source_data.get("content", ""))

                if content_text:
                    references.append({
                        "ref_id": doc_key,
                        "content": {
                            "text": content_text,
                            "title": source_data.get("document_title", ""),
                        },
                        "content_type": "text",
                        "score": 1.0,
                        "reranker_score": None,
                        "source_subquery": None,
                        "_raw": source_data,
                    })
                    logger.info(f"Added reference from sourceData: {doc_key[:50]}... with {len(content_text)} chars")

        logger.info(f"Parsed {len(references)} references from Knowledge Base response")
        return {
            "references": references,
            "subqueries": subqueries,
            "query_rewrites": query_rewrites,
            "total_results": len(references),
        }

    def _build_reference_from_item(
        self, item: dict, reference_map: dict
    ) -> Optional[RetrievalResult]:
        """Build a RetrievalResult from a parsed response item."""
        ref_id = item.get("ref_id", "")

        # Look up additional info from reference map
        ref_info = reference_map.get(str(ref_id), {})
        doc_key = ref_info.get("doc_key", ref_id) or ref_id

        return {
            "ref_id": doc_key,
            "content": {
                "text": item.get("content", item.get("text", "")),
                "title": item.get("title", ""),
                "terms": item.get("terms", []),
            },
            "content_type": "text",
            "score": item.get("@search.score", 1.0),
            "reranker_score": item.get("@search.rerankerScore"),
            "source_subquery": None,
            "_raw": ref_info.get("source_data", {}),
        }

    async def get_document(self, doc_id: str) -> dict:
        """Retrieve a full document by ID for citation purposes."""
        if not self._search_client:
            await self.initialize()
        return await self._search_client.get_document(doc_id)

    async def close(self):
        """Close the client connections."""
        if self._index_client:
            await self._index_client.close()
        if self._search_client:
            await self._search_client.close()
