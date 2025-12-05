"""
Hybrid Search Retriever with semantic reranking and query rewrite.

This module provides direct search capabilities for cases where
the Knowledge Base API is not used, implementing:
- Hybrid search (vector + keyword)
- Generative query rewriting
- Semantic reranking
- Scoring profiles
"""

import logging
from typing import List, Optional
from azure.core.credentials import TokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    VectorizableTextQuery,
)

from models import (
    Message,
    RetrievalResult,
    RetrievalResults,
    SearchConfigV2,
    SubQuery,
)
from query_planner import QueryPlanner

logger = logging.getLogger("backend_v2.search_retriever")


class HybridSearchRetriever:
    """
    Hybrid search retriever with modern Azure AI Search features.

    Implements the 2025 Azure AI Search capabilities:
    - Hybrid search (vector + BM25)
    - Generative query rewriting
    - Semantic reranking with scoring profiles
    - Multi-subquery execution
    """

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        credential: TokenCredential,
        query_planner: QueryPlanner,
    ):
        self.endpoint = endpoint
        self.index_name = index_name
        self.credential = credential
        self.query_planner = query_planner

        self._search_client: Optional[SearchClient] = None

    async def initialize(self):
        """Initialize the search client."""
        if self._search_client is None:
            self._search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )

    async def retrieve(
        self,
        user_message: str,
        chat_history: List[Message],
        config: SearchConfigV2,
        subqueries: Optional[List[SubQuery]] = None,
    ) -> RetrievalResults:
        """
        Execute hybrid search with all modern features.

        Args:
            user_message: The user's question
            chat_history: Chat history for context
            config: Search configuration
            subqueries: Pre-computed subqueries (if query planning was done externally)

        Returns:
            RetrievalResults with references and metadata
        """
        await self.initialize()

        # Get subqueries if not provided
        if subqueries is None:
            plan = await self.query_planner.plan_query(
                user_message, chat_history, max_subqueries=4
            )
            subqueries = plan.subqueries

        all_results: List[RetrievalResult] = []
        all_query_rewrites: List[str] = []
        executed_subqueries: List[str] = []

        # Execute each subquery
        for subquery in subqueries:
            results, rewrites = await self._execute_search(
                query=subquery.query,
                config=config,
            )

            # Tag results with source subquery
            for result in results:
                result["source_subquery"] = subquery.query

            all_results.extend(results)
            all_query_rewrites.extend(rewrites)
            executed_subqueries.append(subquery.query)

        # Deduplicate and re-rank combined results
        deduplicated = self._deduplicate_results(all_results)

        # Sort by reranker score if available, otherwise by search score
        deduplicated.sort(
            key=lambda x: x.get("reranker_score") or x.get("score", 0),
            reverse=True,
        )

        # Limit to requested count
        max_results = config.get("chunk_count", 10)
        final_results = deduplicated[:max_results]

        return {
            "references": final_results,
            "subqueries": executed_subqueries,
            "query_rewrites": all_query_rewrites,
            "total_results": len(final_results),
        }

    async def _execute_search(
        self,
        query: str,
        config: SearchConfigV2,
    ) -> tuple[List[RetrievalResult], List[str]]:
        """
        Execute a single search query with all configured features.

        Args:
            query: The search query
            config: Search configuration

        Returns:
            Tuple of (results, query_rewrites)
        """
        # Build search parameters
        search_params = {
            "search_text": query,
            "top": config.get("chunk_count", 10),
            "select": [
                "content_id",
                "content_text",
                "document_title",
                "text_document_id",
                "image_document_id",
                "locationMetadata",
                "content_path",
            ],
        }

        # Add vector query for hybrid search
        vector_query = VectorizableTextQuery(
            text=query,
            fields="content_embedding",
            k_nearest_neighbors=config.get("chunk_count", 10),
        )
        search_params["vector_queries"] = [vector_query]

        # Configure semantic ranking - only if semantic_configuration_name is provided
        # Semantic mode requires a valid semantic configuration in the index
        semantic_config = config.get("semantic_configuration_name")
        use_semantic = config.get("use_semantic_ranker", False) and semantic_config

        if use_semantic:
            search_params["query_type"] = "semantic"
            search_params["semantic_configuration_name"] = semantic_config

            # Add scoring profile if specified
            if config.get("scoring_profile"):
                search_params["scoring_profile"] = config["scoring_profile"]

            # Add generative query rewrite - only works with semantic mode
            if config.get("use_query_rewrite", True):
                search_params["query_rewrites"] = f"generative|count-{config.get('query_rewrite_count', 5)}"
                search_params["query_language"] = "en-US"

        query_rewrites = []

        # Execute the search
        try:
            search_results = await self._search_client.search(**search_params)

            results: List[RetrievalResult] = []
            async for result in search_results:
                retrieval_result = self._parse_search_result(result)
                results.append(retrieval_result)

                # Extract query rewrites from debug info if available
                if "@search.debug" in result:
                    debug = result["@search.debug"]
                    if "queryRewrites" in debug:
                        query_rewrites.extend(debug["queryRewrites"])

            return results, query_rewrites

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            raise

    def _parse_search_result(self, result: dict) -> RetrievalResult:
        """Parse a search result into RetrievalResult format."""
        is_image = result.get("image_document_id") is not None
        is_text = result.get("text_document_id") is not None

        if is_text and result.get("content_text"):
            content_type = "text"
            content = {
                "ref_id": result["content_id"],
                "text": result["content_text"],
            }
        elif is_image and result.get("content_path"):
            content_type = "image"
            content = result["content_path"]
        else:
            content_type = "text"
            content = {"ref_id": result.get("content_id", ""), "text": ""}

        return {
            "ref_id": result["content_id"],
            "content": content,
            "content_type": content_type,
            "score": result.get("@search.score", 0.0),
            "reranker_score": result.get("@search.rerankerScore"),
            "source_subquery": None,
            # Include full result for citation extraction
            "_raw": result,
        }

    def _deduplicate_results(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Deduplicate results by ref_id, keeping the highest-scored version."""
        seen: dict[str, RetrievalResult] = {}

        for result in results:
            ref_id = result["ref_id"]
            if ref_id not in seen:
                seen[ref_id] = result
            else:
                # Keep the one with higher score
                existing_score = seen[ref_id].get("reranker_score") or seen[ref_id].get("score", 0)
                new_score = result.get("reranker_score") or result.get("score", 0)
                if new_score > existing_score:
                    seen[ref_id] = result

        return list(seen.values())

    async def get_document(self, doc_id: str) -> dict:
        """Retrieve a full document by ID."""
        await self.initialize()
        return await self._search_client.get_document(doc_id)

    async def close(self):
        """Close the search client."""
        if self._search_client:
            await self._search_client.close()
