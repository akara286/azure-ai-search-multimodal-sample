"""
RAG Pipeline Orchestrator - V2

Orchestrates the complete RAG pipeline:
1. Query Planning (GPT-5-mini)
2. Subquery Generation + Query Rewrite
3. Hybrid Search + Semantic Reranking
4. Answer Generation (GPT-5-mini)
5. Citation Extraction
"""

import asyncio
import json
import logging
import time
import uuid
from typing import List, Optional, AsyncIterator
from enum import Enum

from models import (
    AnswerFormat,
    Message,
    PipelineStep,
    RetrievalResults,
    SearchConfigV2,
)
from query_planner import QueryPlanner
from knowledge_base import KnowledgeBaseClient
from search_retriever import HybridSearchRetriever
from answer_generator import AnswerGenerator
from citation_handler import CitationHandler

logger = logging.getLogger("backend_v2.pipeline")


class RetrievalMode(str, Enum):
    """Mode for retrieval - Knowledge Base or Direct Search."""
    KNOWLEDGE_BASE = "knowledge_base"
    HYBRID_SEARCH = "hybrid_search"


class RAGPipelineV2:
    """
    Modernized RAG Pipeline with Azure AI Search 2025 features.

    Pipeline stages:
    1. Query Analysis & Planning
    2. Retrieval (Knowledge Base API or Hybrid Search)
    3. Answer Generation with Citations
    4. Response Formatting

    All stages emit observability events for frontend display.
    """

    def __init__(
        self,
        query_planner: QueryPlanner,
        knowledge_base_client: Optional[KnowledgeBaseClient],
        search_retriever: HybridSearchRetriever,
        answer_generator: AnswerGenerator,
        citation_handler: CitationHandler,
    ):
        self.query_planner = query_planner
        self.knowledge_base_client = knowledge_base_client
        self.search_retriever = search_retriever
        self.answer_generator = answer_generator
        self.citation_handler = citation_handler

    async def execute(
        self,
        user_message: str,
        chat_history: List[Message],
        config: SearchConfigV2,
        event_queue: Optional[asyncio.Queue] = None,
    ) -> dict:
        """
        Execute the full RAG pipeline.

        Args:
            user_message: User's question
            chat_history: Previous conversation
            config: Search and generation configuration
            event_queue: Optional queue for streaming events

        Returns:
            Dict with answer, citations, and pipeline metadata
        """
        request_id = str(uuid.uuid4())
        pipeline_start = time.time()

        try:
            # Stage 1: Query Planning
            await self._emit_step(event_queue, request_id, "query_planning", "running")

            plan = await self.query_planner.plan_query(
                user_message, chat_history, max_subqueries=4
            )

            await self._emit_step(
                event_queue, request_id, "query_planning", "completed",
                metadata={
                    "subqueries": [sq.query for sq in plan.subqueries],
                    "reasoning": plan.reasoning,
                }
            )

            # Stage 2: Retrieval
            await self._emit_step(event_queue, request_id, "retrieval", "running")

            retrieval_mode = (
                RetrievalMode.KNOWLEDGE_BASE
                if config.get("use_knowledge_base", False) and self.knowledge_base_client
                else RetrievalMode.HYBRID_SEARCH
            )

            if retrieval_mode == RetrievalMode.KNOWLEDGE_BASE:
                retrieval_results = await self.knowledge_base_client.retrieve(
                    user_message, chat_history, config
                )
            else:
                retrieval_results = await self.search_retriever.retrieve(
                    user_message, chat_history, config, plan.subqueries
                )

            await self._emit_step(
                event_queue, request_id, "retrieval", "completed",
                metadata={
                    "mode": retrieval_mode.value,
                    "results_count": retrieval_results["total_results"],
                    "subqueries_executed": retrieval_results["subqueries"],
                    "query_rewrites": retrieval_results.get("query_rewrites", []),
                }
            )

            # Stage 3: Answer Generation
            await self._emit_step(event_queue, request_id, "generation", "running")

            if config.get("use_streaming", False) and event_queue:
                # Streaming generation
                answer = await self._generate_streaming(
                    user_message, chat_history, retrieval_results,
                    event_queue, request_id
                )
            else:
                # Non-streaming generation
                answer = await self.answer_generator.generate(
                    user_message, chat_history, retrieval_results, streaming=False
                )

            await self._emit_step(
                event_queue, request_id, "generation", "completed",
                metadata={"answer_length": len(answer.answer)}
            )

            # Stage 4: Citation Extraction
            await self._emit_step(event_queue, request_id, "citation_extraction", "running")

            citations = await self.citation_handler.extract_citations(
                answer, retrieval_results
            )

            # Verify citations
            verification = self.answer_generator.verify_citations(
                answer, retrieval_results
            )

            await self._emit_step(
                event_queue, request_id, "citation_extraction", "completed",
                metadata={
                    "text_citations_count": len(citations["text_citations"]),
                    "image_citations_count": len(citations["image_citations"]),
                    "verification": verification,
                }
            )

            pipeline_duration = (time.time() - pipeline_start) * 1000

            return {
                "request_id": request_id,
                "answer": answer.answer,
                "text_citations": citations["text_citations"],
                "image_citations": citations["image_citations"],
                "metadata": {
                    "pipeline_duration_ms": pipeline_duration,
                    "retrieval_mode": retrieval_mode.value,
                    "subqueries": [sq.query for sq in plan.subqueries],
                    "query_rewrites": retrieval_results.get("query_rewrites", []),
                    "results_retrieved": retrieval_results["total_results"],
                    "citation_verification": verification,
                },
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            await self._emit_error(event_queue, request_id, str(e))
            raise

    async def execute_streaming(
        self,
        user_message: str,
        chat_history: List[Message],
        config: SearchConfigV2,
    ) -> AsyncIterator[str]:
        """
        Execute pipeline with SSE streaming output.

        Yields SSE-formatted events for real-time frontend updates.
        """
        event_queue = asyncio.Queue()

        # Start pipeline in background
        async def run_pipeline():
            try:
                result = await self.execute(
                    user_message, chat_history, config, event_queue
                )
                await event_queue.put(("result", result))
            except Exception as e:
                await event_queue.put(("error", str(e)))
            finally:
                await event_queue.put(None)

        asyncio.create_task(run_pipeline())

        # Yield events as SSE
        while True:
            event = await event_queue.get()
            if event is None:
                yield self._format_sse("end", {})
                break

            event_type, event_data = event
            yield self._format_sse(event_type, event_data)

    async def _generate_streaming(
        self,
        user_message: str,
        chat_history: List[Message],
        retrieval_results: RetrievalResults,
        event_queue: asyncio.Queue,
        request_id: str,
    ) -> AnswerFormat:
        """Generate answer with streaming, emitting partial results."""
        msg_id = str(uuid.uuid4())
        final_answer = None

        async for partial, final in self.answer_generator.generate_streaming(
            user_message, chat_history, retrieval_results
        ):
            if final is not None:
                final_answer = final
            else:
                await event_queue.put((
                    "answer_partial",
                    {
                        "request_id": request_id,
                        "message_id": msg_id,
                        "content": partial,
                    }
                ))

        return final_answer

    async def _emit_step(
        self,
        queue: Optional[asyncio.Queue],
        request_id: str,
        step_name: str,
        status: str,
        metadata: Optional[dict] = None,
    ):
        """Emit a pipeline step event."""
        if queue is None:
            return

        step: PipelineStep = {
            "name": step_name,
            "status": status,
            "duration_ms": None,
            "metadata": metadata,
        }

        await queue.put((
            "processing_step",
            {
                "request_id": request_id,
                "step": step,
            }
        ))

    async def _emit_error(
        self,
        queue: Optional[asyncio.Queue],
        request_id: str,
        error_message: str,
    ):
        """Emit an error event."""
        if queue is None:
            return

        await queue.put((
            "error",
            {
                "request_id": request_id,
                "message": error_message,
            }
        ))

    def _format_sse(self, event_type: str, data: dict) -> str:
        """Format data as SSE event."""
        return f"event:{event_type}\ndata: {json.dumps(data)}\n\n"
