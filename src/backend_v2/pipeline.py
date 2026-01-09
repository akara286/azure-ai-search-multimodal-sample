"""
RAG Pipeline Orchestrator - V2 (Knowledge Base only)

Orchestrates the complete RAG pipeline:
1. Retrieval via Azure AI Search Knowledge Base API
2. Answer Generation (gpt-oss-120b by default)
3. Citation Extraction
"""

import asyncio
import json
import logging
import time
import uuid
from typing import List, Optional, AsyncIterator
from models import (
    AnswerFormat,
    Message,
    RetrievalResults,
    SearchConfigV2,
)
from knowledge_base import KnowledgeBaseClient
from answer_generator import AnswerGenerator
from citation_handler import CitationHandler
from prompts import NO_GROUNDING_RESPONSE

logger = logging.getLogger("backend_v2.pipeline")


class RAGPipelineV2:
    """
    Modernized RAG Pipeline with Azure AI Search 2025 features.

    Pipeline stages:
    1. Retrieval (Knowledge Base API with agentic decomposition)
    2. Answer Generation with Citations
    3. Response Formatting

    All stages emit observability events for frontend display.
    """

    def __init__(
        self,
        knowledge_base_client: KnowledgeBaseClient,
        answer_generator: AnswerGenerator,
        citation_handler: CitationHandler,
    ):
        self.knowledge_base_client = knowledge_base_client
        self.answer_generator = answer_generator
        self.citation_handler = citation_handler

    async def execute(
        self,
        user_message: str,
        chat_history: List[Message],
        config: SearchConfigV2,
        event_queue: Optional[asyncio.Queue] = None,
        streaming_msg_id: Optional[str] = None,
    ) -> dict:
        """
        Execute the full RAG pipeline.

        Args:
            user_message: User's question
            chat_history: Previous conversation
            config: Search and generation configuration
            event_queue: Optional queue for streaming events
            streaming_msg_id: Message ID for streaming (to maintain consistency)

        Returns:
            Dict with answer, citations, and pipeline metadata
        """
        request_id = str(uuid.uuid4())
        msg_id = streaming_msg_id or str(uuid.uuid4())
        pipeline_start = time.time()
        timings: dict[str, float] = {}

        try:
            # Stage 1: Query planning is handled internally by the Knowledge Base API.
            await self._emit_step(
                event_queue,
                request_id,
                "query_planning",
                "completed",
                metadata={
                    "subqueries": [user_message],
                    "reasoning": "Delegated to Knowledge Base API",
                    "duration_ms": 0,
                },
            )
            timings["query_planning_ms"] = 0
            logger.info("[TIMING] Query planning: 0ms (delegated to KB)")

            # Stage 2: Retrieval (Knowledge Base only)
            stage_start = time.time()
            retrieval_mode = "knowledge_base"
            mode_description = "Using Knowledge Base API (agentic retrieval with automatic query decomposition)"
            await self._emit_step(
                event_queue,
                request_id,
                "retrieval",
                "running",
                metadata={"mode": retrieval_mode, "description": mode_description},
            )

            try:
                retrieval_results = await self.knowledge_base_client.retrieve(
                    user_message, chat_history, config
                )
            except Exception as e:
                error_msg = f"Knowledge Base retrieval failed: {str(e)}"
                logger.error(error_msg)
                await self._emit_error(event_queue, request_id, error_msg)
                raise ValueError(error_msg)

            timings["retrieval_ms"] = (time.time() - stage_start) * 1000
            logger.info(
                f"[TIMING] Retrieval (knowledge_base): {timings['retrieval_ms']:.0f}ms"
            )

            planned_subqueries = retrieval_results.get("subqueries") or [user_message]

            await self._emit_step(
                event_queue,
                request_id,
                "retrieval",
                "completed",
                metadata={
                    "mode": retrieval_mode,
                    "mode_label": "Knowledge Base",
                    "results_count": retrieval_results["total_results"],
                    "subqueries_executed": planned_subqueries,
                    "query_rewrites": retrieval_results.get("query_rewrites", []),
                    "duration_ms": timings["retrieval_ms"],
                },
            )

            # GROUNDING CHECK: If no documents retrieved, return no-grounding response
            # This prevents the LLM from hallucinating answers without evidence
            if retrieval_results["total_results"] == 0 or not retrieval_results.get("references"):
                logger.warning(f"No grounding documents found for query: {user_message[:100]}...")
                await self._emit_step(
                    event_queue, request_id, "grounding_check", "failed",
                    metadata={"reason": "No relevant documents found in knowledge base"}
                )

                pipeline_duration = (time.time() - pipeline_start) * 1000
                return {
                    "request_id": request_id,
                    "message_id": msg_id,
                    "answer": NO_GROUNDING_RESPONSE,
                    "text_citations": [],
                    "image_citations": [],
                    "retrieval_mode": retrieval_mode,
                    "retrieval_mode_label": "Knowledge Base",
                    "metadata": {
                        "pipeline_duration_ms": pipeline_duration,
                        "retrieval_mode": retrieval_mode,
                        "retrieval_mode_label": "Knowledge Base",
                        "subqueries": planned_subqueries,
                        "query_rewrites": retrieval_results.get("query_rewrites", []),
                        "results_retrieved": 0,
                        "no_grounding": True,
                        "no_grounding_reason": "No relevant documents found in knowledge base",
                    },
                }

            # Stage 3: Answer Generation
            stage_start = time.time()
            await self._emit_step(event_queue, request_id, "generation", "running")

            # Note: We always use non-streaming for LLM generation to ensure complete JSON
            # output with citations. The SSE streaming to frontend still works - we just
            # don't stream partial LLM tokens. This is more reliable because streaming
            # JSON generation can get truncated, losing citations.
            answer = await self.answer_generator.generate(
                user_message, chat_history, retrieval_results, streaming=False
            )

            timings["answer_generation_ms"] = (time.time() - stage_start) * 1000
            logger.info(f"[TIMING] Answer generation: {timings['answer_generation_ms']:.0f}ms")

            await self._emit_step(
                event_queue, request_id, "generation", "completed",
                metadata={
                    "answer_length": len(answer.answer),
                    "duration_ms": timings["answer_generation_ms"],
                }
            )

            # Stage 4: Citation Extraction
            stage_start = time.time()
            await self._emit_step(event_queue, request_id, "citation_extraction", "running")

            citations = await self.citation_handler.extract_citations(
                answer, retrieval_results
            )

            # Verify citations
            verification = self.answer_generator.verify_citations(
                answer, retrieval_results
            )

            timings["citation_extraction_ms"] = (time.time() - stage_start) * 1000
            logger.info(f"[TIMING] Citation extraction: {timings['citation_extraction_ms']:.0f}ms")

            await self._emit_step(
                event_queue, request_id, "citation_extraction", "completed",
                metadata={
                    "text_citations_count": len(citations["text_citations"]),
                    "image_citations_count": len(citations["image_citations"]),
                    "verification": verification,
                    "duration_ms": timings["citation_extraction_ms"],
                }
            )

            pipeline_duration = (time.time() - pipeline_start) * 1000
            timings["total_pipeline_ms"] = pipeline_duration

            # Log timing summary
            logger.info(f"[TIMING] === Pipeline Summary ===")
            logger.info(f"[TIMING]   Query Planning: {timings.get('query_planning_ms', 0):.0f}ms")
            logger.info(f"[TIMING]   Retrieval: {timings.get('retrieval_ms', 0):.0f}ms")
            logger.info(f"[TIMING]   Answer Generation: {timings.get('answer_generation_ms', 0):.0f}ms")
            logger.info(f"[TIMING]   Citation Extraction: {timings.get('citation_extraction_ms', 0):.0f}ms")
            logger.info(f"[TIMING]   TOTAL: {pipeline_duration:.0f}ms")

            return {
                "request_id": request_id,
                "message_id": msg_id,
                "answer": answer.answer,
                "text_citations": citations["text_citations"],
                "image_citations": citations["image_citations"],
                "retrieval_mode": retrieval_mode,
                "retrieval_mode_label": "Knowledge Base",
                "metadata": {
                    "pipeline_duration_ms": pipeline_duration,
                    "timings": timings,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_mode_label": "Knowledge Base",
                    "subqueries": planned_subqueries,
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
        Includes periodic heartbeat messages to keep the connection alive.
        """
        event_queue = asyncio.Queue()
        # Create a consistent message ID for the entire response
        streaming_msg_id = str(uuid.uuid4())
        pipeline_done = asyncio.Event()

        # Start pipeline in background
        async def run_pipeline():
            try:
                result = await self.execute(
                    user_message, chat_history, config, event_queue, streaming_msg_id
                )
                await event_queue.put(("result", result))
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                await event_queue.put(("error", str(e)))
            finally:
                await event_queue.put(None)
                pipeline_done.set()

        asyncio.create_task(run_pipeline())

        # Yield events as SSE with heartbeat to keep connection alive
        heartbeat_interval = 15  # seconds

        while True:
            try:
                # Use timeout to allow periodic heartbeats
                event = await asyncio.wait_for(event_queue.get(), timeout=heartbeat_interval)

                if event is None:
                    yield self._format_sse("[END]", {})
                    break

                event_type, event_data = event

                # Transform result into answer format frontend expects
                if event_type == "result":
                    # Use the same msg_id from the pipeline result
                    answer_event = {
                        "request_id": event_data["request_id"],
                        "message_id": event_data["message_id"],
                        "type": "answer",
                        "role": "assistant",
                        "answerPartial": {"answer": event_data["answer"]},
                        "textCitations": event_data.get("text_citations", []),
                        "imageCitations": event_data.get("image_citations", []),
                        "retrievalMode": event_data.get("retrieval_mode"),
                        "retrievalModeLabel": event_data.get("retrieval_mode_label"),
                    }
                    yield self._format_sse("answer", answer_event)
                elif event_type == "error":
                    # Emit error event to frontend
                    yield self._format_sse("error", {"message": event_data})
                else:
                    yield self._format_sse(event_type, event_data)

            except asyncio.TimeoutError:
                # Send heartbeat comment to keep connection alive
                yield ": heartbeat\n\n"

                # Check if pipeline has finished (safety check)
                if pipeline_done.is_set():
                    yield self._format_sse("[END]", {})
                    break

    async def _generate_streaming(
        self,
        user_message: str,
        chat_history: List[Message],
        retrieval_results: RetrievalResults,
        event_queue: asyncio.Queue,
        request_id: str,
        msg_id: str,
    ) -> AnswerFormat:
        """Generate answer with streaming, emitting partial results."""
        final_answer = None

        async for partial, final in self.answer_generator.generate_streaming(
            user_message, chat_history, retrieval_results
        ):
            if final is not None:
                final_answer = final
            else:
                await event_queue.put((
                    "answer",
                    {
                        "request_id": request_id,
                        "message_id": msg_id,
                        "role": "assistant",
                        "answerPartial": {"answer": partial},
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
        """Emit a pipeline step event in format frontend expects."""
        if queue is None:
            return

        # Format matching frontend's ProcessingStepsMessage
        step_titles = {
            "query_planning": "Query Planning",
            "retrieval": "Document Retrieval",
            "generation": "Answer Generation",
            "citation_extraction": "Citation Extraction",
        }

        step_descriptions = {
            "running": f"Processing {step_name}...",
            "completed": f"Completed {step_name}",
        }

        content = ""
        if metadata:
            # Show mode description prominently for retrieval step
            if "description" in metadata:
                content = metadata["description"]
            elif "subqueries" in metadata:
                content = f"Subqueries: {', '.join(metadata['subqueries'])}"
            elif "results_count" in metadata:
                mode_label = metadata.get("mode_label", "")
                mode_prefix = f"[{mode_label}] " if mode_label else ""
                content = f"{mode_prefix}Retrieved {metadata['results_count']} results"
            elif "answer_length" in metadata:
                content = f"Generated answer ({metadata['answer_length']} chars)"

        await queue.put((
            "processing_step",
            {
                "request_id": request_id,
                "message_id": str(uuid.uuid4()),
                "processingStep": {
                    "title": step_titles.get(step_name, step_name),
                    "description": step_descriptions.get(status, status),
                    "type": "text",
                    "content": content,
                }
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

    def _is_simple_query(self, user_message: str) -> bool:
        """
        Determine if a query is simple enough to skip LLM-based decomposition.

        Simple queries can be searched directly, saving ~1-2s LLM overhead.
        """
        message_lower = user_message.lower()
        word_count = len(user_message.split())

        # Indicators of complex queries that need decomposition
        complexity_indicators = [
            " and " in message_lower,
            " or " in message_lower,
            " compare " in message_lower,
            " difference " in message_lower,
            " vs " in message_lower,
            " versus " in message_lower,
            user_message.count("?") > 1,  # Multiple questions
            word_count > 25,  # Long queries
            ", " in user_message and word_count > 15,  # Lists in longer queries
        ]

        # If 2+ complexity indicators, definitely complex
        if sum(complexity_indicators) >= 2:
            return False

        # Very short queries are always simple
        if word_count <= 8:
            return True

        # Medium length with no complexity indicators = simple
        if sum(complexity_indicators) == 0 and word_count <= 20:
            return True

        return False
