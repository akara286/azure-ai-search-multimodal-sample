import logging
import json
import time
import uuid
import asyncio
from typing import List, Any
from abc import ABC, abstractmethod
from enum import Enum

from fastapi import Request
from fastapi.responses import StreamingResponse
import instructor
from openai import AsyncAzureOpenAI

from backend.grounding_retriever import GroundingRetriever
from backend.models import (
    AnswerFormat,
    SearchConfig,
    GroundingResult,
    GroundingResults,
)
from backend.processing_step import ProcessingStep

logger = logging.getLogger("rag")


class MessageType(Enum):
    ANSWER = "answer"
    CITATION = "citation"
    LOG = "log"
    ERROR = "error"
    END = "[END]"
    ProcessingStep = "processing_step"
    INFO = "info"


class RagBase(ABC):
    def __init__(
        self,
        openai_client: AsyncAzureOpenAI,
        chatcompletions_model_name: str,
    ):
        self.openai_client = openai_client
        self.chatcompletions_model_name = chatcompletions_model_name

    async def _handle_request(self, request: Request):
        try:
            request_params = await request.json()
        except Exception:
            request_params = {}

        search_text = request_params.get("query", "")
        chat_thread = request_params.get("chatThread", [])
        config_dict = request_params.get("config", {})
        search_config = SearchConfig(
            chunk_count=config_dict.get("chunk_count", 10),
            openai_api_mode=config_dict.get("openai_api_mode", "chat_completions"),
            use_semantic_ranker=config_dict.get("use_semantic_ranker", False),
            use_streaming=config_dict.get("use_streaming", False),
            use_knowledge_agent=config_dict.get("use_knowledge_agent", False),
        )
        request_id = request_params.get("request_id", str(int(time.time())))

        # Queue for streaming response
        queue = asyncio.Queue()

        async def process_task():
            try:
                await self._process_request(
                    request_id, queue, search_text, chat_thread, search_config
                )
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                await self._send_error_message(request_id, queue, str(e))
            finally:
                await self._send_end(queue)
                await queue.put(None)  # Sentinel to stop stream

        # Start processing in background
        asyncio.create_task(process_task())

        async def stream_generator():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
            },
        )

    @abstractmethod
    async def _process_request(
        self,
        request_id: str,
        response: asyncio.Queue,
        search_text: str,
        chat_thread: list,
        search_config: SearchConfig,
    ):
        pass

    async def _formulate_response(
        self,
        request_id: str,
        response: asyncio.Queue,
        messages: list,
        grounding_retriever: GroundingRetriever,
        grounding_results: GroundingResults,
        search_config: SearchConfig,
    ):
        """Handles streaming chat completion and sends citations."""

        logger.info("Formulating LLM response")
        await self._send_processing_step_message(
            request_id,
            response,
            ProcessingStep(title="LLM Payload", type="code", content=messages),
        )

        complete_response: dict = {}

        if search_config.get("use_streaming", False):
            logger.info("Streaming chat completion")
            # Use raw OpenAI streaming to properly accumulate tokens
            stream = await self.openai_client.chat.completions.create(
                stream=True,
                model=self.chatcompletions_model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            msg_id = str(uuid.uuid4())
            accumulated_content = ""
            last_sent_answer = ""

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    accumulated_content += chunk.choices[0].delta.content
                    # Try to extract partial answer from accumulated JSON for streaming UI
                    try:
                        if '"answer"' in accumulated_content:
                            import re

                            # Match answer value, handling escaped quotes
                            match = re.search(
                                r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)',
                                accumulated_content,
                            )
                            if match:
                                partial_answer = (
                                    match.group(1)
                                    .replace('\\"', '"')
                                    .replace("\\n", "\n")
                                )
                                if partial_answer != last_sent_answer:
                                    await self._send_answer_message(
                                        request_id, response, msg_id, partial_answer
                                    )
                                    last_sent_answer = partial_answer
                    except Exception:
                        pass

            # Parse final response
            logger.info(f"Final accumulated content length: {len(accumulated_content)}")
            try:
                parsed = AnswerFormat.model_validate_json(accumulated_content)
                complete_response = parsed.model_dump()
                # Send final complete answer
                await self._send_answer_message(
                    request_id, response, msg_id, parsed.answer
                )
            except Exception as e:
                logger.error(f"Failed to parse with Pydantic: {e}")
                # Fall back to manual JSON parsing
                import json

                try:
                    data = json.loads(accumulated_content)
                    complete_response = {
                        "answer": data.get("answer", ""),
                        "text_citations": data.get(
                            "text_citations", data.get("text_Citations", [])
                        ),
                        "image_citations": data.get(
                            "image_citations", data.get("image_Citations", [])
                        ),
                    }
                    await self._send_answer_message(
                        request_id, response, msg_id, complete_response["answer"]
                    )
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Could not parse JSON response: {accumulated_content[:500]}"
                    )

        else:
            logger.info("Waiting for chat completion")
            # Use raw OpenAI without streaming for consistency
            chat_response = await self.openai_client.chat.completions.create(
                stream=False,
                model=self.chatcompletions_model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            msg_id = str(uuid.uuid4())

            if chat_response.choices and chat_response.choices[0].message.content:
                content = chat_response.choices[0].message.content
                logger.info(f"Non-streaming response length: {len(content)}")
                try:
                    parsed = AnswerFormat.model_validate_json(content)
                    complete_response = parsed.model_dump()
                except Exception as e:
                    logger.error(f"Failed to parse with Pydantic: {e}")
                    import json

                    data = json.loads(content)
                    complete_response = {
                        "answer": data.get("answer", ""),
                        "text_citations": data.get(
                            "text_citations", data.get("text_Citations", [])
                        ),
                        "image_citations": data.get(
                            "image_citations", data.get("image_Citations", [])
                        ),
                    }
                await self._send_answer_message(
                    request_id, response, msg_id, complete_response["answer"]
                )
            else:
                raise ValueError("No response received from chat completion.")

        await self._send_processing_step_message(
            request_id,
            response,
            ProcessingStep(
                title="LLM response", type="code", content=complete_response
            ),
        )

        logger.info(
            f"Extracting citations - text_citations: {complete_response.get('text_citations', [])}, image_citations: {complete_response.get('image_citations', [])}"
        )
        await self._extract_and_send_citations(
            request_id,
            response,
            grounding_retriever,
            grounding_results["references"],
            complete_response.get("text_citations") or [],
            complete_response.get("image_citations") or [],
        )

    async def _extract_and_send_citations(
        self,
        request_id: str,
        response: asyncio.Queue,
        grounding_retriever: GroundingRetriever,
        grounding_results: List[GroundingResult],
        text_citation_ids: list,
        image_citation_ids: list,
    ):
        """Extracts and sends citations from search results."""
        citations = await self.extract_citations(
            grounding_retriever,
            grounding_results,
            text_citation_ids,
            image_citation_ids,
        )

        await self._send_citation_message(
            request_id,
            response,
            request_id,
            citations.get("text_citations", []),
            citations.get("image_citations", []),
        )

    @abstractmethod
    async def extract_citations(
        self,
        grounding_retriever: GroundingRetriever,
        grounding_results: List[GroundingResult],
        text_citation_ids: list,
        image_citation_ids: list,
    ) -> dict:
        pass

    async def _send_error_message(
        self, request_id: str, response: asyncio.Queue, message: str
    ):
        """Sends an error message through the stream."""
        await self._send_message(
            response,
            MessageType.ERROR.value,
            {
                "request_id": request_id,
                "message_id": str(uuid.uuid4()),
                "message": message,
            },
        )

    async def _send_info_message(
        self,
        request_id: str,
        response: asyncio.Queue,
        message: str,
        details: str = None,
    ):
        """Sends an info message through the stream."""
        await self._send_message(
            response,
            MessageType.INFO.value,
            {
                "request_id": request_id,
                "message_id": str(uuid.uuid4()),
                "message": message,
                "details": details,
            },
        )

    async def _send_processing_step_message(
        self,
        request_id: str,
        response: asyncio.Queue,
        processing_step: ProcessingStep,
    ):
        logger.info(
            f"Sending processing step message for step: {processing_step.title}"
        )
        await self._send_message(
            response,
            MessageType.ProcessingStep.value,
            {
                "request_id": request_id,
                "message_id": str(uuid.uuid4()),
                "processingStep": processing_step.to_dict(),
            },
        )

    async def _send_answer_message(
        self,
        request_id: str,
        response: asyncio.Queue,
        message_id: str,
        content: str,
    ):
        await self._send_message(
            response,
            MessageType.ANSWER.value,
            {
                "request_id": request_id,
                "message_id": message_id,
                "role": "assistant",
                "answerPartial": {"answer": content},
            },
        )

    async def _send_citation_message(
        self,
        request_id: str,
        response: asyncio.Queue,
        message_id: str,
        text_citations: list,
        image_citations: list,
    ):

        await self._send_message(
            response,
            MessageType.CITATION.value,
            {
                "request_id": request_id,
                "message_id": message_id,
                "textCitations": text_citations,
                "imageCitations": image_citations,
            },
        )

    async def _send_message(self, response: asyncio.Queue, event, data):
        try:
            await response.put(f"event:{event}\ndata: {json.dumps(data)}\n\n")
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def _send_end(self, response: asyncio.Queue):
        await self._send_message(response, MessageType.END.value, {})

    def attach_to_app(self, app, path):
        """Attaches the handler to the web app."""
        app.add_api_route(path, self._handle_request, methods=["POST"])
