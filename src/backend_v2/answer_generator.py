"""
Answer Generator using GPT-5-mini.

Generates grounded answers with citations from retrieved content,
supporting both streaming and non-streaming modes.
"""

import json
import logging
import re
from typing import List, Optional, AsyncIterator
from openai import AsyncAzureOpenAI

from models import (
    AnswerFormat,
    Message,
    RetrievalResult,
    RetrievalResults,
)
from prompts import SYSTEM_PROMPT

logger = logging.getLogger("backend_v2.answer_generator")


class AnswerGenerator:
    """
    Answer generator using GPT-5-mini with grounded citations.

    Features:
    - Multimodal context (text + images)
    - Structured JSON output with citations
    - Streaming support for real-time responses
    - Citation verification
    """

    def __init__(
        self,
        openai_client: AsyncAzureOpenAI,
        model_name: str = "gpt-5-mini",
        blob_helper=None,
    ):
        self.openai_client = openai_client
        self.model_name = model_name
        self.blob_helper = blob_helper  # For fetching images from blob storage

    async def generate(
        self,
        user_message: str,
        chat_history: List[Message],
        retrieval_results: RetrievalResults,
        streaming: bool = False,
    ) -> AnswerFormat:
        """
        Generate an answer from retrieved content.

        Args:
            user_message: The user's question
            chat_history: Previous conversation messages
            retrieval_results: Retrieved documents and images
            streaming: Whether to stream the response

        Returns:
            AnswerFormat with answer and citations
        """
        # Prepare LLM messages with grounding context
        messages = await self._prepare_messages(
            user_message, chat_history, retrieval_results
        )

        if streaming:
            return await self._generate_streaming(messages)
        else:
            return await self._generate_non_streaming(messages)

    async def generate_streaming(
        self,
        user_message: str,
        chat_history: List[Message],
        retrieval_results: RetrievalResults,
    ) -> AsyncIterator[tuple[str, Optional[AnswerFormat]]]:
        """
        Generate an answer with streaming, yielding partial results.

        Yields:
            Tuples of (partial_answer, final_response)
            - partial_answer: The current accumulated answer text
            - final_response: None until complete, then the full AnswerFormat
        """
        messages = await self._prepare_messages(
            user_message, chat_history, retrieval_results
        )

        stream = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            stream=True,
            max_completion_tokens=4000,
        )

        accumulated_content = ""
        last_sent_answer = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                accumulated_content += chunk.choices[0].delta.content

                # Try to extract partial answer for streaming UI
                partial_answer = self._extract_partial_answer(accumulated_content)
                if partial_answer and partial_answer != last_sent_answer:
                    last_sent_answer = partial_answer
                    yield (partial_answer, None)

        # Parse final response
        final_response = self._parse_response(accumulated_content)
        yield (final_response.answer, final_response)

    async def _generate_streaming(self, messages: List[dict]) -> AnswerFormat:
        """Internal streaming generation that accumulates the full response."""
        stream = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            stream=True,
            max_completion_tokens=4000,
        )

        accumulated_content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                accumulated_content += chunk.choices[0].delta.content

        return self._parse_response(accumulated_content)

    async def _generate_non_streaming(self, messages: List[dict]) -> AnswerFormat:
        """Non-streaming generation."""
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
            max_completion_tokens=4000,
        )

        content = response.choices[0].message.content
        return self._parse_response(content)

    async def _prepare_messages(
        self,
        user_message: str,
        chat_history: List[Message],
        retrieval_results: RetrievalResults,
    ) -> List[dict]:
        """
        Prepare messages for the LLM including grounding context.

        Formats retrieved documents and images into the context.
        """
        # Build document context
        document_context = []

        for ref in retrieval_results["references"]:
            if ref["content_type"] == "text":
                document_context.append({
                    "type": "text",
                    "text": json.dumps({
                        "ref_id": ref["ref_id"],
                        "content": ref["content"].get("text", str(ref["content"])),
                    }),
                })

            elif ref["content_type"] == "image":
                # Add image reference
                document_context.append({
                    "type": "text",
                    "text": f"The image below has the ID: [{ref['ref_id']}]",
                })

                # Fetch and add the actual image if blob helper is available
                if self.blob_helper:
                    try:
                        image_base64 = await self.blob_helper.get_image_base64(
                            ref["content"]
                        )
                        if image_base64:
                            document_context.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            })
                    except Exception as e:
                        logger.warning(f"Failed to fetch image {ref['ref_id']}: {e}")

        # Build the message list
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            *chat_history,
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}],
            },
        ]

        # Add grounding documents as a separate user message
        if document_context:
            messages.append({
                "role": "user",
                "content": document_context,
            })

        return messages

    def _extract_partial_answer(self, accumulated: str) -> Optional[str]:
        """Extract partial answer from accumulated JSON for streaming."""
        try:
            if '"answer"' in accumulated:
                # Match answer value, handling escaped quotes
                match = re.search(
                    r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)',
                    accumulated,
                )
                if match:
                    partial = (
                        match.group(1)
                        .replace('\\"', '"')
                        .replace("\\n", "\n")
                    )
                    return partial
        except Exception:
            pass
        return None

    def _parse_response(self, content: str) -> AnswerFormat:
        """Parse the LLM response into AnswerFormat."""
        try:
            # Try Pydantic validation first
            return AnswerFormat.model_validate_json(content)
        except Exception as e:
            logger.warning(f"Pydantic parsing failed: {e}, trying manual parse")

            try:
                data = json.loads(content)
                return AnswerFormat(
                    answer=data.get("answer", ""),
                    text_citations=data.get(
                        "text_citations",
                        data.get("text_Citations", [])
                    ),
                    image_citations=data.get(
                        "image_citations",
                        data.get("image_Citations", [])
                    ),
                )
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing failed: {je}")
                raise ValueError(f"Could not parse response: {content[:500]}")

    def verify_citations(
        self,
        answer: AnswerFormat,
        retrieval_results: RetrievalResults,
    ) -> dict:
        """
        Verify that citations in the answer match retrieved content.

        Returns:
            Dict with verification results and any invalid citations
        """
        available_refs = {
            ref["ref_id"] for ref in retrieval_results["references"]
        }

        all_citations = set(answer.text_citations + answer.image_citations)
        valid_citations = all_citations & available_refs
        invalid_citations = all_citations - available_refs

        return {
            "valid": len(invalid_citations) == 0,
            "valid_citations": list(valid_citations),
            "invalid_citations": list(invalid_citations),
            "coverage": len(valid_citations) / len(all_citations) if all_citations else 1.0,
        }
