"""
Answer Generator using GPT-5-mini.

Generates grounded answers with citations from retrieved content,
supporting both streaming and non-streaming modes.
"""

import json
import logging
import re
import time
from typing import List, Optional, AsyncIterator
from openai import AsyncAzureOpenAI

from models import (
    AnswerFormat,
    Message,
    RetrievalResult,
    RetrievalResults,
)
from prompts import SYSTEM_PROMPT, NO_GROUNDING_RESPONSE

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
        # GROUNDING CHECK: Defense-in-depth - ensure we have documents before calling LLM
        if (
            not retrieval_results.get("references")
            or len(retrieval_results["references"]) == 0
        ):
            logger.warning("No grounding documents provided to answer generator")
            return AnswerFormat(
                answer=NO_GROUNDING_RESPONSE,
                text_citations=[],
                image_citations=[],
            )

        # Prepare LLM messages with grounding context
        prep_start = time.time()
        messages = await self._prepare_messages(
            user_message, chat_history, retrieval_results
        )
        prep_time = (time.time() - prep_start) * 1000
        logger.info(
            f"[TIMING] Message preparation: {prep_time:.0f}ms (refs: {len(retrieval_results['references'])})"
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
        # GROUNDING CHECK: Defense-in-depth - ensure we have documents before calling LLM
        if (
            not retrieval_results.get("references")
            or len(retrieval_results["references"]) == 0
        ):
            logger.warning(
                "No grounding documents provided to streaming answer generator"
            )
            no_grounding_answer = AnswerFormat(
                answer=NO_GROUNDING_RESPONSE,
                text_citations=[],
                image_citations=[],
            )
            yield (NO_GROUNDING_RESPONSE, no_grounding_answer)
            return

        messages = await self._prepare_messages(
            user_message, chat_history, retrieval_results
        )

        # Define strict JSON schema for structured outputs
        answer_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "text_citations": {"type": "array", "items": {"type": "string"}},
                        "image_citations": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["answer", "text_citations", "image_citations"],
                    "additionalProperties": False
                }
            }
        }

        # gpt-oss-120b requires low temperature to avoid empty responses
        stream = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format=answer_schema,
            stream=True,
            max_completion_tokens=8000,
            reasoning_effort="high",
            temperature=0.0,
            top_p=1.0,
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
        # Define strict JSON schema for structured outputs
        answer_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "text_citations": {"type": "array", "items": {"type": "string"}},
                        "image_citations": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["answer", "text_citations", "image_citations"],
                    "additionalProperties": False
                }
            }
        }
        # gpt-oss-120b requires low temperature to avoid empty responses
        stream = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format=answer_schema,
            stream=True,
            max_completion_tokens=8000,
            reasoning_effort="high",
            temperature=0.0,
            top_p=1.0,
        )

        accumulated_content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                accumulated_content += chunk.choices[0].delta.content

        return self._parse_response(accumulated_content)

    async def _generate_non_streaming(
        self, messages: List[dict], max_retries: int = 2
    ) -> AnswerFormat:
        """Non-streaming generation with retries."""
        from json_repair import repair_json

        # Define strict JSON schema for structured outputs
        # This enforces the exact response format from the model
        answer_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer in Markdown format with inline citations"
                        },
                        "text_citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of text reference IDs used"
                        },
                        "image_citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of image reference IDs used"
                        }
                    },
                    "required": ["answer", "text_citations", "image_citations"],
                    "additionalProperties": False
                }
            }
        }

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                llm_start = time.time()
                # gpt-oss-120b requires low temperature to avoid empty responses
                # See: https://huggingface.co/openai/gpt-oss-120b/discussions/67
                response = await self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=answer_schema,
                    stream=False,
                    max_completion_tokens=8000,
                    reasoning_effort="low",
                    temperature=0.0,
                    top_p=1.0,
                )
                llm_time = (time.time() - llm_start) * 1000

                content = response.choices[0].message.content
                usage = response.usage
                logger.info(
                    f"[TIMING] LLM API call (attempt {attempt+1}): {llm_time:.0f}ms (prompt_tokens: {usage.prompt_tokens}, completion_tokens: {usage.completion_tokens})"
                )

                # Log raw content for debugging empty responses
                if not content or len(content.strip()) == 0:
                    logger.error(f"LLM returned empty content. Response object: {response}")
                else:
                    logger.debug(f"LLM raw content (first 500 chars): {content[:500] if content else 'NONE'}")

                # Check for malformed whitespace-only response before parsing
                # This indicates the model failed to generate proper content
                non_ws = content.replace(" ", "").replace("\n", "").replace("\t", "").replace("{", "").replace("}", "") if content else ""
                if len(non_ws) < 10:
                    logger.warning(f"LLM returned whitespace-only content (attempt {attempt+1}). Will retry.")
                    last_error = Exception("Malformed whitespace-only response")
                    continue  # Retry

                parse_start = time.time()
                result = self._parse_response(content)
                parse_time = (time.time() - parse_start) * 1000
                logger.info(f"[TIMING] Response parsing: {parse_time:.0f}ms, answer length: {len(result.answer)}")

                # If answer is empty or a fallback message, retry
                if not result.answer or len(result.answer.strip()) == 0 or "apologize" in result.answer.lower():
                    logger.warning(f"Parsed answer is empty/fallback (attempt {attempt+1}). Raw content: {content[:500] if content else 'NONE'}")
                    last_error = Exception("Empty or fallback answer")
                    continue  # Retry

                return result

            except Exception as e:
                logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                last_error = e
                # Retry immediately for now, could add backoff if needed

        # If we exhausted retries, return fallback
        logger.error(f"All {max_retries+1} attempts failed. Returning fallback.")
        return AnswerFormat(
            answer="I apologize, but I'm having trouble formatting the response correctly after multiple attempts. Please try again.",
            text_citations=[],
            image_citations=[],
        )

    async def _prepare_messages(
        self,
        user_message: str,
        chat_history: List[Message],
        retrieval_results: RetrievalResults,
    ) -> List[dict]:
        """
        Prepare messages for the LLM including grounding context.

        Formats retrieved documents and images into the context.
        Images are fetched in parallel for better performance.
        """
        import asyncio

        # Separate text and image refs for parallel processing
        text_refs = []
        image_refs = []

        for ref in retrieval_results["references"]:
            if ref["content_type"] == "text":
                text_refs.append(ref)
            elif ref["content_type"] == "image":
                image_refs.append(ref)

        # Fetch all images in parallel if blob helper is available
        image_data = {}
        if self.blob_helper and image_refs:
            image_fetch_start = time.time()

            async def fetch_image(ref):
                try:
                    base64_data = await self.blob_helper.get_image_base64(
                        ref["content"]
                    )
                    return (ref["ref_id"], base64_data)
                except Exception as e:
                    logger.warning(f"Failed to fetch image {ref['ref_id']}: {e}")
                    return (ref["ref_id"], None)

            results = await asyncio.gather(*[fetch_image(ref) for ref in image_refs])
            image_data = {ref_id: data for ref_id, data in results if data}
            image_fetch_time = (time.time() - image_fetch_start) * 1000
            logger.info(
                f"[TIMING] Image fetching: {image_fetch_time:.0f}ms ({len(image_refs)} images, {len(image_data)} successful)"
            )

        # Build document context with pre-fetched images
        document_context = []

        for ref in retrieval_results["references"]:
            if ref["content_type"] == "text":
                document_context.append(
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "ref_id": ref["ref_id"],
                                "content": ref["content"].get(
                                    "text", str(ref["content"])
                                ),
                            }
                        ),
                    }
                )

            elif ref["content_type"] == "image":
                # Add image reference
                document_context.append(
                    {
                        "type": "text",
                        "text": f"The image below has the ID: [{ref['ref_id']}]",
                    }
                )

                # Add pre-fetched image if available
                if ref["ref_id"] in image_data:
                    document_context.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data[ref['ref_id']]}"
                            },
                        }
                    )

        # Build the message list
        # Filter chat_history to remove messages with empty content
        filtered_history = []
        for msg in chat_history:
            content = msg.get("content", [])
            # Check if content is a list with text items
            if isinstance(content, list):
                # Filter out empty text content
                filtered_content = [
                    item
                    for item in content
                    if item.get("type") != "text"
                    or (item.get("text") and item.get("text").strip())
                ]
                if filtered_content:
                    filtered_history.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": filtered_content,
                        }
                    )
            elif isinstance(content, str) and content.strip():
                # String content - keep if not empty
                filtered_history.append(msg)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            *filtered_history,
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}],
            },
        ]

        # Add grounding documents as a separate user message
        if document_context:
            messages.append(
                {
                    "role": "user",
                    "content": document_context,
                }
            )

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
                    partial = match.group(1).replace('\\"', '"').replace("\\n", "\n")
                    return partial
        except Exception:
            pass
        return None

    def _normalize_citation_brackets(self, answer: str) -> str:
        """
        Normalize citation brackets in the answer text.

        Some models (e.g., gpt-oss-120b) use Chinese lenticular brackets【】
        instead of standard square brackets []. The frontend expects [...].
        """
        # Replace Chinese lenticular brackets with standard square brackets
        answer = answer.replace("【", "[").replace("】", "]")
        return answer

    def _parse_response(self, content: str) -> AnswerFormat:
        """Parse the LLM response into AnswerFormat."""
        from json_repair import repair_json

        # Handle None or empty content
        if not content or not content.strip():
            logger.error("Content is None or empty in _parse_response")
            return AnswerFormat(
                answer="I apologize, but I received an empty response. Please try again.",
                text_citations=[],
                image_citations=[],
            )

        try:
            # 1. Try generic robust repair first
            data = repair_json(content)
            logger.debug(f"repair_json returned type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}")

            # Check for empty dict (malformed LLM response with only whitespace)
            if not data or (isinstance(data, dict) and len(data) == 0):
                logger.error(f"repair_json returned empty dict. Raw content length: {len(content)}")
                # Check if content is mostly whitespace - indicates LLM failure
                non_whitespace = content.replace(" ", "").replace("\n", "").replace("\t", "").replace("{", "").replace("}", "")
                if len(non_whitespace) < 10:
                    logger.error("LLM returned malformed response (whitespace only). Returning retry message.")
                    return AnswerFormat(
                        answer="I apologize, but I received an incomplete response from the AI model. Please try your question again.",
                        text_citations=[],
                        image_citations=[],
                    )

            # Handle wrapped response format (e.g., {'final': {'answer': '...'}})
            if "answer" not in data:
                logger.debug(f"No 'answer' key in data. Available keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                # Look for nested dict containing 'answer' (e.g., 'final' wrapper)
                for key, value in data.items():
                    if isinstance(value, dict) and "answer" in value:
                        data = value
                        logger.debug(f"Found nested answer in dict key '{key}'")
                        break
                    elif isinstance(value, str) and '"answer"' in value:
                        try:
                            nested_data = repair_json(value)
                            if "answer" in nested_data:
                                data = nested_data
                                logger.debug(f"Found nested answer in JSON string key '{key}'")
                                break
                        except:
                            continue

            # 2. Validate and helper extraction
            answer_text = self._normalize_citation_brackets(data.get("answer", ""))

            # Log if answer is still empty
            if not answer_text:
                logger.warning(f"Answer text is empty after parsing. data.get('answer') = {data.get('answer')!r}")
                logger.warning(f"Full parsed data: {data}")

            return AnswerFormat(
                answer=answer_text,
                text_citations=data.get("text_citations", [])
                or data.get("text_Citations", []),
                image_citations=data.get("image_citations", [])
                or data.get("image_Citations", []),
            )

        except Exception as e:
            logger.warning(f"JSON repair/parsing failed: {e}. Falling back to regex.")

            # 3. Last resort: Regex extraction
            try:
                answer = self._extract_partial_answer(content) or ""
                text_citations = []
                image_citations = []

                # Try to extract citations with regex
                text_cit_match = re.search(
                    r'"text_citations"\s*:\s*\[(.*?)\]', content, re.DOTALL
                )
                if text_cit_match:
                    citations_str = text_cit_match.group(1)
                    text_citations = re.findall(r'"([^"]+)"', citations_str)

                image_cit_match = re.search(
                    r'"image_citations"\s*:\s*\[(.*?)\]', content, re.DOTALL
                )
                if image_cit_match:
                    citations_str = image_cit_match.group(1)
                    image_citations = re.findall(r'"([^"]+)"', citations_str)

                if answer:
                    logger.info(
                        f"Regex extraction recovered answer with {len(text_citations)} text and {len(image_citations)} image citations"
                    )
                    return AnswerFormat(
                        answer=self._normalize_citation_brackets(answer),
                        text_citations=text_citations,
                        image_citations=image_citations,
                    )

                logger.error(f"All parsing methods failed for content: {content[:500]}")
                return AnswerFormat(
                    answer="I apologize, but I encountered an issue interpreting the response. Please try asking your question again.",
                    text_citations=[],
                    image_citations=[],
                )

            except Exception as e:
                logger.error(f"Regex extraction failed: {e}")
                return AnswerFormat(
                    answer="I apologize, but I encountered an issue interpreting the response. Please try asking your question again.",
                    text_citations=[],
                    image_citations=[],
                )

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
        available_refs = {ref["ref_id"] for ref in retrieval_results["references"]}

        all_citations = set(answer.text_citations + answer.image_citations)
        valid_citations = all_citations & available_refs
        invalid_citations = all_citations - available_refs

        return {
            "valid": len(invalid_citations) == 0,
            "valid_citations": list(valid_citations),
            "invalid_citations": list(invalid_citations),
            "coverage": (
                len(valid_citations) / len(all_citations) if all_citations else 1.0
            ),
        }
