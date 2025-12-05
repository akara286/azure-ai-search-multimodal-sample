"""
Domain models for the multimodal RAG application.
Uses modern Python 3.10+ type hints.
"""

from typing import Literal, TypedDict
from pydantic import BaseModel, Field


class SearchConfig(TypedDict, total=False):
    """Configuration for search parameters."""

    chunk_count: int
    openai_api_mode: Literal["chat_completions"]
    use_semantic_ranker: bool
    use_streaming: bool
    use_knowledge_agent: bool


class SearchRequestParameters(TypedDict, total=False):
    """Structure for search request payload."""

    search: str
    top: int
    vector_queries: list[dict[str, str]] | None
    semantic_configuration_name: str | None
    search_fields: list[str] | None


class GroundingResult(TypedDict):
    """Structure for individual grounding results."""

    ref_id: str
    content: dict
    content_type: Literal["text", "image"]


class GroundingResults(TypedDict):
    """Structure for grounding results with references and queries."""

    references: list["GroundingResult"]
    search_queries: list[str]


class AnswerFormat(BaseModel):
    """Format for chat completion responses.

    Uses Field aliases to match the camelCase JSON output from the LLM.
    """

    answer: str
    text_citations: list[str] = Field(default_factory=list, alias="text_Citations")
    image_citations: list[str] = Field(default_factory=list, alias="image_Citations")

    model_config = {"populate_by_name": True}

    @classmethod
    def json_schema_for_openai(cls) -> dict:
        """Generate JSON schema compatible with OpenAI structured outputs."""
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer in Markdown format",
                },
                "text_Citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Reference IDs of text documents used",
                },
                "image_Citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Reference IDs of images used",
                },
            },
            "required": ["answer", "text_Citations", "image_Citations"],
            "additionalProperties": False,
        }


class MessageContent(TypedDict):
    """Content within a chat message."""

    text: str
    type: Literal["text"]


class Message(TypedDict):
    """Structure for chat messages."""

    role: Literal["user", "assistant", "system"]
    content: list[MessageContent]
