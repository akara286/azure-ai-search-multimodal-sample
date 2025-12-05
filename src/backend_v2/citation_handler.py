"""
Citation Handler for extracting and formatting citations.

Handles both text and image citations, including location metadata
for precise document references.
"""

import logging
from typing import List, Optional
from azure.storage.blob.aio import BlobServiceClient, ContainerClient

from models import (
    AnswerFormat,
    Citation,
    RetrievalResult,
    RetrievalResults,
)

logger = logging.getLogger("backend_v2.citation_handler")


class CitationHandler:
    """
    Handles citation extraction, formatting, and document retrieval.

    Features:
    - Extract citations from LLM responses
    - Match citations to retrieved documents
    - Fetch full document details for citation display
    - Handle both text and image citations
    """

    def __init__(
        self,
        blob_service_client: Optional[BlobServiceClient] = None,
        artifacts_container: Optional[ContainerClient] = None,
    ):
        self.blob_service_client = blob_service_client
        self.artifacts_container = artifacts_container

    async def extract_citations(
        self,
        answer: AnswerFormat,
        retrieval_results: RetrievalResults,
    ) -> dict:
        """
        Extract and format citations from the answer.

        Args:
            answer: The generated answer with citation IDs
            retrieval_results: The retrieved documents

        Returns:
            Dict with text_citations and image_citations lists
        """
        text_citations = await self._get_citations(
            answer.text_citations,
            retrieval_results["references"],
            content_type="text",
        )

        image_citations = await self._get_citations(
            answer.image_citations,
            retrieval_results["references"],
            content_type="image",
        )

        return {
            "text_citations": text_citations,
            "image_citations": image_citations,
        }

    async def _get_citations(
        self,
        ref_ids: List[str],
        references: List[RetrievalResult],
        content_type: str,
    ) -> List[Citation]:
        """
        Get formatted citations for a list of reference IDs.

        Args:
            ref_ids: List of reference IDs from the answer
            references: Retrieved documents
            content_type: Filter for text or image

        Returns:
            List of Citation objects
        """
        if not ref_ids:
            return []

        # Build lookup from ref_id to reference
        ref_lookup = {
            ref["ref_id"]: ref for ref in references
        }

        citations = []
        for ref_id in ref_ids:
            if ref_id in ref_lookup:
                ref = ref_lookup[ref_id]
                citation = self._format_citation(ref)
                citations.append(citation)
            else:
                logger.warning(f"Citation ref_id '{ref_id}' not found in references")

        return citations

    def _format_citation(self, ref: RetrievalResult) -> Citation:
        """Format a reference into a Citation object."""
        raw_data = ref.get("_raw", {})

        # Extract location metadata if available
        location_metadata = raw_data.get("locationMetadata")
        if isinstance(location_metadata, str):
            import json
            try:
                location_metadata = json.loads(location_metadata)
            except json.JSONDecodeError:
                location_metadata = None

        # Get document ID
        doc_id = (
            raw_data.get("text_document_id") or
            raw_data.get("image_document_id") or
            ref["ref_id"]
        )

        # Get text content
        text = None
        if ref["content_type"] == "text":
            content = ref["content"]
            if isinstance(content, dict):
                text = content.get("text", "")
            else:
                text = str(content)

        return {
            "ref_id": ref["ref_id"],
            "content_type": ref["content_type"],
            "location_metadata": location_metadata,
            "text": text,
            "title": raw_data.get("document_title"),
            "doc_id": doc_id,
        }

    async def get_citation_document(
        self,
        doc_id: str,
        search_client,
    ) -> Optional[dict]:
        """
        Retrieve the full document for a citation.

        Args:
            doc_id: The document ID
            search_client: Azure Search client

        Returns:
            Full document data or None
        """
        try:
            document = await search_client.get_document(doc_id)
            return self._format_document_for_display(document)
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None

    def _format_document_for_display(self, document: dict) -> dict:
        """Format a document for citation display."""
        return {
            "content_id": document.get("content_id"),
            "title": document.get("document_title"),
            "text": document.get("content_text"),
            "location_metadata": document.get("locationMetadata"),
            "content_path": document.get("content_path"),
            "doc_id": (
                document.get("text_document_id") or
                document.get("image_document_id")
            ),
        }

    async def get_image_as_base64(self, blob_path: str) -> Optional[str]:
        """
        Retrieve an image from blob storage as base64.

        Args:
            blob_path: Path to the blob

        Returns:
            Base64-encoded image or None
        """
        if not self.artifacts_container:
            return None

        try:
            from io import BytesIO
            import base64

            blob_client = self.artifacts_container.get_blob_client(blob_path)
            stream = BytesIO()
            download_stream = await blob_client.download_blob()
            await download_stream.readinto(stream)

            return base64.b64encode(stream.getvalue()).decode("utf-8")

        except Exception as e:
            logger.warning(f"Failed to retrieve blob {blob_path}: {e}")

            # Try alternative path format (for portal-created indexes)
            try:
                path_parts = blob_path.split("/")
                if len(path_parts) > 1:
                    container_name = path_parts[0]
                    blob_name = "/".join(path_parts[1:])

                    alt_container = self.blob_service_client.get_container_client(
                        container_name
                    )
                    blob_client = alt_container.get_blob_client(blob_name)

                    stream = BytesIO()
                    download_stream = await blob_client.download_blob()
                    await download_stream.readinto(stream)

                    return base64.b64encode(stream.getvalue()).decode("utf-8")
            except Exception as e2:
                logger.error(f"Alternative blob retrieval also failed: {e2}")

            return None


class BlobHelper:
    """Helper class for blob storage operations, passed to AnswerGenerator."""

    def __init__(self, citation_handler: CitationHandler):
        self.citation_handler = citation_handler

    async def get_image_base64(self, blob_path: str) -> Optional[str]:
        """Get image as base64 from blob storage."""
        return await self.citation_handler.get_image_as_base64(blob_path)
