"""
Citation Handler for extracting and formatting citations.

Handles both text and image citations, including location metadata
for precise document references.
"""

import asyncio
import json
import os
import logging
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Tuple
from io import BytesIO

# SAS token expiry in minutes - configurable via environment variable
SAS_TOKEN_EXPIRY_MINUTES = int(os.environ.get("SAS_TOKEN_EXPIRY_MINUTES", "60"))
# Refresh delegation key when less than this many minutes remain
DELEGATION_KEY_REFRESH_THRESHOLD = 5

from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

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
    - Cached SAS tokens and user delegation keys for performance
    """

    def __init__(
        self,
        blob_service_client: Optional[BlobServiceClient] = None,
        artifacts_container: Optional[ContainerClient] = None,
        search_client=None,
    ):
        self.blob_service_client = blob_service_client
        self.artifacts_container = artifacts_container
        self.search_client = search_client

        # Cache for user delegation key (shared across all SAS tokens)
        self._delegation_key = None
        self._delegation_key_expiry: Optional[datetime] = None
        self._delegation_key_lock = asyncio.Lock()

        # Cache for SAS URLs: {blob_name: (url_without_page, expiry_time)}
        self._sas_cache: Dict[str, Tuple[str, datetime]] = {}
        self._sas_cache_lock = asyncio.Lock()

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
        Fetches citation details in parallel for better performance.

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
        ref_lookup = {ref["ref_id"]: ref for ref in references}

        # Collect refs to process
        refs_to_process = []
        for ref_id in ref_ids:
            if ref_id in ref_lookup:
                refs_to_process.append(ref_lookup[ref_id])
            else:
                logger.warning(f"Citation ref_id '{ref_id}' not found in references")

        if not refs_to_process:
            return []

        # Fetch all citations in parallel
        citations = await asyncio.gather(
            *[self._format_citation_async(ref) for ref in refs_to_process]
        )

        return list(citations)

    async def _format_citation_async(self, ref: RetrievalResult) -> Citation:
        """Format a reference into a Citation object, fetching locationMetadata if missing."""
        raw_data = ref.get("_raw", {})

        # Extract location metadata if available
        location_metadata = raw_data.get("locationMetadata")
        if isinstance(location_metadata, str):
            try:
                location_metadata = json.loads(location_metadata)
            except json.JSONDecodeError:
                location_metadata = None

        # Determine if we need to fetch from search index.
        # The Knowledge Base API often doesn't include locationMetadata in sourceData,
        # or returns incomplete/default values. We need to detect this reliably.
        #
        # Fetch from search index if:
        # 1. locationMetadata is None or empty
        # 2. locationMetadata is a dict but missing pageNumber
        # 3. locationMetadata came from Knowledge Base API (indicated by missing
        #    other expected fields like content_text in raw_data)
        needs_lookup = (
            location_metadata is None
            or not isinstance(location_metadata, dict)
            or "pageNumber" not in location_metadata
            # If raw_data doesn't have content_text, it's likely from KB API
            # which doesn't populate sourceData fully
            or (not raw_data.get("content_text") and not raw_data.get("locationMetadata"))
        )

        if needs_lookup and self.search_client:
            doc_id = ref["ref_id"]
            try:
                document = await self.search_client.get_document(doc_id)
                if document:
                    # Get locationMetadata from search index (authoritative source)
                    fetched_location = document.get("locationMetadata")
                    if fetched_location:
                        # Parse if it's a JSON string
                        if isinstance(fetched_location, str):
                            try:
                                fetched_location = json.loads(fetched_location)
                            except json.JSONDecodeError:
                                fetched_location = None
                        if fetched_location:
                            location_metadata = fetched_location
                            logger.debug(
                                f"Fetched locationMetadata for {doc_id}: page {location_metadata.get('pageNumber', 'N/A')}"
                            )

                    # Also update raw_data for other fields if missing
                    if not raw_data.get("document_title"):
                        raw_data["document_title"] = document.get("document_title")
                    if not raw_data.get("text_document_id"):
                        raw_data["text_document_id"] = document.get("text_document_id")
                    if not raw_data.get("image_document_id"):
                        raw_data["image_document_id"] = document.get(
                            "image_document_id"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to fetch document {doc_id} for locationMetadata: {e}"
                )

        # Get content_id (use ref_id as fallback)
        content_id = raw_data.get("content_id") or ref["ref_id"]
        ref_id = ref["ref_id"]

        # Extract page number from content_id or ref_id if locationMetadata is missing or has default value.
        # The content_id pattern is: {prefix}_pages_{page_index} where page_index is 0-based.
        # Example: "abc_pages_6" means page 7 (0-indexed).
        # This is a fallback for when locationMetadata isn't properly populated in the index.
        page_from_id = None
        # Try content_id first, then ref_id
        for id_to_check in [content_id, ref_id]:
            pages_match = re.search(r'_pages_(\d+)', id_to_check)
            if pages_match:
                page_from_id = int(pages_match.group(1)) + 1  # Convert 0-based to 1-based
                logger.debug(f"Extracted page {page_from_id} from ID: {id_to_check}")
                break

        # Ensure location_metadata has required structure for frontend
        if location_metadata is None:
            location_metadata = {"pageNumber": page_from_id or 1, "boundingPolygons": ""}
            if page_from_id:
                logger.info(f"Using page {page_from_id} extracted from ID (no locationMetadata)")
        elif isinstance(location_metadata, dict):
            # Use page from content_id/ref_id if locationMetadata has default pageNumber of 1
            # and we extracted a different page from the ID
            if "pageNumber" not in location_metadata:
                location_metadata["pageNumber"] = page_from_id or 1
            elif location_metadata.get("pageNumber") == 1 and page_from_id and page_from_id > 1:
                # Override default pageNumber with extracted value
                location_metadata["pageNumber"] = page_from_id
                logger.info(f"Overriding pageNumber=1 with extracted page {page_from_id} from ID: {content_id}")
            if "boundingPolygons" not in location_metadata:
                location_metadata["boundingPolygons"] = ""

        # Get document ID (for the parent document)
        doc_id = (
            raw_data.get("text_document_id")
            or raw_data.get("image_document_id")
            or ref["ref_id"]
        )

        # Get document title (PDF filename)
        doc_title = raw_data.get("document_title") or "Document"

        # Get text content
        text = None
        if ref["content_type"] == "text":
            content = ref["content"]
            if isinstance(content, dict):
                text = content.get("text", "")
            else:
                text = str(content)

        # Return in frontend-expected format with camelCase keys
        return {
            "docId": doc_id,
            "content_id": content_id,
            "title": doc_title,
            "text": text,
            "locationMetadata": location_metadata,
        }

    # Note: _format_citation sync version removed - use _format_citation_async instead

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
                document.get("text_document_id") or document.get("image_document_id")
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

    async def _get_delegation_key(self):
        """Get or refresh the cached user delegation key."""
        now = datetime.now(timezone.utc)

        async with self._delegation_key_lock:
            # Check if we have a valid cached key
            if (
                self._delegation_key is not None
                and self._delegation_key_expiry is not None
                and self._delegation_key_expiry > now + timedelta(minutes=DELEGATION_KEY_REFRESH_THRESHOLD)
            ):
                return self._delegation_key, self._delegation_key_expiry

            # Generate new delegation key
            start_time = now
            expiry_time = now + timedelta(minutes=SAS_TOKEN_EXPIRY_MINUTES)

            self._delegation_key = await self.blob_service_client.get_user_delegation_key(
                key_start_time=start_time,
                key_expiry_time=expiry_time,
            )
            self._delegation_key_expiry = expiry_time
            logger.debug(f"Generated new user delegation key, expires at {expiry_time}")

            return self._delegation_key, self._delegation_key_expiry

    async def get_document_url(
        self, file_name: str, page_number: int = 1
    ) -> Optional[str]:
        """
        Generate a SAS URL for accessing a document (PDF).
        Uses cached delegation keys and SAS tokens for performance.

        Args:
            file_name: Name of the file (e.g., "document.pdf")
            page_number: Optional page number to link to (default: 1)

        Returns:
            Signed URL with SAS token for accessing the document, or dict with error
        """
        if not self.artifacts_container or not self.blob_service_client:
            logger.warning("Blob storage not configured, cannot generate document URL")
            return None

        try:
            # Normalize blob name
            blob_name = file_name.replace("\\", "/")
            now = datetime.now(timezone.utc)

            # Check SAS cache first
            async with self._sas_cache_lock:
                if blob_name in self._sas_cache:
                    cached_url, expiry = self._sas_cache[blob_name]
                    # Use cached URL if it has at least 5 minutes remaining
                    if expiry > now + timedelta(minutes=DELEGATION_KEY_REFRESH_THRESHOLD):
                        logger.debug(f"Using cached SAS URL for {blob_name}")
                        if page_number > 1:
                            return f"{cached_url}#page={page_number}"
                        return cached_url

            # Need to generate new SAS URL
            blob_client = self.artifacts_container.get_blob_client(blob_name)

            if not await blob_client.exists():
                # Fallback: Try just the filename at root
                flat_name = os.path.basename(blob_name)
                if flat_name != blob_name:
                    logger.info(f"Blob {blob_name} not found, trying root: {flat_name}")
                    blob_name = flat_name
                    blob_client = self.artifacts_container.get_blob_client(blob_name)

            # Get cached delegation key
            user_delegation_key, expiry_time = await self._get_delegation_key()

            sas_token = generate_blob_sas(
                account_name=blob_client.account_name or "",
                container_name=self.artifacts_container.container_name,
                blob_name=blob_name,
                user_delegation_key=user_delegation_key,
                permission=BlobSasPermissions(read=True),
                expiry=expiry_time,
            )

            signed_url = f"{blob_client.url}?{sas_token}"

            # Cache the SAS URL (without page fragment)
            async with self._sas_cache_lock:
                self._sas_cache[blob_name] = (signed_url, expiry_time)

            if page_number > 1:
                signed_url += f"#page={page_number}"
            return signed_url

        except Exception as e:
            error_msg = f"Failed to generate SAS URL for {file_name}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}



class BlobHelper:
    """Helper class for blob storage operations, passed to AnswerGenerator."""

    def __init__(self, citation_handler: CitationHandler):
        self.citation_handler = citation_handler

    async def get_image_base64(self, blob_path: str) -> Optional[str]:
        """Get image as base64 from blob storage."""
        return await self.citation_handler.get_image_as_base64(blob_path)
