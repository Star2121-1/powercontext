# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Page-addressable PDF Sources with exact text evidence and citation lookup."""

from __future__ import annotations

import hashlib
import io
from typing import Any

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"


class PDFDocumentInput(BaseModel):
    """One complete PDF document."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One extracted page, identified by its document revision and ordinal."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A captured PDF page with stable document revision identity."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without leading or trailing whitespace")


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable, bounded identity for one page of one revision."""

    _validate_document_id(document_id)
    document_hash = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    source_id = f"pdf:{document_hash}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every page atomically, preserving blank pages and page order."""

    _validate_document_id(value.document_id)
    if not value.content:
        raise ValueError("PDF content must not be empty")
    try:
        reader = PdfReader(io.BytesIO(value.content))
        if reader.is_encrypted:
            if reader.decrypt("") == 0:
                raise ValueError("PDF requires a non-empty password")
        digest = hashlib.sha256(value.content).hexdigest()
        pages: list[PDFPageInput] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception as error:
                raise ValueError("PDF page text extraction failed") from error
            pages.append(
                PDFPageInput(
                    document_id=value.document_id,
                    document_sha256=digest,
                    page_number=number,
                    text="" if text is None else str(text),
                )
            )
        if not pages:
            raise ValueError("PDF document must contain at least one page")
        return tuple(pages)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF content is not a readable document") from error


class _PDFPageAdapter:
    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class _PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    _PDFPageAdapter(),
    projections=(_PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(memory: Any, question: str, limit: int = 10) -> tuple[Any, ...]:
    """Return exact page citations for a natural-language question."""

    import re

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=max(limit * 5, 50)))
    query_terms = set(re.findall(r"[a-z]+|\d+", question.lower()))
    scored: dict[tuple[str, int, str, str], tuple[float, Any]] = {}
    for hit in page.hits:
        key = (hit.memory_ref.family, hit.memory_ref.artifact_id, hit.entry_id, hit.entry_version_id)
        overlap = len(query_terms & set(re.findall(r"[a-z]+|\d+", hit.text.lower())))
        score = float(overlap)
        scored[key] = (
            score,
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            ),
        )
    listed = await memory.list()
    for record in listed.entries:
        citation = record.citation
        key = (
            citation.memory_ref.family,
            citation.memory_ref.artifact_id,
            citation.entry_id,
            citation.entry_version_id,
        )
        overlap = len(query_terms & set(re.findall(r"[a-z]+|\d+", record.entry.text.lower())))
        if overlap > 0 and (key not in scored or float(overlap) > scored[key][0]):
            scored[key] = (float(overlap), citation)
    ordered = sorted(scored.values(), key=lambda item: item[0], reverse=True)
    return tuple(citation for _, citation in ordered[:limit])


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
