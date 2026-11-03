"""
Hybrid retriever — combines pgvector semantic (cosine) search with
Postgres full-text keyword search, merges and de-duplicates results.

Why hybrid: semantic search alone misses exact terms (scholarship
names, program codes); keyword search alone misses paraphrased
questions. Combining both covers more real user phrasing.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import Document, DocumentChunk
from app.services.embedding_service import embed_text

settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_text: str
    score: float  # higher = more relevant, roughly 0-1


async def _semantic_search(
    db: AsyncSession, query_embedding: list[float], top_k: int, category: str | None = None
) -> list[RetrievedChunk]:
    # cosine distance via pgvector's `<=>` operator; convert to a similarity score (1 - distance)
    conditions = [Document.status == "active"]
    if category:
        conditions.append(Document.category == category)

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_text,
            Document.title,
            DocumentChunk.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(*conditions)
        .order_by("distance")
        .limit(top_k)
    )
    rows = (await db.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.title,
            chunk_text=row.chunk_text,
            score=max(0.0, 1.0 - row.distance),
        )
        for row in rows
    ]


async def _keyword_search(
    db: AsyncSession, query: str, top_k: int, category: str | None = None
) -> list[RetrievedChunk]:
    # Built conditionally rather than an "IS NULL OR ..." bind parameter —
    # asyncpg's extended query protocol needs an unambiguous type for every
    # parameter, and a parameter that's sometimes NULL and sometimes a string
    # in the same query is exactly the kind of case that trips it up. Simplest
    # fix: only include the parameter (and the clause referencing it) when
    # there actually is a category to filter by.
    category_clause = "AND d.category = :category" if category else ""
    stmt = text(
        f"""
        SELECT dc.id, dc.document_id, dc.chunk_text, d.title,
               ts_rank(to_tsvector('english', dc.chunk_text), plainto_tsquery('english', :query)) AS rank
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE d.status = 'active'
          {category_clause}
          AND to_tsvector('english', dc.chunk_text) @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :top_k
        """
    )
    params = {"query": query, "top_k": top_k}
    if category:
        params["category"] = category
    rows = (await db.execute(stmt, params)).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.title,
            chunk_text=row.chunk_text,
            score=float(row.rank),
        )
        for row in rows
    ]


async def hybrid_search(
    db: AsyncSession, query: str, top_k: int | None = None, category: str | None = None
) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k

    query_embedding = await embed_text(query)
    semantic_results = await _semantic_search(db, query_embedding, top_k, category=category)

    # Keyword search on non-English/Roman Urdu queries usually returns little —
    # that's expected and fine, semantic search carries most of the weight.
    try:
        keyword_results = await _keyword_search(db, query, top_k, category=category)
    except Exception:
        keyword_results = []

    merged: dict[uuid.UUID, RetrievedChunk] = {}
    for result in semantic_results + keyword_results:
        existing = merged.get(result.chunk_id)
        if existing is None or result.score > existing.score:
            merged[result.chunk_id] = result

    ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return ranked[:top_k]
