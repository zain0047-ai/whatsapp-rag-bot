"""
Test chat endpoint — lets you talk to the bot from a browser/Postman
without needing WhatsApp set up yet. Uses the exact same rag_service
as the WhatsApp webhook, so behavior is identical.
"""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database.models import Document, User
from app.services import rag_service

router = APIRouter()


class ChatRequest(BaseModel):
    test_user_id: str  # any string you choose, e.g. "tester1" — simulates a WhatsApp number
    message: str
    # Which counselor's persona/knowledge to use — both optional. Omitted
    # (or either one alone) falls back to the original default identity and
    # unfiltered retrieval; see prompt_builder.py's DEFAULT_COUNSELOR_NAME /
    # DEFAULT_CATEGORY.
    category: str | None = None
    counselor_name: str | None = None


class ChatResponse(BaseModel):
    answer: str
    confidence: float


@router.post("/test", response_model=ChatResponse)
async def test_chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    # Reuse the `whatsapp_number` column to store the test id — keeps one User model for both channels.
    fake_number = f"test:{request.test_user_id}"

    user = await db.scalar(select(User).where(User.whatsapp_number == fake_number))
    if user is None:
        user = User(whatsapp_number=fake_number)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    conversation = await rag_service.get_or_create_conversation(db, user.id)
    answer, confidence = await rag_service.answer_message(
        db, conversation.id, request.message,
        category=request.category, counselor_name=request.counselor_name,
    )

    return ChatResponse(answer=answer, confidence=confidence)


@router.get("/categories", response_model=list[str])
async def list_categories(db: AsyncSession = Depends(get_db)):
    """Every category that has ingested knowledge. Omnichannel offers exactly
    these when an agent registers as a counselor — a counselor's category must
    match a document category exactly, or retrieval silently finds nothing."""
    rows = await db.scalars(
        select(Document.category)
        .where(Document.category.is_not(None), Document.category != "")
        .distinct()
        .order_by(Document.category)
    )
    return list(rows)
