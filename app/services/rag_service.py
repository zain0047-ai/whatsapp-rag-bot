"""
RAG service — the orchestrator. Given a user message, this:

1. Loads the conversation's full history from Postgres (not a capped,
   expiring cache — see _load_conversation_history below for why)
2. Retrieves relevant knowledge (hybrid search)
3. Checks retrieval confidence against the configured threshold
4. Builds the grounded system prompt
5. Calls the LLM
6. Persists the exchange (messages table)

This is the single entrypoint both the WhatsApp webhook and the
test chat endpoint call — keeping behavior identical across channels.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import Conversation, Message
from app.rag.prompt_builder import average_confidence, build_system_prompt
from app.rag.retriever import hybrid_search
from app.services.llm_service import generate_answer

settings = get_settings()

LOW_CONFIDENCE_FALLBACK = (
    "I don't have enough verified information to answer that accurately. "
    "Let me connect you with someone who can help."
)

# How many past messages to feed back to the LLM as context. This used to be
# a Redis cache capped at 12 messages with a 6-hour TTL — real WhatsApp
# conversations routinely run longer than that or resume after a longer gap,
# so the bot would silently lose everything the client had already told it
# and re-ask the same qualifying questions from scratch. Postgres already
# keeps every message for the conversation's lifetime, so pulling from there
# instead means the client is never asked the same thing twice within one
# conversation. The limit here is just a token-budget safety net, not a
# memory-loss window like the old TTL was.
CONTEXT_HISTORY_LIMIT = 100


async def _load_conversation_history(db: AsyncSession, conversation_id: uuid.UUID) -> list[dict]:
    result = await db.execute(
        select(Message.role, Message.content)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(CONTEXT_HISTORY_LIMIT)
    )
    rows = result.all()
    rows.reverse()  # oldest first, as generate_answer expects
    return [{"role": role, "content": content} for role, content in rows]


async def answer_message(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    user_message: str,
    category: str | None = None,
    counselor_name: str | None = None,
) -> tuple[str, float]:
    """Returns (answer_text, confidence_score).

    `category`/`counselor_name` come from whichever counselor Omnichannel has
    assigned to the WhatsApp account this message belongs to — both default
    to None, which keeps retrieval unfiltered and the identity at today's
    original default (see prompt_builder.py) for any caller that doesn't
    know about counselors yet.
    """

    chunks = await hybrid_search(db, user_message, category=category)
    confidence = average_confidence(chunks)

    history = await _load_conversation_history(db, conversation_id)

    if confidence < settings.retrieval_confidence_threshold and not chunks:
        answer = LOW_CONFIDENCE_FALLBACK
    else:
        system_prompt = build_system_prompt(chunks, counselor_name=counselor_name, category=category)
        conversation_messages = history + [{"role": "user", "content": user_message}]
        answer = await generate_answer(system_prompt, conversation_messages)

    # Persist to DB
    db.add(Message(conversation_id=conversation_id, role="user", content=user_message))
    db.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            retrieved_chunk_ids=[str(c.chunk_id) for c in chunks],
            confidence_score=confidence,
        )
    )
    await db.commit()

    return answer, confidence


async def get_or_create_conversation(db: AsyncSession, user_id: uuid.UUID) -> Conversation:
    """
    MVP simplification: one active conversation per user. Multi-conversation
    threading (e.g. explicit "new topic") can be added later without
    changing this function's callers.
    """
    from sqlalchemy import select

    conversation = await db.scalar(
        select(Conversation)
        .where(Conversation.user_id == user_id, Conversation.status == "active")
        .order_by(Conversation.created_at.desc())
    )
    if conversation is None:
        conversation = Conversation(user_id=user_id, status="active")
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
    return conversation
