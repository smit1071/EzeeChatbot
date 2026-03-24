"""
chat.py — POST /chat

Flow:
  1. Look up bot by bot_id.
  2. Embed the user's query.
  3. Retrieve top-K chunks from FAISS.
  4. Merge incoming conversation_history with stored window memory
     (last 10 turns), keeping most recent 10 pairs.
  5. Build prompt, call Groq, stream response via SSE.
  6. Update stats + memory in the bot record.

Two response modes:
  - Default: StreamingResponse (SSE, recommended for production)
  - ?stream=false: returns a regular JSON ChatResponse (useful for testing)
"""

import time
import json
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.services.embedder import embed_query
from app.services.vector_store import retrieve
from app.services.llm import stream_chat, stream_chat_sse, is_not_found_response
from app.core.store import bot_store, ConversationTurn
from app.core.config import settings

router = APIRouter()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_window_history(bot_id: str, incoming: list[ChatMessage]) -> list[ChatMessage]:
    """
    Merge incoming history (from client) with stored memory, deduplicate,
    and return the last WINDOW_MEMORY_SIZE pairs (user+assistant = 2 items per pair).
    Max list length = 2 * WINDOW_MEMORY_SIZE.
    """
    record = bot_store.get(bot_id)
    if not record:
        return incoming[-settings.WINDOW_MEMORY_SIZE * 2:]

    # Convert stored memory turns to ChatMessage
    stored = [
        ChatMessage(role=t.role, content=t.content)
        for t in record.memory
    ]

    # Prefer incoming history (it may already include stored turns from client)
    combined = incoming if incoming else stored
    cap = settings.WINDOW_MEMORY_SIZE * 2
    return combined[-cap:]


def _update_record(
    bot_id: str,
    user_msg: str,
    assistant_msg: str,
    latency_ms: float,
    in_tok: int,
    out_tok: int,
    grounded: bool,
):
    record = bot_store.get(bot_id)
    if not record:
        return

    record.total_messages += 1
    record.total_latency_ms += latency_ms
    record.total_input_tokens += in_tok
    record.total_output_tokens += out_tok
    if not grounded:
        record.unanswered_count += 1

    # Append to sliding window memory
    record.memory.append(ConversationTurn(role="user", content=user_msg))
    record.memory.append(ConversationTurn(role="assistant", content=assistant_msg))

    # Keep only last WINDOW_MEMORY_SIZE * 2 turns
    cap = settings.WINDOW_MEMORY_SIZE * 2
    if len(record.memory) > cap:
        record.memory = record.memory[-cap:]

    bot_store.update(record)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    request: ChatRequest,
    stream: bool = Query(default=True, description="Set false for non-streaming JSON"),
):
    record = bot_store.get(request.bot_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Bot '{request.bot_id}' not found.")
    if record.faiss_index is None:
        raise HTTPException(status_code=400, detail="Bot has no indexed knowledge base yet.")

    # 1. Embed query
    q_emb = embed_query(request.user_message)

    # 2. Retrieve chunks
    results = retrieve(
        index=record.faiss_index,
        chunks=record.chunks,
        query_embedding=q_emb,
        top_k=settings.TOP_K_CHUNKS,
    )
    context_chunks = [chunk for chunk, _ in results]

    # 3. Window memory
    history = _get_window_history(request.bot_id, request.conversation_history or [])

    # ── Streaming mode ─────────────────────────────────────────────────────────
    if stream:
        start = time.time()

        # We need to capture stats after streaming completes; wrap generator
        full_response_parts = []
        stats_payload = {}

        def _sse_wrapper():
            nonlocal stats_payload
            t0 = time.time()
            for chunk in stream_chat_sse(context_chunks, history, request.user_message):
                full_response_parts.append(chunk)
                yield chunk
                # Parse the [DONE] frame to capture stats
                if '"type": "done"' in chunk:
                    try:
                        payload = json.loads(chunk.replace("data: ", "").strip())
                        stats_payload = payload
                    except Exception:
                        pass

            latency = (time.time() - t0) * 1000
            _update_record(
                bot_id=request.bot_id,
                user_msg=request.user_message,
                assistant_msg=stats_payload.get("full_text", ""),
                latency_ms=latency,
                in_tok=stats_payload.get("input_tokens", 0),
                out_tok=stats_payload.get("output_tokens", 0),
                grounded=stats_payload.get("grounded", True),
            )

        return StreamingResponse(
            _sse_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming JSON mode ────────────────────────────────────────────────
    t0 = time.time()
    answer, in_tok, out_tok = stream_chat(context_chunks, history, request.user_message)
    latency_ms = (time.time() - t0) * 1000
    grounded = not is_not_found_response(answer)

    _update_record(
        bot_id=request.bot_id,
        user_msg=request.user_message,
        assistant_msg=answer,
        latency_ms=latency_ms,
        in_tok=in_tok,
        out_tok=out_tok,
        grounded=grounded,
    )

    return ChatResponse(
        answer=answer,
        sources_used=len(context_chunks),
        grounded=grounded,
        latency_ms=round(latency_ms, 2),
    )
