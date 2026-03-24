"""
stats.py — GET /stats/{bot_id}

Returns aggregate runtime metrics for a single bot.
Cost is estimated from token counts × Groq's listed rates (configurable in config.py).
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import StatsResponse
from app.core.store import bot_store
from app.core.config import settings

router = APIRouter()


@router.get("/stats/{bot_id}", response_model=StatsResponse)
async def get_stats(bot_id: str):
    record = bot_store.get(bot_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found.")

    avg_latency = (
        record.total_latency_ms / record.total_messages
        if record.total_messages > 0
        else 0.0
    )

    # Cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    estimated_cost = (
        record.total_input_tokens * settings.INPUT_COST_PER_1M
        + record.total_output_tokens * settings.OUTPUT_COST_PER_1M
    ) / 1_000_000

    return StatsResponse(
        bot_id=bot_id,
        bot_name=record.name,
        total_messages=record.total_messages,
        avg_latency_ms=round(avg_latency, 2),
        estimated_cost_usd=round(estimated_cost, 6),
        unanswered_questions=record.unanswered_count,
        chunks_in_index=len(record.chunks),
    )
