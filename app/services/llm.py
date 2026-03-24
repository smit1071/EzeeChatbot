"""
llm.py -- Groq LLM interface.

Key design decisions:
  1. GROUNDING SYSTEM PROMPT: The bot is explicitly instructed to answer only
     from the provided context. When context is empty or irrelevant it must say
     so -- this prevents hallucination of information outside the KB.
  2. STREAMING: Uses Groq's streaming API so /chat can stream tokens to the
     client via Server-Sent Events, reducing perceived latency.
  3. WINDOW MEMORY: The last N conversation turns are prepended as alternating
     user/assistant messages -- Groq's chat completions accept multi-turn history
     natively, so no special memory chain is needed.
  4. TOKEN COUNTING: Groq returns usage metadata in the final chunk; we capture
     it for cost estimation.
"""

import json
from groq import Groq
from typing import List, Tuple
from app.core.config import settings
from app.models.schemas import ChatMessage


SYSTEM_PROMPT_TEMPLATE = (
    "You are EzeeChatBot, a helpful assistant that answers questions "
    "strictly based on the knowledge base provided below.\n\n"
    "RULES:\n"
    "- Answer ONLY using information found in the CONTEXT section.\n"
    "- If the answer is not present in the context, respond with exactly:\n"
    "  'I could not find an answer to that in the uploaded knowledge base.'\n"
    "- Do NOT fabricate, extrapolate, or use outside knowledge.\n"
    "- Be concise and direct. Cite the relevant part of the context when helpful.\n"
    "- If the user's question is ambiguous, ask for clarification.\n\n"
    "CONTEXT:\n"
    "{context}"
)

NOT_FOUND_SIGNAL = "i could not find an answer"


def build_system_prompt(context_chunks: List[str]) -> str:
    if not context_chunks:
        context = "(No relevant content was found in the knowledge base.)"
    else:
        parts = []
        for i, chunk in enumerate(context_chunks):
            parts.append("[Chunk " + str(i + 1) + "]: " + chunk)
        context = "\n\n---\n\n".join(parts)
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)


def build_messages(
    system_prompt: str,
    history: List[ChatMessage],
    user_message: str,
) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_message})
    return messages


def stream_chat(
    context_chunks: List[str],
    history: List[ChatMessage],
    user_message: str,
) -> Tuple[str, int, int]:
    client = Groq(api_key=settings.GROQ_API_KEY)
    system_prompt = build_system_prompt(context_chunks)
    messages = build_messages(system_prompt, history, user_message)

    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
        stream=False,
    )

    content = response.choices[0].message.content or ""
    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0

    return content, in_tok, out_tok


def stream_chat_sse(
    context_chunks: List[str],
    history: List[ChatMessage],
    user_message: str,
):
    client = Groq(api_key=settings.GROQ_API_KEY)
    system_prompt = build_system_prompt(context_chunks)
    messages = build_messages(system_prompt, history, user_message)

    stream = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
        stream=True,
    )

    full_text = []
    in_tok = 0
    out_tok = 0

    for chunk in stream:
        delta = None
        if chunk.choices:
            delta = chunk.choices[0].delta.content

        if delta:
            full_text.append(delta)
            payload = json.dumps({"type": "delta", "text": delta})
            yield "data: " + payload + "\n\n"

        if chunk.usage:
            in_tok = chunk.usage.prompt_tokens
            out_tok = chunk.usage.completion_tokens

    complete_text = "".join(full_text)
    is_grounded = NOT_FOUND_SIGNAL not in complete_text.lower()

    done_data = {
        "type": "done",
        "full_text": complete_text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "grounded": is_grounded,
    }
    yield "data: " + json.dumps(done_data) + "\n\n"


def is_not_found_response(text: str) -> bool:
    return NOT_FOUND_SIGNAL in text.lower()
