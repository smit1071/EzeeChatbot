# EzeeChatBot — Backend API

A FastAPI backend that lets users upload their own knowledge base (PDF, URL, or plain text) and instantly chat with a grounded LLM that answers only from that content.

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI | Async, fast, native SSE streaming |
| LLM | Groq (`llama3-8b-8192`) | Fast open-source LLM, free tier available |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, runs on CPU, excellent quality |
| Vector DB | FAISS (IndexFlatIP) | In-process, zero setup, isolated per bot |
| PDF | PyMuPDF | Robust text extraction, preserves page order |
| HTTP | httpx + BeautifulSoup | Async-friendly, clean HTML stripping |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```
# Required
GROQ_API_KEY='Groq API Key'

# Optional overrides (defaults shown)
```GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_CHUNKS=5
WINDOW_MEMORY_SIZE=10
```

### 4. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI available at: `http://localhost:8000/docs`

---

## API Reference

### `POST /upload` — Upload plain text or URL

```json
{
  "source_type": "text",
  "content": "LangChain is a framework for building LLM applications...",
  "bot_name": "LangChain Bot"
}
```

```json
{
  "source_type": "url",
  "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
  "bot_name": "AI Wikipedia Bot"
}
```

**Response:**
```json
{
  "bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "bot_name": "LangChain Bot",
  "chunks_indexed": 42,
  "message": "Knowledge base ready. 42 chunks indexed."
}
```

---

### `POST /upload/pdf` — Upload PDF (multipart)

```bash
curl -X POST http://localhost:8000/upload/pdf \
  -F "file=@my_document.pdf" \
  -F "bot_name=My PDF Bot"
```

---

### `POST /chat` — Chat with a bot

```json
{
  "bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_message": "What is the main topic of this document?",
  "conversation_history": []
}
```

**Default**: streams Server-Sent Events (SSE).

```
data: {"type": "delta", "text": "The main"}
data: {"type": "delta", "text": " topic is..."}
data: {"type": "done", "full_text": "The main topic is...", "input_tokens": 312, "output_tokens": 48, "grounded": true}
```

**Non-streaming** (add `?stream=false`):
```json
{
  "answer": "The main topic is...",
  "sources_used": 4,
  "grounded": true,
  "latency_ms": 823.4
}
```

---

### `GET /stats/{bot_id}` — Runtime metrics

```json
{
  "bot_id": "550e8400-e29b-41d4-a716-446655440000",
  "bot_name": "LangChain Bot",
  "total_messages": 15,
  "avg_latency_ms": 921.3,
  "estimated_cost_usd": 0.000042,
  "unanswered_questions": 2,
  "chunks_in_index": 42
}
```

`unanswered_questions` counts responses where the bot said it could not find the answer.

---

## Architecture

```
POST /upload
  └─ loader.py         # fetch text from URL / PDF / raw string
  └─ chunker.py        # sentence-aware sliding window chunks
  └─ embedder.py       # sentence-transformers → float32 vectors
  └─ vector_store.py   # FAISS IndexFlatIP, isolated per bot
  └─ store.py          # in-memory BotRecord registry

POST /chat
  └─ embedder.py       # embed query
  └─ vector_store.py   # retrieve top-K chunks (cosine similarity)
  └─ store.py          # load window memory (last 10 turns)
  └─ llm.py            # Groq streaming, grounding system prompt
  └─ store.py          # update memory + stats
```

---

## Chunking Strategy

**Approach: Sentence-aware sliding-window chunking**

Located in `app/services/chunker.py`.

### Why not character splits?

A naive character split (e.g. every 500 characters) breaks mid-sentence, producing fragments like `"…was found. The patient the"`. This produces:
- **Poor embeddings** — the vector for a broken sentence does not represent any coherent concept.
- **Worse retrieval** — the cosine similarity of a fragmented chunk to a natural-language question is lower than a complete-sentence chunk covering the same concept.
- **LLM confusion** — the LLM cannot infer meaning from truncated context.

### How our chunker works

1. **Sentence splitting** using a regex that respects common abbreviations (`Mr.`, `Dr.`, `U.S.`, etc.) and terminal punctuation (`?`, `!`, `.`).
2. **Word-budget accumulation**: sentences are appended to the current chunk until the word count exceeds `CHUNK_SIZE` (default 512 words ≈ ~680 tokens). Word count is used instead of token count because it is fast and the difference is negligible for budget purposes.
3. **Sentence-level overlap**: when a chunk is sealed, the last 2 sentences are carried into the next chunk. This preserves cross-boundary context — e.g. an answer that spans two chunks remains retrievable.
4. **Giant sentence fallback**: if a single sentence exceeds the budget (common in legal/technical docs), it is hard-split at word boundaries with a `CHUNK_OVERLAP`-word overlap.
5. **Metadata per chunk**: `source`, `chunk_index`, `word_count` are stored alongside each chunk for traceability.

### Result

Complete, semantically coherent units that embed cleanly and retrieve accurately. The overlap prevents boundary blindness without the index bloat of full sliding windows.

---

## Hallucination Handling

The system prompt (`app/services/llm.py`) explicitly instructs the model:

> "Answer ONLY using information found in the CONTEXT section. If the answer is not present, respond with: 'I could not find an answer to that in the uploaded knowledge base.'"

Additional guardrails:
- **Score threshold (0.25)**: FAISS results below this cosine similarity floor are dropped, so the LLM sees no context rather than irrelevant context.
- **Empty context signal**: when no chunks pass the threshold, the prompt context section says `"(No relevant content was found in the knowledge base.)"` — the LLM reliably triggers the fallback.
- **`grounded` field**: every response includes a boolean `grounded` indicating whether the fallback was triggered, and the `/stats` endpoint counts these as `unanswered_questions`.

---

## Window Memory

The last 10 user+assistant turn pairs (20 messages total) are stored per bot in `BotRecord.memory`. On each `/chat` call the stored turns are merged with any `conversation_history` supplied by the client, capped at 20 messages, and injected as standard `user`/`assistant` messages in the Groq API call — no special chain is needed.

---

## Multi-Bot Isolation

Each bot gets:
- A separate UUID (`bot_id`)
- Its own `BotRecord` in the in-memory `BotStore`
- Its own FAISS index (`IndexFlatIP`) — retrieval queries **only** search the index for that bot

There is zero cross-bot bleed at the FAISS layer.

---

## What I Would Do Differently With More Time

1. **Persistent storage**: Replace the in-memory `BotStore` with Redis (memory + TTL) for bot metadata and a disk-persisted FAISS index (`faiss.write_index`). Currently all knowledge bases are lost on server restart.

2. **Semantic chunking with a small classifier**: Use a lightweight boundary classifier (e.g. fine-tuned on paragraph-break patterns) rather than regex sentence splitting. This would handle bullet lists, tables, and code blocks more gracefully — all of which appear frequently in real PDFs and break regex-based sentence splitters.

3. **Async embedding**: Move `embed_texts` into a thread pool (`run_in_executor`) so the event loop is not blocked during large uploads.

4. **Reranking**: Add a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) after FAISS retrieval to re-score the top-K candidates before injecting them into the prompt. This significantly improves answer accuracy at the cost of ~50 ms extra latency.

5. **Authentication + rate limiting**: Add per-bot API keys and request throttling for production multi-tenant use.

---

## Running Tests (Manual)

```bash
# 1. Upload text
curl -s -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"source_type":"text","content":"The Eiffel Tower is 330 metres tall and located in Paris, France.","bot_name":"Test Bot"}' | jq .

# 2. Chat (non-streaming for easy inspection)
curl -s -X POST "http://localhost:8000/chat?stream=false" \
  -H "Content-Type: application/json" \
  -d '{"bot_id":"<BOT_ID>","user_message":"How tall is the Eiffel Tower?"}' | jq .

# 3. Ask something not in the KB (should trigger fallback)
curl -s -X POST "http://localhost:8000/chat?stream=false" \
  -H "Content-Type: application/json" \
  -d '{"bot_id":"<BOT_ID>","user_message":"What is the population of Paris?"}' | jq .

# 4. Stats
curl -s http://localhost:8000/stats/<BOT_ID> | jq .
```
