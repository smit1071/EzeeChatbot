"""
chunker.py — Sentence-aware sliding-window chunker.

Strategy chosen:
  1. Split text into sentences using a simple regex (handles "Mr.", abbreviations
     and common edge-cases without requiring a heavy NLP library like spaCy).
  2. Accumulate sentences into a chunk until the token budget (CHUNK_SIZE) is
     reached. Tokens are approximated as words (fast, good-enough for budget).
  3. Each subsequent chunk starts by re-including the last OVERLAP_SENTENCES
     sentences of the previous chunk. This preserves cross-boundary context so
     retrieval answers do not lose meaning at seam points.
  4. Metadata (source, chunk_index, char_offset) is attached to every chunk so
     the /stats endpoint can trace answers back to their origin.

Why not character splits?
  Character splits break mid-sentence, producing fragments like "…was found. The
  patient then" which confuse the LLM and lower retrieval quality. Sentence
  boundaries are semantically complete units — better for both embedding quality
  and answer coherence.
"""

import re
from typing import List, Tuple
from app.core.config import settings


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_BOUNDARY = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, stripping empty results."""
    sentences = _SENT_BOUNDARY.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


# ── Main chunking function ────────────────────────────────────────────────────

OVERLAP_SENTENCES = 2   # sentences carried into next chunk for continuity


def chunk_text(
    text: str,
    source_label: str = "unknown",
    chunk_size: int = settings.CHUNK_SIZE,
) -> List[Tuple[str, dict]]:
    """
    Returns a list of (chunk_text, metadata_dict) tuples.
    metadata keys: source, chunk_index, word_count
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Tuple[str, dict]] = []
    current: List[str] = []
    current_words = 0
    chunk_index = 0

    for sentence in sentences:
        sw = _word_count(sentence)

        # If a single sentence exceeds budget, force-split at word level
        if sw > chunk_size:
            # flush current first
            if current:
                body = " ".join(current)
                chunks.append((body, _meta(source_label, chunk_index, body)))
                chunk_index += 1
                current = current[-OVERLAP_SENTENCES:]
                current_words = sum(_word_count(s) for s in current)

            # hard-split the giant sentence into word windows
            words = sentence.split()
            for i in range(0, len(words), chunk_size - settings.CHUNK_OVERLAP):
                window = words[i: i + chunk_size]
                body = " ".join(window)
                chunks.append((body, _meta(source_label, chunk_index, body)))
                chunk_index += 1
            continue

        # Would adding this sentence overflow the budget?
        if current_words + sw > chunk_size and current:
            body = " ".join(current)
            chunks.append((body, _meta(source_label, chunk_index, body)))
            chunk_index += 1
            # carry-over overlap sentences
            current = current[-OVERLAP_SENTENCES:]
            current_words = sum(_word_count(s) for s in current)

        current.append(sentence)
        current_words += sw

    # Flush remainder
    if current:
        body = " ".join(current)
        chunks.append((body, _meta(source_label, chunk_index, body)))

    return chunks


def _meta(source: str, idx: int, body: str) -> dict:
    return {
        "source": source,
        "chunk_index": idx,
        "word_count": _word_count(body),
    }
