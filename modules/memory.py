"""Generic in-process conversation memory for chat endpoints.

Module-agnostic: a conversation is a sequence of (question, answer) prose
exchanges under an opaque key the caller chooses. Plant Agent keys carry the
JWT identity (client, login) besides the caller-chosen conversation_id, so a
guessed id never leaks another tenant's conversation; other modules can adopt
the same scheme.

Conversations are kept whole — nothing is trimmed on write. The only limits
are safety bounds, not features:
  - `history()` replays the most recent exchanges that fit a character budget
    (a cheap proxy for tokens, ~4 chars/token), so a long conversation
    degrades gracefully — oldest turns drop out of the replayed context
    instead of the conversation being cut short.
  - TTL evicts conversations idle for a day, and a global cap bounds total
    process memory (oldest evicted first).

The store is a plain dict guarded by the single asyncio event loop.
Persistence across restarts is out of scope (frontend follow-up spec).
"""
from __future__ import annotations

import os
import time
from collections import deque

TTL_S = int(os.environ.get("MTIA_MEMORY_TTL_S", "86400"))                 # 24 h
# Replay budget in characters (~4 chars/token). Sized to share NUM_CTX
# (16384 tokens) with the tool catalog and the reasoning budget.
MAX_CHARS = int(os.environ.get("MTIA_MEMORY_MAX_CHARS", "24000"))
MAX_CONVERSATIONS = int(os.environ.get("MTIA_MEMORY_MAX_CONVERSATIONS", "1000"))
# Backstop against a runaway single conversation; far above real usage.
_MAX_EXCHANGES = 500


class ConversationStore:
    def __init__(self, ttl_s: float = TTL_S, max_chars: int = MAX_CHARS,
                 max_conversations: int = MAX_CONVERSATIONS,
                 clock=time.monotonic):
        self._ttl_s = ttl_s
        self._max_chars = max_chars
        self._max_conversations = max_conversations
        self._clock = clock
        # key -> [last_used, deque[(question, answer)]]
        self._data: dict = {}

    def history(self, key) -> list[dict]:
        """Most recent exchanges fitting the char budget, oldest first.

        The newest exchange is always included even if it alone exceeds the
        budget — otherwise one long answer would wipe the memory entirely.
        """
        self._prune()
        entry = self._data.get(key)
        if entry is None:
            return []
        entry[0] = self._clock()
        picked: list = []
        used = 0
        for question, answer in reversed(entry[1]):
            cost = len(question) + len(answer)
            if picked and used + cost > self._max_chars:
                break
            picked.append((question, answer))
            used += cost
        messages = []
        for question, answer in reversed(picked):
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

    def append(self, key, question: str, answer: str) -> None:
        self._prune()
        entry = self._data.get(key)
        if entry is None:
            entry = self._data[key] = [self._clock(), deque(maxlen=_MAX_EXCHANGES)]
        entry[0] = self._clock()
        entry[1].append((question, answer))

    def _prune(self) -> None:
        now = self._clock()
        expired = [k for k, (used, _) in self._data.items()
                   if now - used > self._ttl_s]
        for k in expired:
            del self._data[k]
        while len(self._data) >= self._max_conversations:
            oldest = min(self._data, key=lambda k: self._data[k][0])
            del self._data[oldest]


store = ConversationStore()
