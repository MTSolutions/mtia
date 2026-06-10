"""In-process conversation memory for the Plant Agent chat.

Stores only the prose exchanges (question, final answer) — never the tool
transcript: replaying tool calls would blow the num_ctx budget and the figures
go stale anyway (the system prompt forces fresh tool calls for any number).

Keys carry the JWT identity (client, login) besides the caller-chosen
conversation_id, so a guessed id never leaks another tenant's conversation.
The store is a plain dict guarded by the single asyncio event loop; entries
expire by TTL and the total conversation count is capped (oldest evicted).
Persistence across restarts is out of scope (frontend follow-up spec).
"""
from __future__ import annotations

import os
import time
from collections import deque

# Exchanges replayed per conversation. Each one costs ~question+answer tokens
# out of the same NUM_CTX budget as the tool catalog, so keep it modest.
MAX_TURNS = int(os.environ.get("PLANTAGENT_MEMORY_TURNS", "5"))
TTL_S = int(os.environ.get("PLANTAGENT_MEMORY_TTL_S", "1800"))
MAX_CONVERSATIONS = int(os.environ.get("PLANTAGENT_MEMORY_MAX", "500"))


class ConversationStore:
    def __init__(self, max_turns: int = MAX_TURNS, ttl_s: float = TTL_S,
                 max_conversations: int = MAX_CONVERSATIONS,
                 clock=time.monotonic):
        self._max_turns = max_turns
        self._ttl_s = ttl_s
        self._max_conversations = max_conversations
        self._clock = clock
        # key -> [last_used, deque[(question, answer)]]
        self._data: dict = {}

    def history(self, key) -> list[dict]:
        """Prior exchanges as chat messages, oldest first."""
        self._prune()
        entry = self._data.get(key)
        if entry is None:
            return []
        entry[0] = self._clock()
        messages = []
        for question, answer in entry[1]:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

    def append(self, key, question: str, answer: str) -> None:
        self._prune()
        entry = self._data.get(key)
        if entry is None:
            entry = self._data[key] = [self._clock(), deque(maxlen=self._max_turns)]
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
