"""ConversationStore tests — replay budget, TTL, eviction; injected clock, no sleep."""
from __future__ import annotations

from modules.memory import ConversationStore


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


KEY = ("degasa", "tester", 7, "conv-1")


def test_history_replays_exchanges_in_order():
    store = ConversationStore(clock=Clock())
    store.append(KEY, "¿OEE hoy?", "El OEE es 87%.")
    store.append(KEY, "¿y ayer?", "Ayer fue 80%.")

    assert store.history(KEY) == [
        {"role": "user", "content": "¿OEE hoy?"},
        {"role": "assistant", "content": "El OEE es 87%."},
        {"role": "user", "content": "¿y ayer?"},
        {"role": "assistant", "content": "Ayer fue 80%."},
    ]


def test_unknown_key_is_empty_and_keys_are_isolated():
    store = ConversationStore(clock=Clock())
    store.append(KEY, "q", "a")

    assert store.history(("otrocliente", "tester", 7, "conv-1")) == []
    assert store.history(("degasa", "otrologin", 7, "conv-1")) == []


def test_replay_budget_drops_oldest_but_keeps_conversation():
    # Each exchange costs 20 chars; a 40-char budget replays only the last two.
    store = ConversationStore(max_chars=40, clock=Clock())
    for i in range(10):
        store.append(KEY, f"q{i}".ljust(10, "x"), f"a{i}".ljust(10, "x"))

    history = store.history(KEY)
    assert len(history) == 4                      # 2 exchanges * 2 messages
    assert history[0]["content"].startswith("q8")  # most recent two, in order
    assert history[2]["content"].startswith("q9")

    # Nothing was lost on write: widening the budget reveals the whole thread.
    store._max_chars = 10**6
    assert len(store.history(KEY)) == 20


def test_newest_exchange_always_replayed_even_over_budget():
    store = ConversationStore(max_chars=10, clock=Clock())
    store.append(KEY, "una pregunta larga", "una respuesta mucho más larga que el budget")

    assert len(store.history(KEY)) == 2           # never an empty memory


def test_ttl_expires_idle_conversations():
    clock = Clock()
    store = ConversationStore(ttl_s=10, clock=clock)
    store.append(KEY, "q", "a")

    clock.t = 9
    assert store.history(KEY)                     # still alive; access refreshes
    clock.t = 25
    assert store.history(KEY) == []


def test_conversation_cap_evicts_oldest():
    clock = Clock()
    store = ConversationStore(max_conversations=2, ttl_s=1000, clock=clock)
    for i, key in enumerate(["a", "b", "c"]):
        clock.t = float(i)
        store.append(("degasa", "tester", 7, key), "q", "a")

    assert store.history(("degasa", "tester", 7, "a")) == []   # evicted
    assert store.history(("degasa", "tester", 7, "c"))
