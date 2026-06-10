"""ConversationStore tests — TTL, turn cap, eviction; injected clock, no sleep."""
from __future__ import annotations

from modules.plantagent.memory import ConversationStore


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


def test_turn_cap_keeps_most_recent():
    store = ConversationStore(max_turns=2, clock=Clock())
    for i in range(4):
        store.append(KEY, f"q{i}", f"a{i}")

    history = store.history(KEY)
    assert len(history) == 4                      # 2 exchanges * 2 messages
    assert history[0]["content"] == "q2"          # oldest kept is the 3rd


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
