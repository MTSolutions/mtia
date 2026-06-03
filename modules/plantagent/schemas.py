"""SSE event names for the Plant Agent chat stream.

Event payloads (JSON in the SSE `data:` field):
  - tool  : {"name", "args", "period"?, "error"?}  — one per tool call (audit trail)
  - token : {"text"}                               — incremental answer chunks
  - done  : {}                                      — end of stream
  - error : {"message"}                            — handled failure
"""
from __future__ import annotations

EVENT_TOOL = "tool"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"
