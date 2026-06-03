"""Plant Agent — conversational access to official mtapi2 plant indicators.

The agent is the interface; mtapi2 is the source of truth. This module never
computes indicator figures itself — it selects which mtapi2 function to call,
validates the call against the client's own device tree, executes it, and
narrates the returned values.
"""
