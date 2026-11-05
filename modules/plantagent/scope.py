"""Plant/client scoping — validate a plant belongs to the JWT client and
resolve the set of device ids beneath a tree node.

mtapi2's flat helpers (``getdevs``/``getlines``) return *all* of a client's
devices/lines, not a hierarchical slice, so the authoritative way to get the
devices under a plant/line/section is ``devtree``, which we walk for ``dev``
nodes. ``getplants`` is the source of truth for which plants a client owns.
"""
from __future__ import annotations

import datetime as dt
import difflib
import re
from typing import Callable

from modules.plantagent import mtapi


class PlantNotFound(ValueError):
    """The requested plant_id does not belong to the client."""


def validate_plant(client: str, plant_id: int, mtapi_call: Callable = mtapi.call) -> dict:
    """Return the plant dict for ``plant_id`` if it belongs to ``client``.

    Raises:
        PlantNotFound: the plant is not among the client's plants.
    """
    plants = mtapi_call("getplants", client)
    for plant in plants:
        if plant.get("id") == plant_id:
            return plant
    raise PlantNotFound(
        "plant_id {!r} no pertenece al cliente {!r}".format(plant_id, client))


def _collect_dev_ids(node: dict) -> list[int]:
    """Depth-first collect ids of every ``type == 'dev'`` node in a devtree."""
    ids: list[int] = []
    if node.get("type") == "dev":
        ids.append(node["id"])
    for child_key in ("plants", "lines", "sections", "devs"):
        for child in node.get(child_key, []):
            ids.extend(_collect_dev_ids(child))
    return ids


def device_ids(
    client: str,
    node_type: str,
    node_id: int,
    start: dt.datetime,
    end: dt.datetime,
    mtapi_call: Callable = mtapi.call,
) -> list[int]:
    """Resolve the device ids under a tree node (plant/line/section/dev).

    Uses ``devtree`` with no indicators (cheap — the dates do not affect the
    device set) and walks the result for ``dev`` nodes.
    """
    tree = mtapi_call("devtree", client, start, end, node_type, node_id, [], False)
    return _collect_dev_ids(tree)


def named_tree(client: str, plant_id: int, mtapi_call: Callable = mtapi.call) -> dict:
    """The plant's name-annotated configuration tree (via mtapi2 devtree_named).

    Lightweight (no dates/indicators) and fault-free. Caller must have already
    validated that plant_id belongs to the client (devtree_named is not
    client-scoped) — see validate_plant.
    """
    return mtapi_call("devtree_named", client, "plant", plant_id)


def _collect_named_devs(node: dict, acc: list[dict]) -> None:
    if node.get("type") == "dev":
        acc.append({"id": node["id"], "name": node.get("name")})
    for child_key in ("plants", "lines", "sections", "devs"):
        for child in node.get(child_key, []):
            _collect_named_devs(child, acc)


def devices_in(tree: dict) -> list[dict]:
    """Flat ``[{id, name}]`` of every device in a named tree."""
    acc: list[dict] = []
    _collect_named_devs(tree, acc)
    return acc


def _collect_all_nodes(node: dict, acc: list[dict]) -> None:
    if node.get("type") and node.get("id") is not None:
        acc.append(node)
    for child_key in ("plants", "lines", "sections", "devs"):
        for child in node.get(child_key, []):
            _collect_all_nodes(child, acc)


def nodes_in(tree: dict) -> list[dict]:
    """Every node (plant/line/section/dev) in a named tree, flat."""
    acc: list[dict] = []
    _collect_all_nodes(tree, acc)
    return acc


def resolve_node(tree: dict, ref) -> dict | None:
    """Resolve a reference (id, numeric string, or name) to a tree node.

    Matches across ALL node types (device/line/section/plant). Names match
    case-insensitively, exact first then substring. Returns the node subtree
    (so the caller can read its devices via devices_in), or None.
    """
    if ref is None:
        return None
    nodes = nodes_in(tree)
    s = str(ref).strip()
    if s.lstrip("-").isdigit():
        nid = int(s)
        return next((n for n in nodes if n["id"] == nid), None)
    low = s.lower()
    for n in nodes:
        if (n.get("name") or "").strip().lower() == low:
            return n
    for n in nodes:
        if low and low in (n.get("name") or "").strip().lower():
            return n
    # Typo-tolerant fallback (models misspell names, e.g. 'escobillenos') —
    # but digits are identity ('Línea 9' must NOT match 'Línea 1'), so the
    # numbers in the query and the candidate must agree.
    candidates = {(n.get("name") or "").strip().lower(): n
                  for n in nodes if n.get("name")}
    close = difflib.get_close_matches(low, list(candidates), n=1, cutoff=0.8)
    if close and re.findall(r"\d+", low) == re.findall(r"\d+", close[0]):
        return candidates[close[0]]
    return None


def node_device_ids(node: dict) -> list[int]:
    """Device ids under a resolved node (a device node returns just itself)."""
    return [d["id"] for d in devices_in(node)]
