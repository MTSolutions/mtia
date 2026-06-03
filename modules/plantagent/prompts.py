"""System prompt and message construction for the Plant Agent loop."""
from __future__ import annotations

from modules.plantagent import scope
from modules.plantagent.tools import ToolContext


SYSTEM_PROMPT = (
    "Eres Plant Agent, un asistente que responde preguntas sobre los indicadores "
    "OFICIALES de una planta industrial: OEE, disponibilidad, desempeño, calidad, "
    "detenciones y producción.\n"
    "Reglas:\n"
    "- Usa SIEMPRE las herramientas disponibles para obtener cifras. NUNCA "
    "inventes, estimes ni calcules números por tu cuenta.\n"
    "- Si una herramienta no devuelve datos o falla, dilo con claridad; no "
    "rellenes con suposiciones.\n"
    "- Para preguntas comparativas ('qué equipo afecta más el OEE', 'el peor "
    "equipo'), usa la herramienta de clasificación (rank_oee) en vez de pedir "
    "el indicador equipo por equipo.\n"
    "- Para 'producción vs plan' combina la herramienta de producción con la de "
    "cumplimiento. La atraso/desviación por orden no está disponible aún.\n"
    "- Responde en español neutro (sin voseo), de forma breve y precisa, "
    "indicando el equipo y el período consultado.\n"
)


def build_user_message(question: str, ctx: ToolContext) -> str:
    """Frame the question with the named topology (equipment/lines/sections)."""
    nodes = scope.nodes_in(ctx.tree) if ctx.tree else [
        {"type": "dev", "name": d.get("name")} for d in ctx.devices]
    by_type: dict = {}
    for n in nodes:
        if n.get("name"):
            by_type.setdefault(n.get("type"), []).append(n["name"])

    def fmt(node_type: str, label: str) -> str:
        names = by_type.get(node_type, [])
        return "{}: {}".format(label, ", ".join(names[:50])) if names else ""

    parts = [p for p in (fmt("dev", "Equipos"),
                         fmt("line", "Líneas"),
                         fmt("section", "Secciones")) if p]
    plant = ctx.plant_name or "planta {}".format(ctx.plant_id)
    return (
        "Pregunta: {q}\n\n"
        "Contexto: {plant}. {parts}.\n"
        "Refiérete a equipos, líneas, secciones o la planta por su nombre. "
        "Fecha/hora de referencia: {now}."
    ).format(q=question, plant=plant, parts="; ".join(parts), now=ctx.now.isoformat())
