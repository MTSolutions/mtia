"""System prompt and message construction for the Plant Agent loop."""
from __future__ import annotations

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
    """Frame the question with the named device scope and reference time."""
    devs = "; ".join(
        "{} (id {})".format(d.get("name") or "?", d["id"]) for d in ctx.devices[:50])
    more = "" if len(ctx.devices) <= 50 else " (y más)"
    plant = ctx.plant_name or "planta {}".format(ctx.plant_id)
    return (
        "Pregunta: {q}\n\n"
        "Contexto: {plant}. Equipos disponibles: {devs}{more}.\n"
        "Refiérete a los equipos por su nombre. "
        "Fecha/hora de referencia: {now}."
    ).format(q=question, plant=plant, devs=devs, more=more, now=ctx.now.isoformat())
