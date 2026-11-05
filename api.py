"""MTIA — central ML/AI service."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import create_engine

from modules.stopreason import load_model_artifacts, router as stopreason_router
from modules.rag.router import router as rag_router
from modules.plantagent.router import router as plantagent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auditoría: los módulos loguean a nivel INFO (preguntas y tool-calls del
    # plantagent). uvicorn no instala handler de root, así que sin esto solo
    # se verían WARNING+ (handler lastResort de Python).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Fail fast: sin JWT_SECRET todo endpoint autenticado daría 500 en el
    # primer request. Mejor que el contenedor no levante y el log lo diga.
    if not os.environ.get("JWT_SECRET"):
        raise RuntimeError(
            "JWT_SECRET no está definido en el entorno del contenedor mtia "
            "(agrégalo al .env del workspace y recrea con `invoke mtia.up`)")

    models_dir = os.environ.get("MODELS_DIR") or os.environ.get("MODEL_DIR")
    database_url = os.environ["DATABASE_URL"]

    app.state.engine = create_engine(database_url)

    # Load all client models from subdirectories
    client_models = {}
    if not models_dir:
        raise RuntimeError("Set MODELS_DIR (or MODEL_DIR) environment variable")
    models_path = Path(models_dir)

    if models_path.is_dir():
        for subdir in sorted(models_path.iterdir()):
            if subdir.is_dir() and (subdir / "model.joblib").exists():
                client_name = subdir.name.lower()
                client_models[client_name] = load_model_artifacts(str(subdir))
                n_classes = client_models[client_name]["metadata"]["n_classes"]
                print(f"Model loaded: {client_name} ({n_classes} classes)")

    app.state.client_models = client_models
    print(f"Clients loaded: {list(client_models.keys())}")
    yield


app = FastAPI(title="MTIA - ML/AI Service", lifespan=lifespan)
app.include_router(stopreason_router)
app.include_router(rag_router)
app.include_router(plantagent_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "mtia"}
