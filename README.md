# MTIA - Servicio ML/AI

Servicio central para tareas de ML/AI. Cada funcionalidad vive en su propio módulo dentro de `modules/`.

## Estructura

```
mtia/
├── api.py                          # Orquestador FastAPI (incluye routers de módulos)
├── modules/
│   └── stopreason/
│       ├── pipeline.py             # Pipeline ML (extract, train, predict CLI)
│       ├── router.py               # Endpoints FastAPI (POST /predict, GET /health)
│       ├── models/cic/             # Artefactos del modelo entrenado
│       └── data/                   # Datos de entrenamiento extraídos
├── Dockerfile
└── pyproject.toml
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Chequeo de salud del servicio |
| GET | `/stopreason/health` | Salud del módulo stop-reason |
| POST | `/stopreason/predict` | Predecir top-k razones de detención |

### Request de predicción

```json
POST /stopreason/predict
{
  "dev_id": 1079,
  "duration": 60.0,
  "top_k": 3,
  "user_id": null,
  "timestamp": "2026-01-29T14:52:27"
}
```

## CLI (pipeline stopreason)

Cuando una máquina se detiene en la línea de producción, el sistema predice la razón de detención más probable usando un clasificador LightGBM entrenado con datos históricos de detenciones. El pipeline tiene tres pasos: **extract** → **train** → **predict**.

### Extract

Extrae eventos históricos de detención desde PostgreSQL (`j_s_reg`, `j_device`, `j_cod_state`, `j_line`) y construye un vector de features por cada evento de detención:

- **Temporales:** hora, día de la semana, mes
- **Contexto del dispositivo:** ID del dispositivo, operador, duración de la detención
- **Historial:** 2 razones de detención anteriores, tiempo desde la última detención
- **Topología de línea:** posición en la línea, largo de la línea, y estado de 4 vecinos (2 aguas arriba, 2 aguas abajo) en una ventana de 5 minutos

Las razones de detención con menos de `--min-samples` ocurrencias (por defecto 10) se filtran. Genera un archivo Parquet.

```bash
docker compose exec mtia stopreason extract --client-id 31 --output /app/modules/stopreason/data/cic.parquet --verbose
```

### Train

Entrena un clasificador LightGBM multi-clase. Compara dos modelos: solo-dispositivo (9 features) vs dispositivo+línea (25 features con contexto de vecinos), y persiste el modelo dispositivo+línea (mejor accuracy top-3).

Artefactos guardados: `model.joblib`, `label_encoder.joblib`, `metadata.json`.

Resultados CIC: ~85% top-1, ~95% top-3, ~97% top-5 (69 clases).

```bash
docker compose exec mtia stopreason train --data /app/modules/stopreason/data/cic.parquet --output /app/modules/stopreason/models/cic --verbose
```

### Predict

Consulta la base de datos en tiempo real para obtener contexto (info del dispositivo, detenciones recientes, estado de vecinos), arma el mismo vector de 25 features usado durante el entrenamiento, y retorna las top-k predicciones con scores de confianza y descripciones de razones de detención.

Disponible como comando CLI y como endpoint `POST /stopreason/predict`.

```bash
docker compose exec mtia stopreason predict --model-dir /app/modules/stopreason/models/cic --dev-id 1079 --verbose
```

## Agregar un nuevo módulo

1. Crear `modules/newmodule/` con `__init__.py`, `pipeline.py`, `router.py`
2. En `api.py`, agregar: `from modules.newmodule import router` + `app.include_router(router)`
3. Agregar `"modules.newmodule"` a `packages` en `pyproject.toml`

---

## Módulo RAG — Documentos por paso de proceso

RAG local que responde preguntas de operadores sobre los manuales de un paso
productivo en mtworkflows. Todo corre local: Ollama nativo macOS (GPU Metal)
+ Qdrant + mtia + pestaña "Documentos" en el frontend.

### Arquitectura

```
mtworkflows (Vue) ──JWT── mtia (FastAPI) ──┬── Ollama (host, Metal)  ← embeddings + LLM
                                           └── Qdrant (docker)       ← vectores por cliente
```

Modelos por defecto (MacBook Pro M4 24GB):

| Rol         | Modelo            | Memoria aprox. |
|-------------|-------------------|----------------|
| LLM         | `gemma4:e4b`      | ~8 GB          |
| Embeddings  | `bge-m3` (1024d)  | ~1 GB          |

Alternativas: `gemma4:e2b` (más rápido, menos calidad), `gemma4:26b` MoE
(más calidad, justo en memoria).

### Configuración inicial (una sola vez)

1. **Instalar Ollama nativo** — https://ollama.com/download/mac
   Ollama corre como servicio launchd en `localhost:11434`.

2. **Descargar modelos** (desde el host, no desde el contenedor):
   ```bash
   invoke mtia.pull-models
   # o manualmente: ollama pull gemma4:e4b bge-m3
   ```

3. **Levantar el stack RAG** (Qdrant + mtia con las nuevas variables):
   ```bash
   invoke mtia.up
   ```

4. **Verificar salud**:
   ```bash
   invoke mtia.health
   # Debe mostrar OK en mtia, qdrant, ollama, y los dos modelos descargados.
   ```

### Ingestar un manual

```bash
invoke mtia.ingest --client cyt --blueprint empaque \
    --path /ruta/a/manual-empaque.pdf
```

Esto copia el PDF a `mtia/documents/cyt/empaque/` (gitignored) y lo indexa.
Re-ingestar el mismo PDF reemplaza los fragmentos anteriores (idempotente).

Para reindexar todo un cliente (borra la colección en Qdrant y reingesta
desde disco):

```bash
invoke mtia.reindex --client cyt
```

### Endpoints HTTP

Todos requieren `Authorization: JWT <token>` con el mismo token que emite
el servicio `api` (pyramid_jwt, HS512, `JWT_SECRET` compartido vía `.env`).
El claim `client` del token debe coincidir con el parámetro `client`.

| Método | Ruta                                                | Uso                        |
|--------|-----------------------------------------------------|----------------------------|
| GET    | `/rag/documents?client=&blueprint_code=`            | Listar docs de un paso     |
| POST   | `/rag/ingest` (multipart: client, blueprint_code, file) | Subir e indexar       |
| DELETE | `/rag/documents?client=&blueprint_code=&source=`    | Borrar doc + vectores      |
| POST   | `/rag/chat?client=&blueprint_code=&question=&top_k=` | **SSE** stream de respuesta |

El stream de `/rag/chat` emite tres tipos de eventos:

- `citations` — lista `[{source, page, chunk_idx, score}]`, una sola vez al inicio.
- `token` — `{text: "..."}` por cada chunk del LLM.
- `done` — `{}` al final.

### Frontend — pestaña "Documentos"

En `mtworkflows`, la pestaña se agrega automáticamente al abrir una orden. El
`blueprint_code` se deriva del nombre del paso con `slugify()`
(`composables/useRag.js`): "Control de calidad" → `control_de_calidad`.

El proxy Vite `/mtia/*` → `http://mtia:8008` evita problemas de CORS.

### Umbral de rechazo

Si ningún fragmento supera `MIN_RETRIEVAL_SCORE` (0.30 por defecto, similitud
coseno en `bge-m3`), el endpoint emite:

> "No encontré información relevante en los documentos."

En lugar de llamar al LLM y arriesgar alucinaciones. Ajustar en
`modules/rag/router.py` si se valida contra un corpus real.

### Layout en disco

```
mtia/documents/<client>/<blueprint_code>/<archivo>.pdf
```

El árbol completo está en `.gitignore` — los documentos contienen IP del cliente.

### Tests

```bash
docker compose exec mtia python -m pytest modules/rag/tests/
```

Los tests de `test_ollama.py` se saltean automáticamente si Ollama no está
corriendo en el host. Los demás usan Qdrant real y Ollama stub.

### Problemas frecuentes

| Síntoma                                              | Causa / solución                                                 |
|------------------------------------------------------|------------------------------------------------------------------|
| `invoke mtia.health` → `FAIL ollama`                 | Ollama no está corriendo en el host. `brew services start ollama` |
| `ollama CLI missing`                                 | Falta instalar Ollama nativo (no corre en Docker — se pierde Metal) |
| `401 Unauthorized` en `/rag/*`                       | Header `Authorization: JWT <token>` ausente o `JWT_SECRET` distinto entre api y mtia |
| `403 Forbidden`                                      | El `client` pedido no coincide con el claim `client` del JWT     |
| Respuesta dice "No encontré información relevante"   | Normal si el corpus no cubre la pregunta. Ajustar threshold o agregar docs |
| `gemma4:e4b` no responde bien en español técnico     | Probar `gemma4:26b` (ver tabla de modelos arriba) y reindexar no es necesario |
| Embeddings muy lentos                                | Primera llamada a Ollama carga el modelo (cold start, ~5-10s). Tirar un `ollama run bge-m3` para pre-cargar |

---

## Módulo Plant Agent — Indicadores de planta por lenguaje natural

Asistente conversacional que responde preguntas sobre los indicadores
**oficiales** de una planta (OEE, disponibilidad, desempeño, calidad,
detenciones, producción) en español. La clave: **el agente es la interfaz;
mtapi2 es la fuente de verdad**. El LLM nunca calcula ni inventa cifras —
elige qué función de mtapi2 llamar, con qué argumentos, y narra el resultado.

### Arquitectura

```
frontend ──JWT── mtia (FastAPI) ──┬── LLM (Ollama/vLLM, in-environment)  ← elige herramienta
   modules/plantagent             └── mtapi2 (XMLRPC :7777)              ← indicadores oficiales
                                              └── webservice ORM → postgresdb
```

- **Tool-calling**, no text-to-SQL: cada herramienta envuelve una función de
  mtapi2, valida los argumentos contra el alcance del request y ejecuta. Las
  comparaciones ("¿qué equipo afecta más el OEE?") se calculan en código a
  partir de cifras oficiales, no por el modelo.
- **Por nombre y por nodo**: el agente trabaja con **nombres** (no ids) y acepta
  un `node` que puede ser **equipo, línea, sección o planta**. Resuelve el árbol
  con nombres vía `mtapi2.devtree_named`; para un nodo agregado pasa la **lista
  de dev_ids** a la función de indicador (que ya agrega listas internamente).
- **Aislamiento**: el claim `client` del JWT se exige (igual que en `rag`), el
  `plant_id` se valida contra `getplants`, y `device_meta`/`device_names`
  filtran por cliente (ids ajenos se descartan).
- **mtia no necesita GPU**: solo orquesta. El motor de inferencia (Ollama/vLLM)
  es lo único que usa GPU; mtia lo alcanza por `OLLAMA_URL`/`LLM_URL`.

### Endpoints

Requieren `Authorization: JWT <token>` (mismo token que emite `api`).

| Método | Ruta                                              | Uso                        |
|--------|---------------------------------------------------|----------------------------|
| GET    | `/plantagent/health`                              | salud del módulo           |
| POST   | `/plantagent/chat?client=&plant_id=&question=`    | **SSE** stream de respuesta |

Eventos SSE: `tool` (`{name, args, period}` — un evento por llamada a mtapi2,
traza de auditoría de cada cifra), `token` (`{text}`), `done` (`{}`), `error`
(`{message}`). Códigos: `401` sin/JWT inválido, `403` `client` no coincide,
`404` `plant_id` ajeno al cliente, `503` mtapi2 no disponible.

### Catálogo de herramientas

El agente trabaja **por nombre** y casi todas las herramientas aceptan un
**`node`** (equipo, línea, sección o planta) y un **`period`** relativo. Para un
nodo agregado, la familia OEE pasa la lista de dev_ids al indicador (que agrega
internamente); `production` suma; `top_stops`/`rank_downtime`/`stops_detail`/
`sabana` usan el conjunto de equipos del nodo. Cada llamada emite un evento SSE
`tool` (traza de auditoría de la cifra).

| Herramienta | Parámetros | Qué responde / ejemplo |
|---|---|---|
| `oee` · `disponibilidad` · `rendimiento` · `calidad` · `cumplimiento` | `node`, `period` | el indicador del nodo (0–1). *"¿OEE de la Línea 2 esta semana?"* |
| `rank_oee` | `period` | equipos por OEE, peor primero. *"¿qué equipo afecta más el OEE?"* |
| `rank_downtime` | `node?`, `period` | equipos por tiempo total detenido, mayor primero. *"¿qué máquina estuvo más detenida ayer?"* |
| `top_stops` | `period`, `by`=`time`\|`count`, `node?` | motivos de detención agregados (tiempo o conteo). *"¿la detención más repetida/larga?"* |
| `stops_detail` | `node?`, `period`, `reason?` | lista cronológica de paradas con hora inicio/fin, duración y motivo. *"¿a qué hora fueron las detenciones programadas?"* |
| `production` | `node`, `period`, `measure` | producción del nodo (suma). *"¿cuánto produjo la Envasadora ayer?"* |
| `daily_oee` | `node`, `period` | OEE por día + mejor/peor día. *"¿el mejor día de la semana pasada?"* |
| `sabana` | `node?`, `period` | detalle de corridas: totales, producción por producto y muestra de filas. *"¿producción por producto de la Termoformadora 9 ayer?"* |

- **`period`**: `hoy`, `ayer`, `anteayer`, `esta semana`, `semana pasada`,
  `este mes`, `mes pasado`, `últimos N días`.
- **`measure`** (production): `standard` = contador × kp (default), `counter` =
  contador crudo, `tons` = toneladas.
- **`by`** (top_stops): `time` (detención más larga) o `count` (más repetida).
- **`reason`** (stops_detail): substring del motivo, p.ej. `PROGRAMADO`, `AVERIA`.

Notas:
- Tres herramientas de detención, distintas: `top_stops` (motivos agregados) ·
  `rank_downtime` (equipos por tiempo detenido) · `stops_detail` (cada parada
  con su horario).
- **Fuente de datos**: `top_stops`/`rank_downtime`/`stops_detail` leen `S_reg`
  (vía `pareto`/`stop_intervals`) y sirven para **cualquier cliente**; `sabana`
  requiere SabanaRows precalculados (clientes con el DAG sabana).
- `cumplimiento` es específico por cliente; si no está, responde "no disponible"
  en vez de fallar.
- `kp` es un **multiplicador de la especificación del producto**, no una unidad;
  la unidad de medida vive en `Device.unit` (la expone `mtapi2.device_meta`).
- Resolución por nombre/aislamiento: el árbol con nombres viene de
  `mtapi2.devtree_named`; `device_meta`/`device_names` filtran por cliente.

### Variables de entorno

| Var | Uso |
|---|---|
| `OLLAMA_URL` / `LLM_MODEL` | endpoint y modelo del LLM (compartido con RAG) |
| `MTAPI2_URL` | XMLRPC de mtapi2 (default `http://mtapi2:7777/api/xmlrpc`) |
| `JWT_SECRET` | mismo secreto que `api` (HS512) |
| `PLANTAGENT_DEFAULT_TZ` | tz de la planta para resolver "hoy/ayer/…" (default `America/Santiago`; resolver por planta es un pendiente) |

### Modelo: dev vs producción

- **Modelo objetivo: `gemma4:12b`.** Hace tool-calling fiable y resuelve bien
  nodos y períodos relativos. Requiere **Ollama ≥ 0.30.4** (el soporte de la
  12B unificada se agregó en 0.30.3 pero el manifest exige 0.30.4; usar el
  estable cuando salga, o el pre-release `v0.30.4-rcN`). `gemma4:e4b` sirve para
  *plumbing* pero su tool-calling es débil.
- **Apuntar mtia al modelo**: `LLM_MODEL=gemma4:12b` (ver Variables de entorno).
- **Producción (in-environment)**: nodo GPU propio (Linode/on-prem) con
  `gemma4:12b` vía Ollama o vLLM. El backend LLM se selecciona por env
  (`OLLAMA_URL`/`LLM_MODEL`), así que dev → GPU no cambian el código de mtia.

### Ejemplos

La forma más simple de probar es la CLI/TUI (dentro del contenedor; firma un
JWT desde `JWT_SECRET`):

```bash
# interactivo
docker compose exec -it mtia python -m modules.plantagent.tui \
    --client <cliente> --plant-id <id>

# una sola pregunta
docker compose exec -T mtia python -m modules.plantagent.tui \
    --client <cliente> --plant-id <id> \
    --question "¿qué máquina estuvo más tiempo detenida ayer?"
```

Preguntas que cubre (por nombre, cualquier nodo, períodos relativos):
- *¿qué equipo afecta más el OEE esta semana?* → `rank_oee`
- *¿qué máquina estuvo más tiempo detenida ayer?* → `rank_downtime`
- *¿OEE de la sección Químicos el mes pasado?* → `oee` (agrega la sección)
- *¿cuál fue el mejor día de la semana pasada para la Máquina de tostadores?* → `daily_oee`

Vía HTTP directo (SSE): `POST /plantagent/chat?client=&plant_id=&question=`
con `Authorization: JWT <token>`.

### Tests

```bash
docker compose exec mtia python -m pytest modules/plantagent/tests/
```

Los tests stubean **mtapi2 y el LLM** (sin red): periodos, scope, catálogo de
herramientas, loop de tool-calling, auth y forma del SSE.

### Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `ollama pull gemma4:12b` → `412 requires a newer version` | El manifest de la 12B exige Ollama ≥ 0.30.4. Actualizar a 0.30.4 estable (`brew upgrade --cask ollama-app`) o correr el pre-release `v0.30.4-rcN` (binario del `.tgz` de GitHub). |
| `llama-server binary not found` en macOS | La **fórmula** de Homebrew `ollama` viene incompleta. Usar la **app oficial** (`brew install --cask ollama-app` o ollama.com), no la fórmula. |
| `503` en `/plantagent/chat` | mtapi2 no responde. Verificar el contenedor `mtapi2`. |
| "no encontré '<nombre>'" | El nodo no existe o el nombre no coincide; el error lista los disponibles. Probá con el nombre exacto del equipo/línea/sección. |
| Respuestas con tool-calling poco fiables | `gemma4:e4b` es débil; usar `gemma4:12b` (`LLM_MODEL=gemma4:12b`, Ollama ≥ 0.30.4). |
