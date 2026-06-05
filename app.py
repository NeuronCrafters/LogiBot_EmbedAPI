from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from uuid import uuid4
import logging
import os
import time

RAW_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, RAW_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("embedding_service")
logger.setLevel(LOG_LEVEL)

if not hasattr(logging, RAW_LOG_LEVEL):
    logger.warning("invalid_log_level value=%s fallback=INFO", RAW_LOG_LEVEL)


def get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("invalid_int_env name=%s value=%s fallback=%s", name, value, default)
        return default

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
API_KEY = os.getenv("EMBEDDING_API_KEY", "")
MAX_BATCH_SIZE = get_int_env("MAX_BATCH_SIZE", 64)
NORMALIZE = get_bool_env("EMBEDDING_NORMALIZE", True)
LOG_TEXTS = get_bool_env("LOG_TEXTS", False)
LOG_TEXT_PREVIEW_CHARS = get_int_env("LOG_TEXT_PREVIEW_CHARS", 120)

app = FastAPI(title="Embedding Service")
logger.info(
    "service_configured model=%s max_batch_size=%s normalize=%s auth_enabled=%s log_texts=%s",
    MODEL_NAME,
    MAX_BATCH_SIZE,
    NORMALIZE,
    bool(API_KEY),
    LOG_TEXTS,
)

try:
    model_load_started_at = time.perf_counter()
    logger.info("model_loading_started model=%s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info(
        "model_loading_finished model=%s duration_ms=%.2f",
        MODEL_NAME,
        (time.perf_counter() - model_load_started_at) * 1000,
    )
except Exception:
    logger.exception("model_loading_failed model=%s", MODEL_NAME)
    raise

class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    started_at = time.perf_counter()
    client = request.client.host if request.client else "unknown"

    logger.info(
        "request_started request_id=%s method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        client,
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            (time.perf_counter() - started_at) * 1000,
        )
        raise

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_finished request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started_at) * 1000,
    )
    return response

def check_auth(authorization: str | None, request_id: str):
    if not API_KEY:
        logger.debug("auth_skipped request_id=%s reason=no_api_key_configured", request_id)
        return
    if API_KEY and authorization != f"Bearer {API_KEY}":
        logger.warning(
            "auth_failed request_id=%s has_authorization_header=%s",
            request_id,
            bool(authorization),
        )
        raise HTTPException(status_code=401, detail="unauthorized")
    logger.debug("auth_succeeded request_id=%s", request_id)

@app.get("/health")
def health(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.debug("health_checked request_id=%s model=%s", request_id, MODEL_NAME)
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/embed")
def embed(
    payload: EmbedRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        "embed_requested request_id=%s incoming_count=%s max_batch_size=%s normalize=%s",
        request_id,
        len(payload.texts),
        MAX_BATCH_SIZE,
        NORMALIZE,
    )
    check_auth(authorization, request_id)

    texts = [t.strip() for t in payload.texts if t and t.strip()]
    empty_count = len(payload.texts) - len(texts)
    if not texts:
        logger.warning(
            "embed_rejected request_id=%s reason=texts_required incoming_count=%s empty_count=%s",
            request_id,
            len(payload.texts),
            empty_count,
        )
        raise HTTPException(status_code=400, detail="texts required")
    if len(texts) > MAX_BATCH_SIZE:
        logger.warning(
            "embed_rejected request_id=%s reason=too_many_texts valid_count=%s max_batch_size=%s",
            request_id,
            len(texts),
            MAX_BATCH_SIZE,
        )
        raise HTTPException(status_code=413, detail="too many texts")

    text_lengths = [len(text) for text in texts]
    logger.info(
        "embed_processing request_id=%s valid_count=%s empty_count=%s min_chars=%s max_chars=%s avg_chars=%.2f",
        request_id,
        len(texts),
        empty_count,
        min(text_lengths),
        max(text_lengths),
        sum(text_lengths) / len(text_lengths),
    )

    if LOG_TEXTS:
        previews = [text[:LOG_TEXT_PREVIEW_CHARS] for text in texts]
        logger.info("embed_text_previews request_id=%s previews=%r", request_id, previews)

    encode_started_at = time.perf_counter()
    try:
        embeddings = model.encode(texts, normalize_embeddings=NORMALIZE)
    except Exception:
        logger.exception(
            "embed_failed request_id=%s valid_count=%s normalize=%s",
            request_id,
            len(texts),
            NORMALIZE,
        )
        raise HTTPException(status_code=500, detail="embedding failed")

    shape = getattr(embeddings, "shape", ())
    dimensions = shape[1] if len(shape) > 1 else 0
    logger.info(
        "embed_completed request_id=%s valid_count=%s dimensions=%s duration_ms=%.2f",
        request_id,
        len(texts),
        dimensions,
        (time.perf_counter() - encode_started_at) * 1000,
    )
    return {"model": MODEL_NAME, "embeddings": embeddings.tolist()}
