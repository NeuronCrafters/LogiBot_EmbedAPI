from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
API_KEY = os.getenv("EMBEDDING_API_KEY", "")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"

app = FastAPI(title="Embedding Service")
model = SentenceTransformer(MODEL_NAME)

class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1)

def check_auth(authorization: str | None):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/embed")
def embed(payload: EmbedRequest, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="texts required")
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail="too many texts")
    embeddings = model.encode(texts, normalize_embeddings=NORMALIZE)
    return {"model": MODEL_NAME, "embeddings": embeddings.tolist()}