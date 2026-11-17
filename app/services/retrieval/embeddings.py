import os
import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings


BATCH_SIZE=50
PACE_SECONDS = 35
_GEMINI_DIM=3072
_FALLBACK_DIM=768 #BAAI/bge-base-en-v1.5

_active_model=None
_model_type: str | None =None  # gemini or local

# EMBEDDING_PROVIDER=local in .env forces the local model and skips the Gemini probe.
FORCE_LOCAL = os.getenv("EMBEDDING_PROVIDER", "gemini").lower() == "local"

def _probe_gemini():
    """Try one embed call to verify gemini is reachable , return model or None"""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            goggle_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query('probe')
        logfire.info('Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).') # logfire is an observability tool focused on developer experience
        return model
    except Exception as e:
        logfire.warning("gemini probe failed: {error}. will use sentence-transformers fallback.", error=str(e))

        return None
    
    
def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading local embeddings (BAAI/bge-base-en-v1.5,768-dim).")
    return SentenceTransformer('BAAI/bge-base-en-v1.5')

def _init():
    global _active_model,_model_type
    
    if _active_model is not None:
        return
    gemini=None if FORCE_LOCAL else _probe_gemini()
    if gemini:
        _active_model=gemini
        _model_type="gemini"
    else:
        _active_model=_load_fallback()
        _model_type="local"
      
def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type=='gemini' else _FALLBACK_DIM

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    # The API asked to retry in ~37s; wait 60s to be safe.
                    logfire.warning(f"Gemini rate limit hit — retrying in 60s (attempt {attempt + 1}/4).")
                    time.sleep(60)
                else:
                    logfire.error("Gemini embedding failed: {error}", error=str(e))
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")

    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type=='gemini':
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()
    
def embed_texts(texts:list[str]) -> list[list[float]]:
    _init()
    all_embeddings:list[list[float]] = []
    for i in range(0,len(texts),BATCH_SIZE):
        batch = texts[i: i+ BATCH_SIZE]
        with logfire.span('Embed batch',model=_model_type,start=i,size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
        is_last = (i + BATCH_SIZE) >= len(texts)
        if _model_type == "gemini" and not is_last:
            logfire.info(f"Rate-limit pacing: sleeping {PACE_SECONDS}s.")
            time.sleep(PACE_SECONDS)
    return all_embeddings     
