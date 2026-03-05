import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models.classifier import load_classifier, classify
from models.embedding import load_embedding_model, generate_embedding, average_embeddings

# =====================
# FLAGS
# flip to True once snoutscan_backbone.pt is trained
# =====================
EMBEDDING_MODEL_READY = False

# =====================
# LOAD MODELS AT STARTUP
# =====================
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    ml_models['classifier'] = load_classifier('weights/snoutscan_quality.pt')

    if EMBEDDING_MODEL_READY:
        ml_models['embedding'] = load_embedding_model('weights/snoutscan_backbone.pt')
    else:
        ml_models['embedding'] = None
        print("  ⏳ Embedding model not ready — /embed and /register disabled")

    print("✅ API ready\n")
    yield
    ml_models.clear()


# =====================
# APP
# =====================
app = FastAPI(
    title       = "SnoutScan ML API",
    description = "Dog noseprint quality classifier + embedding generator",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


# =====================
# ROUTES
# =====================
@app.get("/")
def root():
    return {
        "service":         "SnoutScan ML API",
        "status":          "running",
        "classifier":      "ready",
        "embedding_model": "ready" if EMBEDDING_MODEL_READY else "not ready"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# =====================
# CLASSIFY
# =====================
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classifies image as proper_noseprint, wrong_noseprint_blurry or not_noseprint.
    Call this before /embed or /register.

    Returns:
        accepted:   bool
        class:      str
        confidence: float
        feedback:   str (show this to the user)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large — max 10MB")

    try:
        return classify(ml_models['classifier'], image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# =====================
# EMBED — single image embedding
# =====================
@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    """
    Generates 512-dim embedding from a validated noseprint.
    Your Node backend stores this in dog_embeddings table with the dog_id.

    Returns:
        embedding:  list[float] (512 dimensions)
        dimensions: int
    """
    if not EMBEDDING_MODEL_READY or ml_models['embedding'] is None:
        raise HTTPException(status_code=503, detail="Embedding model not ready yet")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()

    try:
        embedding = generate_embedding(ml_models['embedding'], image_bytes)
        return {"embedding": embedding, "dimensions": len(embedding)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


# =====================
# REGISTER — 2 image registration
# =====================
@app.post("/register")
async def register_dog(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    Takes 2 noseprint images, averages their embeddings into 1 vector.
    Your Node backend should:
      1. Call /classify on both images first
      2. Only call /register if both are accepted
      3. Store returned embedding in dog_embeddings with the dog_id

    Returns:
        embedding:  list[float] (512 dimensions)
        dimensions: int
    """
    if not EMBEDDING_MODEL_READY or ml_models['embedding'] is None:
        raise HTTPException(status_code=503, detail="Embedding model not ready yet")

    bytes1 = await image1.read()
    bytes2 = await image2.read()

    try:
        emb1 = generate_embedding(ml_models['embedding'], bytes1)
        emb2 = generate_embedding(ml_models['embedding'], bytes2)
        avg  = average_embeddings(emb1, emb2)
        return {"embedding": avg, "dimensions": len(avg)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


# =====================
# RUN LOCALLY
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)