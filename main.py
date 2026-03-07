import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pillow_heif import register_heif_opener

from models.classifier import classify, load_classifier
from models.embedding import average_embeddings, generate_embedding, load_embedding_model

# =====================
# FLAGS
# flip to True once snoutscan_backbone.pt is trained
# =====================
EMBEDDING_MODEL_READY = True

# =====================
# LOAD MODELS AT STARTUP
# =====================
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading models...')

    ml_models['classifier'] = load_classifier('weights/snoutscan_quality.pt')

    if EMBEDDING_MODEL_READY:
        ml_models['embedding'] = load_embedding_model('weights/snoutscan_backbone.pt')
    else:
        ml_models['embedding'] = None
        print('  Embedding model not ready - /embed and /register disabled')

    print('API ready')
    yield
    ml_models.clear()


# =====================
# APP
# =====================
register_heif_opener()

app = FastAPI(
    title='SnoutScan ML API',
    description='Dog noseprint quality classifier + embedding generator',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/webp', 'image/heif', 'image/heic'}
MAX_FILE_BYTES = 10 * 1024 * 1024
REGISTRY_FILE = Path('data/registered_dogs.json')


def _load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {'dogs': []}

    with REGISTRY_FILE.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'dogs' not in data or not isinstance(data['dogs'], list):
        raise ValueError('Registry file is invalid')

    return data


def _save_registry(data: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_FILE.open('w', encoding='utf-8') as f:
        json.dump(data, f)


def _find_name_index(records: list, name: str) -> int:
    name_lower = name.lower()
    for i, row in enumerate(records):
        if isinstance(row, dict) and row.get('name', '').lower() == name_lower:
            return i
    return -1


def _validate_upload(file: UploadFile, image_bytes: bytes) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f'Invalid file type: {file.content_type}')

    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail='File too large - max 10MB')


def _enforce_quality(image_bytes: bytes, image_label: str) -> None:
    try:
        quality = classify(ml_models['classifier'], image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Quality classification failed: {str(e)}')

    if not quality.get('accepted', False):
        quality_class = quality.get('class', 'unknown')
        reason = 'quality_failed'
        message = f'{image_label} failed quality validation'
        guidance = 'Retake the image with clear focus and centered noseprint.'

        if quality_class == 'not_noseprint':
            reason = 'not_dog_nose_image'
            message = f'{image_label} is not recognized as a dog nose image'
            guidance = "Capture your dog's nose directly and fill most of the frame."
        elif quality_class == 'wrong_noseprint_blurry':
            reason = 'low_quality_nose_image'
            message = f'{image_label} is a dog nose image but quality is too low'
            guidance = 'Retake in good light and hold steady to avoid blur.'

        raise HTTPException(
            status_code=422,
            detail={
                'error_type': reason,
                'message': message,
                'guidance': guidance,
                'quality': quality,
            },
        )


# =====================
# ROUTES
# =====================
@app.get('/')
def root():
    return {
        'service': 'SnoutScan ML API',
        'status': 'running',
        'classifier': 'ready',
        'embedding_model': 'ready' if EMBEDDING_MODEL_READY else 'not ready',
    }


@app.get('/health')
def health():
    return {'status': 'ok'}


# =====================
# CLASSIFY
# =====================
@app.post('/classify')
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _validate_upload(file, image_bytes)

    try:
        return classify(ml_models['classifier'], image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Classification failed: {str(e)}')


# =====================
# EMBED — single image embedding
# =====================
@app.post('/embed')
async def embed_image(file: UploadFile = File(...)):
    if not EMBEDDING_MODEL_READY or ml_models['embedding'] is None:
        raise HTTPException(status_code=503, detail='Embedding model not ready yet')

    image_bytes = await file.read()
    _validate_upload(file, image_bytes)
    _enforce_quality(image_bytes, 'Image')

    try:
        embedding = generate_embedding(ml_models['embedding'], image_bytes)
        return {'embedding': embedding, 'dimensions': len(embedding)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Embedding failed: {str(e)}')


# =====================
# REGISTER — 2 image registration
# =====================
@app.post('/register')
async def register_dog(
    name: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    if not EMBEDDING_MODEL_READY or ml_models['embedding'] is None:
        raise HTTPException(status_code=503, detail='Embedding model not ready yet')

    if not name.strip():
        raise HTTPException(status_code=400, detail='Name is required')

    bytes1 = await image1.read()
    bytes2 = await image2.read()

    _validate_upload(image1, bytes1)
    _validate_upload(image2, bytes2)
    _enforce_quality(bytes1, 'image1')
    _enforce_quality(bytes2, 'image2')

    try:
        emb1 = generate_embedding(ml_models['embedding'], bytes1)
        emb2 = generate_embedding(ml_models['embedding'], bytes2)
        avg = average_embeddings(emb1, emb2)

        registry = _load_registry()
        dogs = registry['dogs']

        row = {'name': name.strip(), 'embedding': avg}
        existing_idx = _find_name_index(dogs, name.strip())

        if existing_idx >= 0:
            dogs[existing_idx] = row
        else:
            dogs.append(row)

        _save_registry(registry)

        return {
            'name': name.strip(),
            'embedding': avg,
            'dimensions': len(avg),
            'saved_to': str(REGISTRY_FILE),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Registration failed: {str(e)}')


# =====================
# IDENTIFY — match uploaded image against registered dogs
# =====================
@app.post('/identify')
async def identify_dog(
    file: UploadFile = File(...),
    threshold: float = Form(0.75),
):
    if not EMBEDDING_MODEL_READY or ml_models['embedding'] is None:
        raise HTTPException(status_code=503, detail='Embedding model not ready yet')

    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail='Threshold must be between 0 and 1')

    try:
        registry = _load_registry()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Registry read failed: {str(e)}')

    dogs = registry.get('dogs', [])
    if len(dogs) == 0:
        raise HTTPException(status_code=404, detail='No registered dogs found')

    image_bytes = await file.read()
    _validate_upload(file, image_bytes)
    _enforce_quality(image_bytes, 'Image')

    try:
        probe = np.array(generate_embedding(ml_models['embedding'], image_bytes), dtype=np.float32)

        best_name = None
        best_score = -1.0

        for row in dogs:
            if not isinstance(row, dict) or 'name' not in row or 'embedding' not in row:
                continue

            candidate = np.array(row['embedding'], dtype=np.float32)
            score = float(np.dot(probe, candidate))

            if score > best_score:
                best_score = score
                best_name = row['name']

        if best_name is None:
            raise HTTPException(status_code=500, detail='No valid registered embeddings found')

        return {
            'matched': best_score >= threshold,
            'name': best_name if best_score >= threshold else None,
            'similarity': round(best_score, 4),
            'threshold': threshold,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Identification failed: {str(e)}')


# =====================
# RUN LOCALLY
# =====================
if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)