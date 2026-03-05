import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import io
import numpy as np

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

EMBEDDING_DIM = 512


# =====================
# MODEL ARCHITECTURE
# must match train_embedding_model.py exactly
# =====================
class NoseEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        features   = self.backbone(x)
        embeddings = self.embedding_head(features)
        return embeddings


# =====================
# LOAD
# =====================
def load_embedding_model(model_path: str = 'weights/snoutscan_backbone.pt'):
    model = NoseEmbeddingModel(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"  ✅ Embedding model loaded from {model_path}")
    return model


# =====================
# GENERATE EMBEDDING
# =====================
def generate_embedding(model, image_bytes: bytes) -> list:
    image  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)
        embedding = F.normalize(embedding, dim=1)   # normalise for cosine similarity

    return embedding.squeeze().cpu().numpy().tolist()  # 512-dim list — store in postgres


# =====================
# AVERAGE 2 EMBEDDINGS FOR REGISTRATION
# call this when user registers with 2 images
# =====================
def average_embeddings(embedding1: list, embedding2: list) -> list:
    e1  = np.array(embedding1)
    e2  = np.array(embedding2)
    avg = (e1 + e2) / 2.0

    # Renormalise after averaging
    avg = avg / np.linalg.norm(avg)
    return avg.tolist()