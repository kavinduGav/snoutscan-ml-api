import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import io
import numpy as np
import cv2

EMBEDDING_DIM = 512


# =====================
# IMAGE ENHANCEMENT
# =====================
def enhance_image(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    gaussian  = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    sharpened = cv2.addWeighted(img_bgr, 1.5, gaussian, -0.5, 0)

    lab     = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    result  = cv2.merge([l, a, b])
    result  = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    result  = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result)


# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =====================
# MODEL — LayerNorm instead of BatchNorm1d
# works correctly with batch size 1 at inference
# =====================
class NoseEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        backbone        = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        return self.embedding_head(self.backbone(x))


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
    image  = enhance_image(image)
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)
        embedding = F.normalize(embedding, dim=1)

    return embedding.squeeze().cpu().numpy().tolist()


# =====================
# AVERAGE 2 EMBEDDINGS FOR REGISTRATION
# =====================
def average_embeddings(embedding1: list, embedding2: list) -> list:
    e1  = np.array(embedding1)
    e2  = np.array(embedding2)
    avg = (e1 + e2) / 2.0
    avg = avg / np.linalg.norm(avg)
    return avg.tolist()