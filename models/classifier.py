import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# =====================
# CLASSES + FEEDBACK
# =====================
CLASSES = ['not_noseprint', 'proper_noseprint', 'wrong_noseprint_blurry']

FEEDBACK = {
    'proper_noseprint':       'Noseprint captured successfully',
    'wrong_noseprint_blurry': 'Image is too blurry — hold steady and retake',
    'not_noseprint':          "This doesn't look like a nose — photograph your dog's nose directly"
}

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
# LOAD
# =====================
def load_classifier(model_path: str = 'weights/snoutscan_quality.pt'):
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 3)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"  ✅ Classifier loaded from {model_path}")
    return model


# =====================
# PREDICT
# =====================
def classify(model, image_bytes: bytes) -> dict:
    image  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = CLASSES[pred.item()]

    return {
        'class':      label,
        'confidence': round(conf.item(), 4),
        'feedback':   FEEDBACK[label],
        'accepted':   label == 'proper_noseprint'
    }