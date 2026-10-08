import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import os
from PIL import Image
from torchvision.transforms import transforms
from django.conf import settings

import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import os
from PIL import Image
from torchvision.transforms import transforms
from django.conf import settings

# 1. ARCHITECTURE DEFINITIONS (Required for PyTorch to load the .pth file)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# 2. THE AI SERVICE CLASS
class ObjectDetector:
    _model = None
    _classes = [
        "Apple_scab", "Apple_black_rot", "Apple_cedar_apple_rust", "Apple_healthy",
        "Background_without_leaves", "Blueberry_healthy", "Cherry_powdery_mildew",
        "Cherry_healthy", "Corn_gray_leaf_spot", "Corn_common_rust",
        "Corn_northern_leaf_blight", "Corn_healthy", "Grape_black_rot",
        "Grape_black_measles", "Grape_leaf_blight", "Grape_healthy",
        "Orange_haunglongbing", "Peach_bacterial_spot", "Peach_healthy",
        "Pepper_bacterial_spot", "Pepper_healthy", "Potato_early_blight",
        "Potato_healthy", "Potato_late_blight", "Raspberry_healthy",
        "Soybean_healthy", "Squash_powdery_mildew", "Strawberry_healthy",
        "Strawberry_leaf_scorch", "Tomato_bacterial_spot", "Tomato_early_blight",
        "Tomato_healthy", "Tomato_late_blight", "Tomato_leaf_mold",
        "Tomato_septoria_leaf_spot", "Tomato_spider_mites_two-spotted_spider_mite",
        "Tomato_target_spot", "Tomato_mosaic_virus", "Tomato_yellow_leaf_curl_virus"
    ]

    #@classmethod
    #def get_model(cls):
    #    if cls._model is None:
    #        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Using absolute path to find the model
    #        model_path = os.path.join(settings.BASE_DIR, 'core', 'services', 'models', 'plant-disease-model.pth')
            
            # Use torch.load for 'complete' models
    #        cls._model = torch.load(model_path, map_location=device)
    #        cls._model.eval()
    #    return cls._model
    @classmethod
    def get_model(cls):
        if cls._model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_path = os.path.join(
                settings.BASE_DIR,
                "core",
                "services",
                "models",
                "plant-disease-model.pth"
            )

            # Recreate architecture
            model = ResNet9(in_channels=3, num_diseases=38)

            # Load weights (state_dict)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            model.to(device)
            model.eval()

            cls._model = model

        return cls._model
    
    #@classmethod
    #def detect(cls, image_bytes):
    #    model = cls.get_model()
        
    #    transform = transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #    ])
        
    #    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    #    img_t = transform(image).unsqueeze(0)
        
        # Use no_grad for inference to save memory
    #    with torch.no_grad():
    #        out = model(img_t)
    #        probabilities = torch.nn.functional.softmax(out, dim=1)[0]
    #        conf, index = torch.max(probabilities, dim=0)
            
    #    return {
    #        "label": cls._classes[index.item()],
    #        "confidence": float(conf.item()) * 100
    #    }
    @classmethod
    def detect1(cls, image_bytes):
        model = cls.get_model()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_t = transform(image).unsqueeze(0)

        with torch.no_grad():
            out = model(img_t)
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            # Get the top 6 predictions
            top6_probabilities, top6_classes = torch.topk(probabilities, 1)
            classes_names = [
                "Apple_scab", "Apple_black_rot", "Apple_cedar_apple_rust", "Apple_healthy",
                "Background_without_leaves", "Blueberry_healthy", "Cherry_powdery_mildew",
                "Cherry_healthy", "Corn_gray_leaf_spot", "Corn_common_rust",
                "Corn_northern_leaf_blight", "Corn_healthy", "Grape_black_rot",
                "Grape_black_measles", "Grape_leaf_blight", "Grape_healthy",
                "Orange_haunglongbing", "Peach_bacterial_spot", "Peach_healthy",
                "Pepper_bacterial_spot", "Pepper_healthy", "Potato_early_blight",
                "Potato_healthy", "Potato_late_blight", "Raspberry_healthy",
                "Soybean_healthy", "Squash_powdery_mildew", "Strawberry_healthy",
                "Strawberry_leaf_scorch", "Tomato_bacterial_spot", "Tomato_early_blight",
                "Tomato_healthy", "Tomato_late_blight", "Tomato_leaf_mold",
                "Tomato_septoria_leaf_spot", "Tomato_spider_mites_two-spotted_spider_mite",
                "Tomato_target_spot", "Tomato_mosaic_virus", "Tomato_yellow_leaf_curl_virus"
            ]
            top6_classes = [classes_names[idx] for idx in top6_classes]

            top_prediction = top6_classes[0]
            conf=top6_probabilities[0]

            #conf, index = torch.max(probabilities, dim=0)

        return {
            "label": top6_classes,
            "confidence": float(conf) * 100
        }
    @classmethod
    def detect(cls, image_bytes):
        model = cls.get_model()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_t = transform(image).unsqueeze(0)

        with torch.no_grad():
            out = model(img_t)
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            # Get the top 6 predictions
            top6_probabilities, top6_classes = torch.topk(probabilities, 3)
            classes_names = [
                "Apple_scab", "Apple_black_rot", "Apple_cedar_apple_rust", "Apple_healthy",
                "Background_without_leaves", "Blueberry_healthy", "Cherry_powdery_mildew",
                "Cherry_healthy", "Corn_gray_leaf_spot", "Corn_common_rust",
                "Corn_northern_leaf_blight", "Corn_healthy", "Grape_black_rot",
                "Grape_black_measles", "Grape_leaf_blight", "Grape_healthy",
                "Orange_haunglongbing", "Peach_bacterial_spot", "Peach_healthy",
                "Pepper_bacterial_spot", "Pepper_healthy", "Potato_early_blight",
                "Potato_healthy", "Potato_late_blight", "Raspberry_healthy",
                "Soybean_healthy", "Squash_powdery_mildew", "Strawberry_healthy",
                "Strawberry_leaf_scorch", "Tomato_bacterial_spot", "Tomato_early_blight",
                "Tomato_healthy", "Tomato_late_blight", "Tomato_leaf_mold",
                "Tomato_septoria_leaf_spot", "Tomato_spider_mites_two-spotted_spider_mite",
                "Tomato_target_spot", "Tomato_mosaic_virus", "Tomato_yellow_leaf_curl_virus"
            ]

        
            results = []

            for prob, idx in zip(top6_probabilities, top6_classes):
                results.append({
                    "label": cls._classes[idx.item()],   # ONLY HERE convert index → label
                    "confidence": float(prob.item()) * 100
                })
        return {
            "top_predictions": results
        }
        
        
        
        
        

from core.services.rag.ingestion import run_ingestion
from core.services.rag.rag_chat import ask_question

rag_initialized = False


def rag_query(question, history=None):
    """Answer a question from the PDF knowledge base: {"answer": ..., "sources": [...]}.

    `history` is the list of [question, answer] pairs of the current user session,
    so follow-up questions can be rewritten into standalone ones.
    """
    global rag_initialized

    if not rag_initialized:
        run_ingestion()  # no-op when the index is already populated
        rag_initialized = True

    return ask_question(question, history)
