"""dato il nome di un modello e una modalità di fine-tuning,
restituire un modello PyTorch pronto per il training"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torchvision
from torchvision import models
import torch.nn as nn
import transformers
from transformers import AutoModel
import timm
from config import *

def load_backbone(model_name, model_cfg):
    if model_cfg["source"] == "torchvision":
        if model_name == "ResNet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Identity()
        elif model_name == "ConvNeXt":
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            model.classifier[2] = nn.Identity()
        elif model_name == "Swin-T":
            model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            model.head = nn.Identity()
    elif model_cfg["source"] == "huggingface":
        model = transformers.AutoModel.from_pretrained(model_cfg["pretrained_name"])
    elif model_cfg["source"] == "timm":
        model = timm.create_model(model_cfg["pretrained_name"], pretrained=True, num_classes=0)

    return model

class MalariaClassifier(nn.Module):

    def __init__(self, backbone, source, embedding_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.source = source

        head = nn.Sequential(
            nn.Linear(embedding_dim, num_classes)
        )

        self.head = head

    def forward(self, x_images):
        if self.source == "torchvision" or self.source == "timm":
            features = self.backbone(x_images)
            #self.head(features)
        elif self.source == "huggingface":
            outputs = self.backbone(pixel_values=x_images)
            features = outputs.last_hidden_state[:, 0, :]
            #self.head(token)
        return self.head(features)
    
    
def apply_lora(model, model_cfg):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r = LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=model_cfg["lora_target_modules"],
        bias="none"
    )
    mod_model = get_peft_model(model.backbone, lora_config)
    model.backbone = mod_model
    return model

def build_model(model_name, fine_tune_mode, num_classes=4):
    model_cfg = MODEL_CONFIG[model_name]

    if fine_tune_mode == "lora" and model_cfg["supports_lora"] == False:
        raise ValueError(f"WARNING: {model_name} does not support LoRA")

    backbone = load_backbone(model_name, model_cfg)
    model = MalariaClassifier(
        backbone=backbone,
        source=model_cfg["source"],
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=num_classes
    )

    if fine_tune_mode == "head_only":
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif fine_tune_mode == "full":
        pass
    elif fine_tune_mode == "lora":
        for p in model.backbone.parameters():
            p.requires_grad = False
        model = apply_lora(model, model_cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params
    
    print(f"Model: {model_name} | Mode: {fine_tune_mode}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({percentage:.2f}%)")

    return model
