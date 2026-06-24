from pathlib import Path
import torchvision
from torchvision import models
import transformers
from transformers import AutoModel
from transformers import ViTModel, ViTConfig


BASE_ROOT = BASE_ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_DIR = BASE_ROOT / "csvs" / "kfold_heldout"
TEST_CSV = BASE_ROOT / "csvs" / "splits_heldout" / "test_heldout.csv"
SPECIES_DIRS = {
    "Falciparum": BASE_ROOT / "Falciparum" / "crops",
    "Vivax": BASE_ROOT / "Vivax" / "crops",
    "Ovale": BASE_ROOT / "Ovale" / "crops",
    "Malariae": BASE_ROOT / "Malariae" / "crops",
}
OUTPUT_DIR = BASE_ROOT / "results" / "tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = ["Falciparum", "Vivax", "Ovale", "Malariae"]
SPECIES_TO_ID = {
    "Falciparum": 0,
    "Malariae": 1,
    "Ovale": 2, 
    "Vivax": 3
}
NUM_SPECIES = 4

MODEL_CONFIG = {
    "ResNet50": {
        "image_size" : 224,
        "embedding_dim" : 2048,
        "source": "torchvision",
        "pretrained_name": "resnet50",
        "supports_lora": False,
        "lora_target_modules": [],
        "batch_size": 16,       # effective batch = 16 × 2 = 32
        "grad_accum_steps": 2
    },
    "ConvNeXt": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source":  "torchvision",
        "pretrained_name": "convnext_tiny",
        "supports_lora": False,
        "lora_target_modules": [],
        "batch_size": 16,
        "grad_accum_steps": 2
    },
    "Swin-T": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source":  "torchvision",
        "pretrained_name": "swin_t",
        "supports_lora": True,
        "lora_target_modules": ["qkv"],
        "batch_size": 16,
        "grad_accum_steps": 2
    },
    "ViT-B": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source": "huggingface",
        "pretrained_name": "google/vit-base-patch16-224-in21k",
        "supports_lora": True,
        "lora_target_modules": ["q_proj", "v_proj"],
        "batch_size": 8,        # full fine-tuning: attention map 12×197×197 per ogni layer
        "grad_accum_steps": 4   # effective batch = 8 × 4 = 32
    },
    "RedDino": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source": "timm",
        "pretrained_name": "hf-hub:Snarcy/RedDino-base",
        "supports_lora": True,
        "lora_target_modules": ["qkv"],
        "batch_size": 8,
        "grad_accum_steps": 4
    },
    "DinoBloom": {
        "image_size" : 518,
        "embedding_dim" : 768,
        "source": "timm",
        "pretrained_name": "hf-hub:1aurent/vit_base_patch14_224.dinobloom",
        "supports_lora": True,
        "lora_target_modules": ["qkv"],
        "batch_size": 4,        # 518px: input da ~1GB a batch=32; patch=14 → 1370 token
        "grad_accum_steps": 8   # effective batch = 4 × 8 = 32
    }
}

#IPERPARAMETRI
BATCH_SIZE = 16  # default di fallback; gli script usano MODEL_CONFIG[name]["batch_size"]
NUM_EPOCHS = 50 #con early stopping
LEARNING_RT_HEAD = 1e-3
LEARNING_RT_BACKBONE = 1e-5
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

#PARAMETRI LORA
LORA_R = 8 #rank delle matrici A e B, più è alto più parametri alleno
LORA_ALPHA = LORA_R #parametro di scaling
LORA_DROPOUT = 0.1 #dropout applicato agli adattatori LoRA
#lora target modules inseriti nel dizionario

FINE_TUNE_MODES = ["head_only", "full", "lora"]
INFORMATIVE_FOLDS = [1, 2]

