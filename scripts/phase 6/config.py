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
        "lora_target_modules": []
    },
    "ConvNeXt": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source":  "torchvision",
        "pretrained_name": "convnext_tiny",   
        "supports_lora": False,
        "lora_target_modules": []
    },
    "Swin-T": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source":  "torchvision",
        "pretrained_name": "swin_t",
        "supports_lora": True,
        "lora_target_modules": ["qkv"]
    },
    "ViT-B": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source": "huggingface",
        "pretrained_name": "google/vit-base-patch16-224-in21k",
        "supports_lora": True,
        "lora_target_modules": ["q_proj", "v_proj"]
    },
    "RedDino": {
        "image_size" : 224,
        "embedding_dim" : 768,
        "source": "timm",
        "pretrained_name": "hf-hub:Snarcy/RedDino-base",
        "supports_lora": True,
        "lora_target_modules": ["qkv"]
    },
    "DinoBloom": {
        "image_size" : 518,
        "embedding_dim" : 768,
        "source": "timm",
        "pretrained_name": "hf-hub:1aurent/vit_base_patch14_224.dinobloom",
        "supports_lora": True,
        "lora_target_modules": ["qkv"] 
    }
}

#IPERPARAMETRI
BATCH_SIZE = 32
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

