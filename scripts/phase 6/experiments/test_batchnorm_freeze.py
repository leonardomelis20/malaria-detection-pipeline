import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import torch.nn as nn

from config import *
from data.dataset import MalariaDataset, get_transforms, get_dataloader
from models.build_model import build_model
from training.trainer import train_one_epoch
from training.losses import get_loss_function



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ResNet50"
fold=1
batch_size = 8
model = build_model(model_name, "head_only").to(DEVICE)

bn=None
for m in model.backbone.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn = m
        break
assert bn is not None
    

rm_before = bn.running_mean.detach().clone()

train_csv = SPLIT_DIR / f"fold{fold}_train.csv"

train_transform = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=True)
train_dataset = MalariaDataset(train_csv, train_transform, SPECIES_TO_ID)
train_loader = get_dataloader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0

                            )

train_df = pd.read_csv(train_csv)
label_ids = train_df["label"].map(SPECIES_TO_ID).values

criterion=get_loss_function(label_ids, DEVICE, num_classes=NUM_SPECIES)
optimizer= torch.optim.AdamW(model.head.parameters(), lr=LEARNING_RT_HEAD, weight_decay=WEIGHT_DECAY)

train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

rm_after = bn.running_mean

if torch.allclose(rm_before, rm_after):
    print("FIX OK: running_mean invariato, backbone davvero congelato")
else:
    print("FIX NON ATTIVA: running_mean cambiato")
    print("delta:", (rm_after - rm_before).abs().max().item())