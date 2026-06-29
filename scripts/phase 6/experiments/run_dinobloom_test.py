"""Test isolato DinoBloom full: verifica che full fine-tuning a 518px stia nei 6GB di VRAM."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd

from config import (MODEL_CONFIG, SPLIT_DIR, TEST_CSV, SPECIES_TO_ID,
                    NUM_SPECIES, SPECIES, OUTPUT_DIR)
from data.dataset import MalariaDataset, get_transforms, get_dataloader
from models.build_model import build_model
from training.trainer import train_model
from training.losses import get_loss_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "DinoBloom"
mode = "full"
fold = 1

cfg = MODEL_CONFIG[model_name]
batch_size = cfg["batch_size"]
grad_accum = cfg["grad_accum_steps"]

print(f"Device: {DEVICE}")
print(f"Batch size: {batch_size} | grad_accum_steps: {grad_accum} | effective batch: {batch_size * grad_accum}")
print(f"Image size: {cfg['image_size']}px")
print()

train_csv = SPLIT_DIR / f"fold{fold}_train.csv"
val_csv   = SPLIT_DIR / f"fold_{fold}_val.csv"

train_df = pd.read_csv(train_csv)
label_ids_arr = train_df["label"].map(SPECIES_TO_ID).values
loss_fn = get_loss_function(label_ids_arr, DEVICE, NUM_SPECIES)

train_ds = MalariaDataset(train_csv, get_transforms(cfg["image_size"], True),  SPECIES_TO_ID)
val_ds   = MalariaDataset(val_csv,   get_transforms(cfg["image_size"], False), SPECIES_TO_ID)
train_dl = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
val_dl   = get_dataloader(val_ds,   batch_size=batch_size, shuffle=False)

print(f"Campioni train: {len(train_ds)} | val: {len(val_ds)}")
print("Carico il modello...")

model = build_model(model_name, mode, num_classes=4)
model.to(DEVICE)

if DEVICE == "cuda":
    print(f"VRAM dopo caricamento modello: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

output_dir = OUTPUT_DIR / "dinobloom_test" / model_name / mode / f"fold{fold}"
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / "best_model.pt"

history = train_model(
    model=model,
    train_loader=train_dl,
    val_loader=val_dl,
    loss_fn=loss_fn,
    device=DEVICE,
    save_path=save_path,
    num_epochs=2,
    fine_tune_mode=mode,
    grad_accum_steps=grad_accum,
)

if DEVICE == "cuda":
    print(f"VRAM picco: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("Test DinoBloom completato senza OOM.")
