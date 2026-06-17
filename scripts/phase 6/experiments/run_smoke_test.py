"""
1) white box testing su singolo batch
2) black box testing su pipeline reale ridotta"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd

from config import *
from data.dataset import MalariaDataset, get_transforms, get_dataloader
from models.build_model import build_model
from training.trainer import train_model
from training.losses import get_loss_function
from evaluation.evaluate import run_evaluation, save_results
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 1
batch_size = 8
model_name = "ConvNeXt"
mode = "head_only"
fold = 1
image_size = MODEL_CONFIG[model_name]["image_size"]

#WHITE-BOX TESTING
train_csv = SPLIT_DIR / f"fold{fold}_train.csv"

train_transform = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=True)
train_dataset = MalariaDataset(train_csv, train_transform, SPECIES_TO_ID)
train_loader = get_dataloader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0

                            )

images, labels, *_ = next(iter(train_loader))

images, labels = images.to(DEVICE), labels.to(DEVICE)

assert images.shape == (batch_size, 3, image_size, image_size)
assert labels.min().item() >= 0 and labels.max().item() <= 3
assert torch.isnan(images).any() == False

train_df = pd.read_csv(train_csv)
print("distribuzione label nel batch:", torch.bincount(labels, minlength=4).tolist())

model = build_model(model_name=model_name,
                    fine_tune_mode=mode,
                    num_classes=4
)
model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"parametri trainable: {trainable} / {total}")

train_df = pd.read_csv(train_csv)
label_ids = train_df["label"].map(SPECIES_TO_ID).values
criterion = get_loss_function(label_ids, DEVICE, num_classes=NUM_SPECIES)

model.train()
logits = model(images)
assert logits.shape == (batch_size, 4)

loss = criterion(logits, labels)
assert torch.isfinite(loss) 
print(f"loss iniziale: {loss.item():.4f} | ln(4) = {math.log(4):.4f}")

param_back = next(iter(model.backbone.parameters()))
backbone_before = param_back.detach().clone()
param_head = next(iter(model.head.parameters()))
head_before = param_head.detach().clone()

optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=LEARNING_RT_HEAD,
            weight_decay=WEIGHT_DECAY
        )
optimizer.zero_grad()
loss.backward()
assert param_back.grad is None
assert param_head.grad is not None and param_head.grad.norm() > 0

optimizer.step()
assert torch.allclose(param_back, backbone_before) 
assert not torch.allclose(param_head, head_before) 

#BLACK-BOX TESTING
model = build_model(model_name=model_name, fine_tune_mode=mode, num_classes=4).to(DEVICE)

train_csv = SPLIT_DIR / f"fold{fold}_train.csv"
val_csv = SPLIT_DIR / f"fold_{fold}_val.csv"



train_df = pd.read_csv(train_csv)
label_ids = train_df["label"].map(SPECIES_TO_ID).values
criterion = get_loss_function(label_ids, DEVICE, num_classes=NUM_SPECIES)


train_transforms = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=True)
train_dataset = MalariaDataset(train_csv, train_transforms, SPECIES_TO_ID)
train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

val_transforms = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=False)
val_dataset = MalariaDataset(val_csv, val_transforms, SPECIES_TO_ID)
val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

test_transforms = get_transforms(MODEL_CONFIG[model_name]["image_size"], is_training=False)
test_dataset = MalariaDataset(TEST_CSV, test_transforms, SPECIES_TO_ID)
test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

output_dir = OUTPUT_DIR / "smoke_test" / model_name / mode / f"fold{fold}"
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / "best_model.pt"


history = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    device=DEVICE,
    save_path=save_path,
    num_epochs=num_epochs,
    fine_tune_mode=mode
)

results = run_evaluation(model, test_dataloader, device=DEVICE, class_names=SPECIES)
save_results(results, history, output_dir, model_name, mode)





