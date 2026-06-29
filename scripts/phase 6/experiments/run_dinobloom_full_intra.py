"""DinoBloom + full fine-tuning, esperimenti intra-dataset (fold1 e fold2).
Script separato perché il loop principale (run_intra.py) ha un guard esplicito
che skippa questa combinazione in attesa di verifica VRAM.
VRAM verificata empiricamente: picco 5.16 GB su GTX 1060 6GB — feasible."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd

from config import (MODEL_CONFIG, SPLIT_DIR, TEST_CSV, SPECIES_TO_ID,
                    NUM_SPECIES, SPECIES, OUTPUT_DIR, NUM_EPOCHS, INFORMATIVE_FOLDS)
from data.dataset import MalariaDataset, get_transforms, get_dataloader
from models.build_model import build_model
from training.trainer import train_model
from training.losses import get_loss_function
from evaluation.evaluate import run_evaluation, save_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "DinoBloom"
FINE_TUNE_MODE = "full"


def run_intra_experiment(fold):
    output_dir = OUTPUT_DIR / "intra" / MODEL_NAME / FINE_TUNE_MODE / f"fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "metrics.json").exists():
        print(f"SKIP — {MODEL_NAME} | {FINE_TUNE_MODE} | fold{fold} già completato")
        return

    train_csv = SPLIT_DIR / f"fold{fold}_train.csv"
    val_csv = SPLIT_DIR / f"fold_{fold}_val.csv"

    train_data = pd.read_csv(train_csv)
    label_ids = train_data["label"].map(SPECIES_TO_ID).values
    loss = get_loss_function(label_ids, DEVICE, num_classes=NUM_SPECIES)

    batch_size = MODEL_CONFIG[MODEL_NAME]["batch_size"]
    grad_accum_steps = MODEL_CONFIG[MODEL_NAME]["grad_accum_steps"]
    image_size = MODEL_CONFIG[MODEL_NAME]["image_size"]

    train_dataset = MalariaDataset(train_csv, get_transforms(image_size, True), SPECIES_TO_ID)
    val_dataset = MalariaDataset(val_csv, get_transforms(image_size, False), SPECIES_TO_ID)
    test_dataset = MalariaDataset(TEST_CSV, get_transforms(image_size, False), SPECIES_TO_ID)

    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(MODEL_NAME, FINE_TUNE_MODE)
    model.to(DEVICE)

    save_path = output_dir / "best_model.pt"

    history = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        loss_fn=loss,
        device=DEVICE,
        save_path=save_path,
        num_epochs=NUM_EPOCHS,
        fine_tune_mode=FINE_TUNE_MODE,
        grad_accum_steps=grad_accum_steps,
    )

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    results = run_evaluation(model, test_dataloader, DEVICE, class_names=SPECIES)
    save_results(results, history, output_dir, MODEL_NAME, FINE_TUNE_MODE)


if __name__ == "__main__":
    print(f"=== DinoBloom | full | intra — {len(INFORMATIVE_FOLDS)} fold ===")
    for fold in INFORMATIVE_FOLDS:
        print(f"\n>>> {MODEL_NAME} | {FINE_TUNE_MODE} | fold{fold}")
        try:
            run_intra_experiment(fold)
        except Exception as e:
            import traceback
            print(f"ERRORE — fold{fold}: {e}")
            traceback.print_exc()
