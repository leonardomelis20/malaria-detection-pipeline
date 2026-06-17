"""per ogni combinazione modello x mode x fold informativo:
1. carica i dati
2. costruisce il modello
3. allena il modello
4. valuta sul test set held-out
5. salva i risultati """

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_intra_experiment(model_name, fine_tune_mode, fold, num_epochs=NUM_EPOCHS, output_subdir="intra"):
    train_csv = SPLIT_DIR / f"fold{fold}_train.csv"
    val_csv = SPLIT_DIR / f"fold_{fold}_val.csv"
    

    train_data = pd.read_csv(train_csv)
    label = train_data["label"]
    label_ids = label.map(SPECIES_TO_ID).values

    loss = get_loss_function(label_ids, DEVICE, num_classes=NUM_SPECIES)

    train_transforms = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=True)
    train_dataset = MalariaDataset(train_csv, train_transforms, SPECIES_TO_ID)
    train_dataloader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_transforms = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=False)
    val_dataset = MalariaDataset(val_csv, val_transforms, SPECIES_TO_ID)
    val_dataloader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_transforms = get_transforms(MODEL_CONFIG[model_name]["image_size"], is_training=False)
    test_dataset = MalariaDataset(TEST_CSV, test_transforms, SPECIES_TO_ID)
    test_dataloader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(model_name, fine_tune_mode)
    model.to(DEVICE)

    output_dir = OUTPUT_DIR / "intra" / model_name / fine_tune_mode / f"fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "best_model.pt"

    
    history = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        loss_fn=loss,
        device=DEVICE,
        save_path=save_path,
        num_epochs=num_epochs,
        fine_tune_mode=fine_tune_mode
    )

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    results = run_evaluation(model, test_dataloader, DEVICE, class_names=SPECIES)

    save_results(results, history, output_dir, model_name, fine_tune_mode)


if __name__ == "__main__":
    for model_name in MODEL_CONFIG:
        for fine_tune_mode in FINE_TUNE_MODES:
            if fine_tune_mode == "lora" and not MODEL_CONFIG[model_name]["supports_lora"]:
                continue
            for fold in INFORMATIVE_FOLDS:
                print(f"\n>>> {model_name} | {fine_tune_mode} | fold{fold}")
                run_intra_experiment(model_name, fine_tune_mode, fold)