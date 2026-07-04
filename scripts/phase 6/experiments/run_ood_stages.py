"""Protocollo OOD cross-specie sul label space degli STADI del ciclo cellulare.

I due protocolli OOD precedenti (run_ood.py, run_ood_loso.py) usano come label
la specie: il modello impara "tutto = specie sorgente" e F1=0 su ogni run,
perché il label space (specie) non è condiviso in modo utile tra training e test.

Qui il label space condiviso è invece lo STADIO del ciclo cellulare
(R=ring, G=gametocyte, S=schizont, T=trophozoite), biologicamente presente
in tutte le specie:
- Training: campioni della specie sorgente (fold1_train + fold2_train), label = phase
- Validation: campioni della specie sorgente (fold1_val + fold2_val), label = phase
- Test: campioni della specie target (test_heldout.csv), label = phase

Solo Falciparum viene usato come sorgente: Vivax e Ovale hanno un solo
group_id ciascuno nell'intero dataset, quindi in uno split group-aware quel
gruppo finisce sempre interamente in train — fold1_val/fold2_val non
contengono MAI campioni di Vivax o Ovale (verificato sui CSV). Usarle come
sorgente lascerebbe il val_loader vuoto e romperebbe l'early stopping.
Malariae è escluso come da indicazione (stadio R assente nel training).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    MODEL_CONFIG, FINE_TUNE_MODES, INFORMATIVE_FOLDS,
    NUM_EPOCHS, SPLIT_DIR, TEST_CSV, OUTPUT_DIR, RANDOM_SEED
)
from data.dataset import MalariaDataset, get_transforms, get_dataloader
from models.build_model import build_model
from training.trainer import train_model
from training.losses import get_loss_function
from evaluation.evaluate import run_evaluation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGES_OUTPUT_DIR = OUTPUT_DIR / "ood_stages"

STAGE_TO_ID = {"R": 0, "G": 1, "S": 2, "T": 3}
STAGE_NAMES = ["R", "G", "S", "T"]
NUM_STAGES = 4

# Solo Falciparum ha sia i 4 stadi presenti in training sia campioni di
# validation reali in fold1_val/fold2_val (vedi analisi nel docstring).
SOURCE_TARGET_PAIRS = [
    ("Falciparum", "Vivax"),
    ("Falciparum", "Ovale"),
    ("Falciparum", "Malariae"),
]


def save_stage_results(results, history, output_dir, model_name, fine_tune_mode):
    """Replica save_results (evaluation/evaluate.py) ma con class_names=STAGE_NAMES
    per la confusion matrix, dato che save_results ha SPECIES hardcodato sugli assi."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(results["accuracy"]),
        "f1_macro": float(results["f1_macro"]),
        "mcc": float(results["mcc"]),
        "model_name": model_name,
        "fine_tune_mode": fine_tune_mode
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(results["classification_report"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        results["confusion_matrix"],
        annot=True,
        fmt="d",
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES,
        cmap="Blues"
    )
    plt.title(f"{model_name} — {fine_tune_mode}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"Results saved in {output_dir}")


def run_ood_stage_experiment(model_name, fine_tune_mode, source_species, target_species):
    output_dir = STAGES_OUTPUT_DIR / f"{source_species}_{target_species}" / model_name / fine_tune_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "metrics.json").exists():
        print(f"SKIP — {model_name} | {fine_tune_mode} | {source_species} -> {target_species} già completato")
        return

    train_dfs, val_dfs = [], []
    for fold in INFORMATIVE_FOLDS:
        df_tr = pd.read_csv(SPLIT_DIR / f"fold{fold}_train.csv")
        df_val = pd.read_csv(SPLIT_DIR / f"fold_{fold}_val.csv")
        train_dfs.append(df_tr[df_tr["label"] == source_species])
        val_dfs.append(df_val[df_val["label"] == source_species])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df[test_df["label"] == target_species].reset_index(drop=True)

    # Sovrascrive "label" con lo stadio (phase resta intatta per il check di
    # validazione delle colonne richieste da MalariaDataset).
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["label"] = train_df["phase"]
    val_df["label"] = val_df["phase"]
    test_df["label"] = test_df["phase"]

    stage_counts = train_df["phase"].value_counts().to_dict()
    print(f"\n=== SORGENTE: {source_species} -> TARGET: {target_species} | MODEL: {model_name} | MODE: {fine_tune_mode} ===")
    print(f"Training samples: {len(train_df)} ({source_species} only, label=stage)")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)} ({target_species} only, label=stage)")
    print(f"Stage distribution training: R={stage_counts.get('R', 0)} G={stage_counts.get('G', 0)} "
          f"S={stage_counts.get('S', 0)} T={stage_counts.get('T', 0)}")

    temp_train_csv = output_dir / "temp_train.csv"
    temp_val_csv = output_dir / "temp_val.csv"
    temp_test_csv = output_dir / "temp_test.csv"
    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)
    test_df.to_csv(temp_test_csv, index=False)

    label_ids = train_df["label"].map(STAGE_TO_ID).values
    loss = get_loss_function(label_ids, DEVICE, num_classes=NUM_STAGES)

    batch_size = MODEL_CONFIG[model_name]["batch_size"]
    grad_accum_steps = MODEL_CONFIG[model_name]["grad_accum_steps"]
    image_size = MODEL_CONFIG[model_name]["image_size"]

    train_transforms = get_transforms(image_size=image_size, is_training=True)
    train_dataset = MalariaDataset(temp_train_csv, train_transforms, STAGE_TO_ID)
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    val_transforms = get_transforms(image_size=image_size, is_training=False)
    val_dataset = MalariaDataset(temp_val_csv, val_transforms, STAGE_TO_ID)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    test_transforms = get_transforms(image_size=image_size, is_training=False)
    test_dataset = MalariaDataset(temp_test_csv, test_transforms, STAGE_TO_ID)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(model_name, fine_tune_mode, num_classes=NUM_STAGES)
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
        fine_tune_mode=fine_tune_mode,
        grad_accum_steps=grad_accum_steps
    )

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    results = run_evaluation(model, test_dataloader, DEVICE, class_names=STAGE_NAMES)

    save_stage_results(results, history, output_dir, model_name, fine_tune_mode)


if __name__ == "__main__":
    planned_runs = []
    for source, target in SOURCE_TARGET_PAIRS:
        for model_name in MODEL_CONFIG:
            for fine_tune_mode in FINE_TUNE_MODES:
                if fine_tune_mode == "lora" and not MODEL_CONFIG[model_name]["supports_lora"]:
                    continue
                if model_name == "DinoBloom" and fine_tune_mode == "full":
                    continue
                planned_runs.append((source, target, model_name, fine_tune_mode))

    print(f"\nTotale run pianificate: {len(planned_runs)}")
    print(f"Struttura output: {STAGES_OUTPUT_DIR}/{{sorgente}}_{{target}}/{{model}}/{{mode}}/")
    print("Contenuto per run: metrics.json, classification_report.txt, confusion_matrix.png, "
          "history.json, best_model.pt, temp_train.csv, temp_val.csv, temp_test.csv\n")

    for source, target, model_name, fine_tune_mode in planned_runs:
        try:
            run_ood_stage_experiment(model_name, fine_tune_mode, source, target)
        except Exception as e:
            import traceback
            print(f"ERRORE — {model_name} | {fine_tune_mode} | {source} -> {target}: {e}")
            traceback.print_exc()
