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
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_ood_experiment(model_name, fine_tune_mode, source_species, target_species):
    filtered_df = []
    
    for fold in range(1, 6):
        train_csv = SPLIT_DIR / f"fold{fold}_train.csv"
        df = pd.read_csv(train_csv)
        filtered_train = df[df["label"] == source_species]
        filtered_df.append(filtered_train)
    source_df = pd.concat(filtered_df, ignore_index=True)

    train_df, val_df = train_test_split(source_df, test_size=0.2, random_state=RANDOM_SEED)
    test = pd.read_csv(TEST_CSV)
    filtered_test = test[test["label"] == target_species]

    output_dir = OUTPUT_DIR / "ood" / f"{source_species}_to_{target_species}" / model_name / fine_tune_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_train_csv = output_dir / "temp_train.csv"
    temp_val_csv = output_dir / "temp_val.csv"
    temp_test_csv = output_dir / "temp_test.csv"

    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)
    filtered_test.to_csv(temp_test_csv, index=False)

    save_path = output_dir / "best_model.pt"

    train_transforms = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=True)
    train_dataset = MalariaDataset(temp_train_csv, train_transforms, SPECIES_TO_ID)
    train_dataloader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_transforms = get_transforms(image_size=MODEL_CONFIG[model_name]["image_size"], is_training=False)
    val_dataset = MalariaDataset(temp_val_csv, val_transforms, SPECIES_TO_ID)
    val_dataloader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_transforms = get_transforms(MODEL_CONFIG[model_name]["image_size"], is_training=False)
    test_dataset = MalariaDataset(temp_test_csv, test_transforms, SPECIES_TO_ID)
    test_dataloader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    labels = train_df["label"]
    label_ids = labels.map(SPECIES_TO_ID).values
    loss = get_loss_function(label_ids, DEVICE, NUM_SPECIES)
    

    model = build_model(
        model_name,
        fine_tune_mode,
        num_classes=NUM_SPECIES
    )
    model.to(DEVICE)
    

    history = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        loss_fn=loss,
        device=DEVICE,
        save_path=save_path,
        num_epochs=NUM_EPOCHS,
        fine_tune_mode=fine_tune_mode
    )

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    results = run_evaluation(model, test_dataloader, DEVICE, class_names=SPECIES)

    save_results(results, history, output_dir, model_name, fine_tune_mode)


if __name__ == "__main__":

    OOD_PAIRS = [("Falciparum", "Vivax"), ("Falciparum", "Ovale"),
        ("Falciparum", "Malariae"), ("Vivax", "Falciparum"),
        ("Vivax", "Ovale"), ("Vivax", "Malariae"),
        ("Ovale", "Falciparum"), ("Ovale", "Vivax"),
        ("Ovale", "Malariae")]
    
    for model_name in MODEL_CONFIG:
        for fine_tune_mode in FINE_TUNE_MODES:
            if fine_tune_mode == "lora" and not MODEL_CONFIG[model_name]["supports_lora"]:
                continue
            for source, target in OOD_PAIRS:
                print(f"\n>>> {model_name} | {fine_tune_mode} | {source} -> {target}")
                run_ood_experiment(model_name, fine_tune_mode, source, target)                


