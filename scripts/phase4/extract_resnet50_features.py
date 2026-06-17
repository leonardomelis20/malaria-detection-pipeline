"""
PER OGNI FOLD:
leggi train CSV e val CSV
leggi held-out test CSV
carica i crop RBC
passarli dentro ResNet50 senza classifier
salva le feature in .h5"""


"""H5 -> HDF5
Hierarchical Data Format Version 5
Used to store and organize large, complex datasets efficiently"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import h5py
import tqdm
from pathlib import Path
import os

BASE_DIR = Path(
    r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis"
)

KFOLD_DIR = BASE_DIR / "csvs" / "kfold_heldout"
HELDOUT_DIR = BASE_DIR / "csvs" / "splits_heldout"
TEST_CSV = HELDOUT_DIR / "test_heldout.csv"

OUTPUT_DIR = Path(
    r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\results\features\resnet50"
)


N_FOLDS = 5
BATCH_SIZE = 32
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#label mapping
LABEL_TO_ID = {
    "Falciparum": 0,
    "Malariae": 1,
    "Ovale": 2,
    "Vivax": 3
}

required_columns = {"filepath", "label", "phase", "group_id", "filename"}

#trasformazioni immagini -> NON augmentation
def get_resnet50_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
    ])

class CropDataSet(Dataset):

    def __init__(self, csv_path: str, transform = None, label_to_id=None):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform
        self.label_to_id = label_to_id

        if not required_columns.issubset(self.data.columns):
            missing = required_columns - set(self.data.columns)
            raise ValueError(f"Colonne mancanti nel CSV {self.csv_path}: {missing}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = row["filepath"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        
        label_text = row["label"]
        label_id = self.label_to_id[label_text]
        group_id = row["group_id"]
        filename = row["filename"]
        phase = row["phase"]

        return image, label_id, label_text, phase, group_id, filename, image_path

#MODELLO
def build_resnet50_feature_extractor(device):
    #RESNET 50 FEATURE EXTRACTION
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()

    model.to(DEVICE)
    model.eval()
    return model

#FEATURES EXTRACTION
def extract_features(dataloader, model, device):
    
    model.eval()
    
    all_features = []
    all_label_ids = []
    all_label_texts = []
    all_phases = []
    all_group_ids = []
    all_filenames = []
    all_filepaths = []

    with torch.no_grad():
        for batch in dataloader:
            images, label_id, label_texts, phases, group_ids, filenames, filepaths = batch

            images = images.to(device)
            features = model(images)
            features = features.cpu().numpy()

            all_features.append(features)
            all_label_ids.append(label_id.numpy())

            all_label_texts.extend(label_texts)
            all_phases.extend(phases)
            all_group_ids.extend(group_ids)
            all_filenames.extend(filenames)
            all_filepaths.extend(filepaths)
    
    features_array = np.concatenate(all_features, axis=0)
    label_ids_array = np.concatenate(all_label_ids, axis=0)

    data = {
        "features": features_array,
        "label_ids": label_ids_array,
        "labels": np.array(all_label_texts),
        "phases": np.array(all_phases),
        "group_ids": np.array(all_group_ids),
        "filenames": np.array(all_filenames),
        "filepaths": np.array(all_filepaths),
    }

    return data

#SAVE H5
def save_h5(data, output_path):
    output_path = Path(output_path)
    #creo la cartella se non esiste
    output_path.parent.mkdir(parents=True, exist_ok=True)
   
    with h5py.File(output_path, "w") as h5f:

        #array numerici
        h5f.create_dataset("features", data=data["features"])
        h5f.create_dataset("label_ids", data=data["label_ids"])

        #stringhe
        h5f.create_dataset("labels", data=data["labels"].astype("S"))
        h5f.create_dataset("phases", data=data["phases"].astype("S"))
        h5f.create_dataset("group_ids", data=data["group_ids"].astype(str).astype("S"))
        h5f.create_dataset("filenames", data=data["filenames"].astype("S"))
        h5f.create_dataset("filepaths", data=data["filepaths"].astype("S"))

    print(f"Saved: {output_path}")

#VALIDATE H5
def validate_h5(output_path):
    output_path = Path(output_path)
    with h5py.File(output_path, "r") as h5f:
        #stampa shape di features
        features = h5f["features"][:]
        labels = h5f["labels"][:]
        label_ids = h5f["label_ids"][:]

        print(f"\nValidating: {output_path}")
        print("Features shape:", features.shape)
        print("Number of labels:", labels.shape[0])
        print("Number of label_ids:", label_ids.shape[0])
        print("Contains NaN:", np.isnan(features).any())

        if np.isnan(features).any():
            raise ValueError(f"NaN trovati in {output_path}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features e labels hanno numero diverso di campioni")
        if features.shape[1] != 2048:
            raise ValueError(f"Dimensione feature errata: {features.shape[1]}, atteso 2048")
        
    print("Validation ok")


#MAIN
def main():
    print("Device:", DEVICE)
    transform = get_resnet50_transform()
    model = build_resnet50_feature_extractor(DEVICE)

    for fold in range(1, N_FOLDS+1):
        print(f"\n========== FOLD {fold} ==========")
        train_csv = KFOLD_DIR / f"fold{fold}_train.csv"
        val_csv = KFOLD_DIR / f"fold_{fold}_val.csv"
        test_csv = TEST_CSV
        
        split_csvs = {
            "train": train_csv,
            "val": val_csv,
            "test": test_csv
        }

        fold_output_dir = OUTPUT_DIR /f"fold{fold}"

        for split_name, csv_path in split_csvs.items():
            print(f"\nSplit: {split_name}")
            print("CSV:", csv_path)

            dataset = CropDataSet(
                csv_path=csv_path,
                transform=transform,
                label_to_id=LABEL_TO_ID
            )

            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            print("Samples:", len(dataset))
            
            data = extract_features(dataloader, model, DEVICE)

            output_path = fold_output_dir / f"{split_name}.h5"
            save_h5(data, output_path)
            validate_h5(output_path)

if __name__ == "__main__":
    main()








