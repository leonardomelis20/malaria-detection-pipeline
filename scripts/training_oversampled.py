"""1. size normalization
2. basic geometric invariances
3. dropout / occlusion
4. reduce color dependence 
5. affine transformations
6. domain-specific
7. normaization

resolution -> geomtry -> occlusion ->
color -> domain variation -> normalization"""

"""SEZIONE 1: config (path, batch, ecc.)
SEZIONE 2: transforms (baseline)
SEZIONE 3: dataset
SEZIONE 4: dataloader
SEZIONE 5: preview/debug
SEZIONE 6: modello
SEZIONE 7: training loop
SEZIONE 8: salvataggio risultati"""

from random import shuffle
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import numpy as np
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import copy



OVERSAMPLED_DIR = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\oversampled_folds"
KFOLD_DIR = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\kfold_heldout"
VAL_CSV = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\kfold"

IMAGE_SIZE = 224
BATCH_SIZE = 32
N_CLASSES = 4
N_FOLDS = 5

LABEL_TO_ID = {
    "Falciparum": 0,
    "Malariae": 1,
    "Ovale": 2,
    "Vivax": 3
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

#pipeline base
def get_baseline_train_trans():
    pipeline_aug = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    return pipeline_aug

def get_baseline_val_trans():
    pipeline_aug = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    return pipeline_aug

#https://arxiv.org/pdf/2303.01178
#policy 1 -> geometrica leggera

def get_train_trans_geo_light():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    

#policy 2 -> geometria + colore leggero
def get_train_trans_geo_color_light():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.3
        ),

        A.Normalize(),
        ToTensorV2()
    ])

#policy 3 -> geomtria + rumore leggero
def get_train_trans_geo_noise_light():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),

        A.GaussNoise(
            var_limit=(5.0, 20.0),
            p=0.2
        ),

        A.Normalize(),
        ToTensorV2()
    ])

def get_train_trans_oversampling():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.SquareSymmetry(p=0.5),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,0.5]),
        ToTensorV2()
    ])
#dataset custom

class MPIDBDataset(Dataset):
  
    def __init__(self, csv_path: str, transforms=None):
        #leggi csv_path con pandas
        self.data = pd.read_csv(csv_path)
        self.transforms = transforms

        required_columns = {"filepath", "label", "group_id", "filename"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError("CSV must contain filepath, label, group_id, filename")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = row["filepath"]
        label = row["label"]
    

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        #converti immagine nel formato corretto per albumentations (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        
        label = LABEL_TO_ID[label]
        
        return image, label
    



   
#controllo di alcune immagini
"""def preview_samples(dataset, n=5):
    for i in range(min(n, len(dataset))):
        image, label = dataset[i]

        #image è un tensore: C x H x W
        image_np = image.permute(1, 2, 0).numpy()

        #riporto in range visibile dopo normalizazzione
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        plt.imshow(image_np)
        plt.title(ID_TO_LABEL[label])
        plt.axis('off')
        plt.show()

preview_samples(train_dataset, n=5)

for images, labels in train_loader:
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    print("Labels:", labels)
    break"""

#validazione della predizione
def validate(model, val_loader, criterion, device):
    model.eval() #modello in modalità valutazione

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad(): #non aggiornare i pesi del modello
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images) #passa le immagini al modello -> output logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1) #trova la classe predetta dal modello

            all_labels.extend(labels.cpu().numpy()) #aggiunge label alla lista convertendo tensor -> array numpy
            all_preds.extend(preds.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        """macro -> media F1 per classe (tutte pesate uguali)
        zero_divsion= 0 -> evita erorri se una classe non è mai predetta"""
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        """importante perchè ho classi sbilanciate  (Falciparum >> Ovale/Vivax)"""

        return val_loss, accuracy, macro_f1, balanced_acc, all_labels, all_preds
    

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #inizializzazione dei gradienti

        output= model(images)
        loss = criterion(output, labels) #predictions, targets

        loss.backward() #calcolo delle derivate di ogni argo,ento del computational graph
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        """loss.item() -> converte tensore loss in numero Python
        images.size(0) -> batch size
        """

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

#TRAINING LOOP BASELINE
#inizializzazione modello

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 4
N_EPOCHS = 10
LEARNING_RATE = 1e-4

#confusion matrix aggregata
all_y_true = []
all_y_pred = []
all_fold_results = []

#report informativi
informative_y_true = []
informative_y_pred = []
per_class_reports = []
informative_fold_results =[]




result_dir = os.path.join("results", "heldout_5fold_oversampling")
os.makedirs(result_dir, exist_ok=True)

for fold in range(1, N_FOLDS+1):
#for fold in range(1, N_FOLDS+1):

    print(f"\n===== FOLD {fold} =====")

     # crea TRAIN_CSV = percorso fold_id_train.csv
    TRAIN_CSV = os.path.join(
        OVERSAMPLED_DIR,
        f"fold{fold}_train_oversampled.csv"
    )

    VAL_CSV = os.path.join(
        KFOLD_DIR,
        f"fold_{fold}_val.csv"
    )


    
    train_transforms = get_train_trans_oversampling()
    val_transforms = get_baseline_val_trans()

    train_dataset = MPIDBDataset(
        csv_path=TRAIN_CSV,
        transforms=train_transforms
    )
    val_dataset = MPIDBDataset(
        csv_path=VAL_CSV,
        transforms=val_transforms
    )

   


    #creazione dei dataloader
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE, #??
        shuffle=True
    )

    val_loader = DataLoader(
        dataset= val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = models.resnet18(weights="IMAGENET1K_V1") #weights da modificare
    """dataset piccolo -> modello leggero
    output = F(x) + x -> training più stabile, gradienti migliori, funziona bene anche senza tuning pesante
    """
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, N_CLASSES)
    )
    """input -> vettore di features
    output: logits per 4 classi"""

    """altra opzione da valutare:
    Conv -> ReLU -> Pool
    Conv -> ReLU -> Pool
    FC -> Output
    """
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    """calcola la cross entropy loss (differenza tra distribuzione di predicted probability
    e distribuzione reale con valori tra 0 e 1)
    input: Tensor con K>1 per casi K-dimesnionali
    target: o classifiche indici nel range [0, C) o probabilità per ogni classe"""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #grad-cam
    best_macro_f1 = -1
    best_model_state = None
    best_epoch_number = None
    #LOOP COMPLETO
    history = []
    for epoch in range(N_EPOCHS):
        train_loss =  train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE
        )

        val_loss, val_accuracy, val_macro_f1, val_balanced_acc, y_true, y_pred = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch_number = epoch +1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy": val_balanced_acc,
            "y_true": y_true,
            "y_pred": y_pred
        }

        history.append(row)

        print(
            f"Fold {fold} | Epoch {epoch + 1}/{N_EPOCHS} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"acc: {val_accuracy:.4f} | "
            f"macro_f1: {val_macro_f1:.4f} | "
            f"bal_acc: {val_balanced_acc:.4f}"
        )

    history_for_csv = [
    {k: v for k, v in row.items() if k not in ["y_true", "y_pred"]}
    for row in history
    ]

    history_df = pd.DataFrame(history_for_csv)

    history_file = os.path.join(result_dir, f"fold_{fold}_results.csv")
    history_df.to_csv(history_file, index=False)

    #salvo modello migliore per grad-cam
    checkpoint_path = os.path.join(result_dir, f"best_model_fold_{fold}.pth")

    torch.save({
        "fold": fold,
        "epoch": best_epoch_number,
        "model_state_dict": best_model_state,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "image_size": IMAGE_SIZE
    }, checkpoint_path)

    print(f"saved best model for fold {fold}: {checkpoint_path}")

    # Migliore epoca del fold
    best_epoch = max(history, key=lambda x: x["val_macro_f1"])
    best_epoch["fold"] = fold

    #report per classe
    y_true_best = best_epoch["y_true"]
    y_pred_best = best_epoch["y_pred"]

    present_classes = sorted(set(y_true_best))
    present_class_names = [ID_TO_LABEL[c] for c in present_classes]

    report = classification_report(
        y_true_best,
        y_pred_best,
        labels=[0,1,2,3],
        target_names=[ID_TO_LABEL[i] for i in [0,1,2,3]],
        zero_division=0,
        output_dict=True
    )
   
    for class_id in present_classes:
        class_name = ID_TO_LABEL[class_id]

        per_class_reports.append({
            "fold" : fold,
            "class" : class_name,
            "precision" : report[class_name]["precision"],
            "recall" : report[class_name]["recall"],
            "f1_score" : report[class_name]["f1-score"],
            "support" : report[class_name]["support"]
        })
    # Accumula predizioni della best epoch per confusion matrix aggregata
    #utilizzo solo fold 1 e fold2 in quanto gli unici che contengono tutti i gruppi
    val_labels_present = set(best_epoch["y_true"])

    all_y_true.extend(y_true_best)
    all_y_pred.extend(y_pred_best)


    informative_y_true.extend(y_true_best)
    informative_y_pred.extend(y_pred_best)
    informative_fold_results.append(best_epoch.copy())

    # Salva best epoch senza liste y_true/y_pred
    best_epoch_for_csv = best_epoch.copy()
    del best_epoch_for_csv["y_true"]
    del best_epoch_for_csv["y_pred"]

    all_fold_results.append(best_epoch_for_csv)

per_class_df = pd.DataFrame(per_class_reports)

per_class_file = os.path.join(result_dir, "per_class_metrics_by_fold.csv")
per_class_df.to_csv(per_class_file, index=False)

per_class_summary = (
    per_class_df
    .groupby("class")
    .agg(
        mean_precision=("precision", "mean"),
        std_precision=("precision", "std"),
        mean_recall=("recall", "mean"),
        std_recall=("recall", "std"),
        mean_f1=("f1_score", "mean"),
        std_f1=("f1_score", "std"),
        total_support=("support", "sum"),
        n_fold_evaluated=("fold", "nunique")
    )
    .reset_index()
)
per_class_summary_file = os.path.join(result_dir, "per_class_summary.csv")
per_class_summary.to_csv(per_class_summary_file, index=False)
cm = confusion_matrix(all_y_true, 
                      all_y_pred,
                      labels = [0,1,2,3])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ID_TO_LABEL.values(), yticklabels=ID_TO_LABEL.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix Aggregata")
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

np.savetxt(os.path.join(result_dir, "confusion_matrix.csv"), cm, delimiter=",")
    
cm_informative = confusion_matrix(
    informative_y_true, 
    informative_y_pred,
    labels=[0,1,2,3]
)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_informative,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ID_TO_LABEL.values(),
    yticklabels=ID_TO_LABEL.values()
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Fold Informativi")
plt.savefig(os.path.join(result_dir, "confusion_matrix_informative.png"), dpi=300, bbox_inches="tight")
plt.show()

np.savetxt(
    os.path.join(result_dir, "confusion_matrix_informative.csv"),
    cm_informative,
    delimiter=","
)
final_results_df = pd.DataFrame(all_fold_results)

summary = {
    "augmentation": "oversampling_square_symmetry_elastic",
    "mean_val_accuracy": final_results_df["val_accuracy"].mean(),
    "std_val_accuracy": final_results_df["val_accuracy"].std(),
    "mean_val_macro_f1": final_results_df["val_macro_f1"].mean(),
    "std_val_macro_f1": final_results_df["val_macro_f1"].std(),
    "mean_val_balanced_accuracy": final_results_df["val_balanced_accuracy"].mean(),
    "std_val_balanced_accuracy": final_results_df["val_balanced_accuracy"].std()
}

summary_df = pd.DataFrame([summary])

summary_file = os.path.join(result_dir, "summary_5fold.csv")
summary_df.to_csv(summary_file, index=False)

best_epochs_file = os.path.join(result_dir, "best_epochs_5fold.csv")
final_results_df.to_csv(best_epochs_file, index=False)

informative_results_df = pd.DataFrame([
    {k: v for k, v in row.items() if k not in ["y_true", "y_pred"]}
    for row in informative_fold_results
])

informative_summary = {
    "augmentation": "oversampling_square_symmetry_elastic",
    "folds_included": list(informative_results_df["fold"]),
    "mean_val_accuracy": informative_results_df["val_accuracy"].mean(),
    "std_val_accuracy": informative_results_df["val_accuracy"].std(),
    "mean_val_macro_f1": informative_results_df["val_macro_f1"].mean(),
    "std_val_macro_f1": informative_results_df["val_macro_f1"].std(),
    "mean_val_balanced_accuracy": informative_results_df["val_balanced_accuracy"].mean(),
    "std_val_balanced_accuracy": informative_results_df["val_balanced_accuracy"].std()
}

informative_summary_df = pd.DataFrame([informative_summary])
informative_summary_df.to_csv(
    os.path.join(result_dir, "summary_informative_folds.csv"),
    index=False
)