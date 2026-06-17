import torch
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

TEST_CSV = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits_heldout\test_heldout.csv"

CHECKPOINT = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\results\final_train_oversampled_heldout\final_model_oversampled_heldout.pth"

RESULT_DIR = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\results\final_train_oversampled"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

N_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_TO_ID = {
    "Falciparum": 0,
    "Malariae": 1,
    "Ovale": 2,
    "Vivax": 3
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def test_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

class MPIDBDataset(Dataset):

    def __init__(self, csv_path: str, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

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
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = LABEL_TO_ID[label]

        return image, label
    
transformation = test_transform()
test_dataset = MPIDBDataset(
    csv_path=TEST_CSV,
    transform=transformation
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, N_CLASSES)
)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)


#valutazione
all_labels = []
all_preds = []
all_probs = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

#calcolo metriche
accuracy = accuracy_score(all_labels, all_preds)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    all_preds
)

report = classification_report(
    all_labels,
    all_preds,
    labels=[0,1,2,3],
    target_names=[ID_TO_LABEL[i] for i in [0,1,2,3]],
    zero_division=0,
    output_dict=True
)

cm = confusion_matrix(all_labels, 
                      all_preds,
                      labels = [0,1,2,3])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ID_TO_LABEL.values(), yticklabels=ID_TO_LABEL.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Held-out Test Matrix")
plt.savefig(os.path.join(RESULT_DIR, "heldout_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

np.savetxt(os.path.join(RESULT_DIR, "heldout_confusion_matrix.csv"), cm, delimiter=",")
   


summary = {
    "test_accuracy": accuracy,
    "test_macro_f1": macro_f1,
    "test_balanced_accuracy": balanced_accuracy,
    "n_test_samples": len(test_dataset)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(RESULT_DIR, "heldout_test_summary.csv"), index=False)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(RESULT_DIR, "heldout_classification_report.csv"))

print("\n===== HELD-OUT TEST RESULTS =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=[0,1,2,3],
    target_names=[ID_TO_LABEL[i] for i in [0,1,2,3]],
    zero_division=0
))

print("\nConfusion matrix:")
print(cm)

print("Test samples:", len(test_dataset))
print("Device:", DEVICE)