"""dopo aver stabilito augmentation, learning rate e numero di epoche
creo il modello definitivo che usa tutto il trainval per sfruttare al massimo il numero di dati disponibili.
questo è il modello che verrà effettivametne testato"""

import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
TRAIN_CSV = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits_heldout\trainval_metadata.csv"
TEST_CSV = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits_heldout\test_heldout.csv"

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 #è giusto?
IMAGE_SIZE = 224
N_CLASSES = 4
N_EPOCHS = 3 #media delle best epoch ottenute con training senza test
result_dir = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\results\final_train_heldout"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

LABEL_TO_ID = {
    "Falciparum": 0,
    "Malariae": 1,
    "Ovale": 2,
    "Vivax": 3
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def geo_color_light():
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

#train dataset con augmentation
train_transform = geo_color_light()  

train_dataset = MPIDBDataset(
    csv_path=TRAIN_CSV,
    transform=train_transform
)

#cl<ss counts sul train
train_labels = train_dataset.data["label"].map(LABEL_TO_ID).values
class_counts = np.bincount(train_labels, minlength=N_CLASSES)


class_weights = np.zeros(N_CLASSES)

for class_id in range(N_CLASSES):
    if class_counts[class_id] > 0:
        class_weights[class_id] = 1.0 / class_counts[class_id]
    else:
        class_weights[class_id] = 0.0

print("Train class counts:", class_counts)
print("Class weights:", class_weights)
print("Train samples:", len(train_dataset))
print("Device:", DEVICE)

sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights = sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

model = models.resnet18(weights="IMAGENET1K_V1")

model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, N_CLASSES)
)

model = model.to(DEVICE)

loss_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss

history = []
for epoch in range(N_EPOCHS):
    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE
    )
    history.append({
        "epoch": epoch +1,
        "train_loss": train_loss
    })

    print(
        f"Final train | Epoch {epoch+1} / {N_EPOCHS} | "
        f"train_loss: {train_loss:.4f}"
    )


history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(result_dir, "final_train_history.csv"), index=False)

checkpoint_path = os.path.join(result_dir, "final_model_heldout.pth")

torch.save({
    "model_state_dict": model.state_dict(),
    "label_to_id": LABEL_TO_ID,
    "id_to_label": ID_TO_LABEL,
    "image_size": IMAGE_SIZE,
    "n_epochs": N_EPOCHS,
    "train_csv": TRAIN_CSV,
    "test_csv": TEST_CSV,
    "class_counts": class_counts.tolist()
}, checkpoint_path)

print(f"Final model saved in: {checkpoint_path}")