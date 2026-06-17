import albumentations as A
from albumentations import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path

def get_transforms(image_size, is_training):
    if is_training == True:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225]),
            ToTensorV2(),
            
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225]),
            ToTensorV2(),
            
        ])
    

class MalariaDataset(Dataset):
    
    def __init__(self, csv_path, transform, label_to_id):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.label_to_id = label_to_id

        required_columns = {"filepath", "label", "phase", "group_id", "filename"}

        if  not required_columns.issubset(self.data.columns):
            missing = required_columns - set(self.data.columns)
            raise ValueError(f"{missing} missing!")
        
        missing_files = [p for p in self.data["filepath"] if not Path(p).exists()]
        if missing_files:
            print(f"Warning: {len(missing_files)} not found on disk")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        filepath = row["filepath"]
        img = Image.open(filepath).convert("RGB")
        img_np = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            img = augmented["image"]
        
        label = row["label"]
        label_id = self.label_to_id[label]

        group_id = row["group_id"]
        filename = row["filename"]
        phase = row["phase"]

        return img, label_id, label, phase, group_id, filename, filepath
    

def get_dataloader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )