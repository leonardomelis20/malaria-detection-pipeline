"""esegue il loop di training e validation,
gestisce eary stopping, salva il best model"""

import torch
from torch.utils.data import DataLoader
from training.losses import get_loss_function
from config import *

class EarlyStopping():
    """tiene traccia della validation loss e
    decide quando fermare il training"""

    def __init__(self, patience, min_delta):

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter == self.patience:
                self.should_stop = True
                print(f"WARNING: {self.counter} epochs with no progress, stopping")

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:

        images, label_ids, _, _, _, _, _ = batch

        images = images.to(device)
        label_ids = label_ids.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, label_ids)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(label_ids)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == label_ids).sum().item()
        total_samples += len(label_ids)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:

            images, label_ids, _, _, _, _, _ = batch

            images = images.to(device)
            label_ids = label_ids.to(device)
            logits = model(images)
            loss = loss_fn(logits, label_ids)

            total_loss += loss.item() * len(label_ids)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == label_ids).sum().item()
            total_samples += len(label_ids)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc

def train_model(model, train_loader, val_loader, loss_fn, device, save_path, num_epochs, fine_tune_mode):
    
    if fine_tune_mode == "head_only":
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=LEARNING_RT_HEAD,
            weight_decay=WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": LEARNING_RT_BACKBONE},
            {"params": model.head.parameters(), "lr": LEARNING_RT_HEAD}
        ], weight_decay=WEIGHT_DECAY)

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.0)#?

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_one_epoch(model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"CHECKPOINT SAVED (val_loss: {val_loss:.4f})")
        
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"TRAINING STOPPED AT EPOCH {epoch}")
            break

        
    return history


            
