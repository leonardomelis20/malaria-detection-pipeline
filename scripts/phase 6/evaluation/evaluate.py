"""dato un modello addestrato e un test set,
calcola le metriche e salva i risultati"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix

import json 
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
def run_evaluation(model, dataloader, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            images, label_ids, _, _, _, _, _ = batch

            images = images.to(device)
            label_ids = label_ids.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            preds_array = preds.cpu().numpy()
            probs_array = probs.cpu().numpy()
            labels_array = label_ids.cpu().numpy()

            all_preds.append(preds_array)
            all_probs.append(probs_array)
            all_labels.append(labels_array)

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    mcc = matthews_corrcoef(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f} | F1 Macro: {f1:.4f} | MCC: {mcc:.4f}")
    print(report)

    results = {
        "accuracy": acc,
        "f1_macro": f1,
        "mcc": mcc,
        "classification_report": report,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels
    }

    return results

def save_results(results, history, output_dir, model_name, fine_tune_mode):

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
        xticklabels=SPECIES,
        yticklabels=SPECIES,
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
    


    


    
