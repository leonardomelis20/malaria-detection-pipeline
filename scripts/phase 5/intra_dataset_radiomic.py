"""
Classificazione intra-dataset sulle feature radiomiche (Fase 5 — radiomica).
Usa le feature estratte da extract_radiomic_features.py (results/features/radiomics/).
Salva i risultati in results/classification/intra_radiomic/.

Differenza rispetto a intra_dataset.py (deep features):
- applica StandardScaler prima dell'addestramento di ogni classificatore.
  Le feature radiomiche hanno scale molto eterogenee: l'area di una ROI può
  valere centinaia, mentre alcune metriche di texture stanno in [0, 1].
  Senza normalizzazione, k-NN calcola distanze dominate dalle feature con
  range più alto, e Logistic Regression converge più lentamente o male.
  Lo scaler viene fittato solo sul training set (fit_transform su X_train,
  transform su X_test) per evitare data leakage.
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from utils import load_features, get_classifiers, save_results

_BASE_ROOT   = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = _BASE_ROOT / "results" / "features" / "radiomics"
RESULTS_PATH = _BASE_ROOT / "results" / "classification" / "intra_radiomic"

MODEL_NAME = "radiomics"


def run_experiment_radiomic(train_h5, test_h5, clf, clf_name, fold_name):
    X_train, y_train, _, _ = load_features(train_h5)
    X_test,  y_test,  _, _ = load_features(test_h5)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    y_pred      = clf.predict(X_test)
    all_classes = np.unique(y_train)

    acc                = accuracy_score(y_test, y_pred)
    f1_macro           = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm                 = confusion_matrix(y_test, y_pred, labels=all_classes)
    precision_macro    = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_macro       = recall_score(y_test, y_pred,    average="macro",    zero_division=0)
    recall_weighted    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    balanced_acc       = balanced_accuracy_score(y_test, y_pred)
    mcc                = matthews_corrcoef(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(
            y_test, clf.predict_proba(X_test), multi_class="ovr", average="macro"
        )
    except Exception:
        roc_auc = float("nan")

    results = {
        "model":               MODEL_NAME,
        "classifier":          clf_name,
        "fold":                fold_name,
        "accuracy":            acc,
        "balanced_accuracy":   balanced_acc,
        "precision_macro":     precision_macro,
        "precision_weighted":  precision_weighted,
        "recall_macro":        recall_macro,
        "recall_weighted":     recall_weighted,
        "f1_macro":            f1_macro,
        "f1_weighted":         f1_weighted,
        "mcc":                 mcc,
        "roc_auc_macro":       roc_auc,
        "n_classes_in_val":    len(np.unique(y_test)),
    }
    return results, cm


all_results = []

for fold_dir in sorted(FEATURES_DIR.iterdir()):
    if not fold_dir.is_dir():
        continue

    fold_name = fold_dir.name
    train_h5  = fold_dir / "train.h5"
    val_h5    = fold_dir / "val.h5"
    test_h5   = fold_dir / "test.h5"

    if not (train_h5.exists() and val_h5.exists() and test_h5.exists()):
        print(f"WARNING: file mancante in {fold_dir}, skip")
        continue

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Classificatori ricreati per ogni fold: partono non addestrati
    for clf_name, clf in get_classifiers().items():
        result_val, cm_val = run_experiment_radiomic(
            train_h5, val_h5, clf, clf_name, fold_name
        )
        result_val["split"] = "val"
        all_results.append(result_val)
        np.save(RESULTS_PATH / f"{fold_name}_cm_{clf_name}_val.npy", cm_val)

        result_test, cm_test = run_experiment_radiomic(
            train_h5, test_h5, clf, clf_name, fold_name
        )
        result_test["split"] = "test"
        all_results.append(result_test)
        np.save(RESULTS_PATH / f"{fold_name}_cm_{clf_name}_test.npy", cm_test)

save_results(all_results, RESULTS_PATH)
print(f"\nRisultati intra-dataset radiomica salvati in: {RESULTS_PATH}")
