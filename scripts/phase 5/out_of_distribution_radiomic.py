"""
Classificazione OOD sulle feature radiomiche (Fase 5 — radiomica).
Usa le feature estratte da extract_radiomic_features.py (results/features/radiomics/).
Salva i risultati in results/classification/ood_radiomic/.

Setup OOD: allena su TUTTI i campioni di una specie sorgente (da tutti i fold
e tutti gli split: train+val+test), poi predice la FASE (stadio di sviluppo
del parassita: R/G/S/T) su campioni di una specie target non vista in training.
Questo testa se le feature radiomiche apprese su una specie descrivono
le fasi del ciclo vitale in modo generalizzabile ad altre specie.

Malariae mai come sorgente (9 coppie: Falciparum, Vivax, Ovale × 3 target ciascuna).

StandardScaler: fittato sui dati della specie sorgente, applicato ai dati target.
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from utils import load_features, get_classifiers, save_results

_BASE_ROOT   = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = _BASE_ROOT / "results" / "features" / "radiomics"
RESULTS_PATH = _BASE_ROOT / "results" / "classification" / "ood_radiomic"

MODEL_NAME     = "radiomics"
SOURCE_SPECIES = ["Falciparum", "Vivax", "Ovale"]
ALL_SPECIES    = ["Falciparum", "Vivax", "Ovale", "Malariae"]


def filter_by_species(x, y, species_array, target_bytes):
    mask = (species_array == target_bytes)
    return x[mask], y[mask]


def encode_phases(y_bytes):
    y_str = [p.decode() for p in y_bytes]
    le = LabelEncoder()
    return le.fit_transform(y_str), le


def encode_phases_with_existing(y_bytes, encoder):
    y_str = [p.decode() for p in y_bytes]
    try:
        return encoder.transform(y_str), True
    except Exception:
        return None, False


all_results = []

for source_species in SOURCE_SPECIES:

    # --- Raccolta training: tutti i campioni della specie sorgente ---
    X_parts, y_parts = [], []
    for fold_dir in sorted(FEATURES_DIR.iterdir()):
        if not fold_dir.is_dir():
            continue
        for split in ["train", "val", "test"]:
            h5_path = fold_dir / f"{split}.h5"
            if not h5_path.exists():
                continue
            x, y, species, _ = load_features(h5_path, target="phases")
            x_filt, y_filt = filter_by_species(x, y, species, source_species.encode())
            if len(x_filt) > 0:
                X_parts.append(x_filt)
                y_parts.append(y_filt)

    if not X_parts:
        print(f"WARNING: nessun campione per sorgente {source_species}, skip")
        continue

    X_train_ood = np.concatenate(X_parts, axis=0)
    y_train_ood = np.concatenate(y_parts, axis=0)
    y_train_encoded, le = encode_phases(y_train_ood)

    # Scaler fittato su training (sorgente), applicato poi a ciascun target
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ood)

    # Classificatori fittati una volta sola per questa sorgente
    # (il training set è lo stesso per tutti i target)
    fitted_classifiers = {}
    for clf_name, clf in get_classifiers().items():
        clf.fit(X_train_scaled, y_train_encoded)
        fitted_classifiers[clf_name] = clf

    # --- Test su ogni specie target ---
    for target_species in ALL_SPECIES:
        if target_species == source_species:
            continue

        # Test set fisso: fold1/test.h5 (test_heldout.csv, uguale per tutti i fold)
        test_h5 = FEATURES_DIR / "fold1" / "test.h5"
        if not test_h5.exists():
            print(f"WARNING: test.h5 mancante, skip {source_species}→{target_species}")
            continue

        x_test_all, y_test_all, species_test, _ = load_features(test_h5, target="phases")
        x_test_filt, y_test_bytes = filter_by_species(
            x_test_all, y_test_all, species_test, target_species.encode()
        )
        if len(x_test_filt) == 0:
            print(f"WARNING: nessun campione target {target_species}, skip")
            continue

        y_test_encoded, ok = encode_phases_with_existing(y_test_bytes, le)
        if not ok:
            print(f"WARNING: fasi non viste in training per {target_species}, skip")
            continue

        X_test_scaled = scaler.transform(x_test_filt)

        output_dir = RESULTS_PATH / f"{source_species}_to_{target_species}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for clf_name, clf in fitted_classifiers.items():
            y_pred      = clf.predict(X_test_scaled)
            all_classes = np.unique(y_train_encoded)

            acc                = accuracy_score(y_test_encoded, y_pred)
            f1_macro           = f1_score(y_test_encoded, y_pred, average="macro",    zero_division=0)
            f1_weighted        = f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0)
            cm                 = confusion_matrix(y_test_encoded, y_pred, labels=all_classes)
            precision_macro    = precision_score(y_test_encoded, y_pred, average="macro",    zero_division=0)
            precision_weighted = precision_score(y_test_encoded, y_pred, average="weighted", zero_division=0)
            recall_macro       = recall_score(y_test_encoded, y_pred,    average="macro",    zero_division=0)
            recall_weighted    = recall_score(y_test_encoded, y_pred,    average="weighted", zero_division=0)
            balanced_acc       = balanced_accuracy_score(y_test_encoded, y_pred)
            mcc                = matthews_corrcoef(y_test_encoded, y_pred)

            try:
                roc_auc = roc_auc_score(
                    y_test_encoded, clf.predict_proba(X_test_scaled),
                    multi_class="ovr", average="macro"
                )
            except Exception:
                roc_auc = float("nan")

            results = {
                "model":               MODEL_NAME,
                "classifier":          clf_name,
                "source_species":      source_species,
                "target_species":      target_species,
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
                "n_classes_in_test":   len(np.unique(y_test_encoded)),
            }
            all_results.append(results)
            np.save(
                output_dir / f"{source_species}_to_{target_species}_{clf_name}.npy",
                cm
            )

save_results(all_results, RESULTS_PATH)
print(f"\nRisultati OOD radiomica salvati in: {RESULTS_PATH}")
