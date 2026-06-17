"""5 fold senza test held out, con fold creati manualmente in modo che ogni classe rara sia presente almeno una volta nel train.
Fold 1 val:
Falciparum: 1305121398, 1307210661
Malariae:   1312132815
Ovale:      1707180816
Vivax:      1703121298

Fold 2 val:
Falciparum: 1405022890, 1408161544
Malariae:   1401063467
Ovale:      1708161076
Vivax:      1709041080

Fold 3 val:
Falciparum: 1408290968, 1409171742
Malariae:   1401080976
Ovale:      -
Vivax:      -

Fold 4 val:
Falciparum: 1409191647, 1603223711
Malariae:   -
Ovale:      -
Vivax:      -

Fold 5 val:
Falciparum: 1701151546, 1704282807
Malariae:   -
Ovale:      -
Vivax:      -
"""

import pandas as pd
import os

N_FOLDS = 5
expected_labels = {"Falciparum", "Malariae", "Ovale", "Vivax"}
MEDATA_PATH = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\mpidb_metadata_nodiag.csv"
OUTPUT_DIR = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\lol_5-fold_group_aware_manual_nodiag"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#CARICAMENTO METADATA
metadata_df = pd.read_csv(MEDATA_PATH)
if metadata_df.empty:
    raise ValueError("Failed to load metadata from the specified path.")

#controllo colonne
required_columns = {'filepath', 'label', 'group_id', 'filename'}
if not required_columns.issubset(metadata_df.columns):
    raise ValueError(f"CSV file must contain: {required_columns}")

metadata_df['group_id'] = metadata_df['group_id'].astype(str)

#controllo labels
dataset_labels = set(metadata_df["label"].unique())
missing_labels = expected_labels - dataset_labels
if missing_labels:
    raise ValueError(f"Following classes are missing: {missing_labels}")

#creazione group_summary.csv
summary_rows = []
for group_id in sorted(metadata_df["group_id"].unique()):
    group_rows = metadata_df[metadata_df['group_id'] == group_id]
    num_samples = len(group_rows)
    num_labels = group_rows['label'].value_counts()

    row = {
        'group_id': group_id,
        'num_samples': num_samples
    }

    for label in expected_labels:
        row[label] = int(num_labels.get(label, 0))
    
    summary_rows.append(row)

group_summary_df = pd.DataFrame(summary_rows)
group_summary_file = os.path.join(OUTPUT_DIR, 'group_summary.csv')
group_summary_df.to_csv(group_summary_file, index = False)

#DEFINIZIONE MANUALE DEI GRUPPI DI VALIDATION PER OGNI FOLD
folds = {
    1: {"1305121398", "1307210661", "1312132815", "1707180816", "1703121298"},
    2: {"1405022890", "1408161544", "1401063467", "1708161076", "1709041080"},
    3: {"1408290968", "1409171742", "1401080976"},
    4: {"1409191647", "1603223711"},
    5: {"1701151546", "1704282807"},
}

if len(folds) != N_FOLDS:
    raise ValueError("Defined number of folds is not equal to N_FOLDS")

all_dataset_groups = set(metadata_df["group_id"].unique())

#controllo che i gruppi manuali esistano nel dataset
all_val_groups = set()

for fold_idx, val_groups in folds.items():
    missing_groups = val_groups - all_dataset_groups

    if missing_groups:
        raise ValueError(f"Fold {fold_idx}: group_id assenti: {missing_groups}")
    
    all_val_groups.update(val_groups)
    
#controllo che ogni group_id sia usato in validation una sola volta
val_groups_list = []

for val_groups in folds.values():
    val_groups_list.extend(val_groups)

if len(val_groups_list) != len(set(val_groups_list)):
    raise ValueError("One ore more group_ids are in multiple validation folds")

#controllo che tutti i gruppi siano assegnati ad almeno una validation
unassigned_groups = all_dataset_groups - all_val_groups

if unassigned_groups:
    raise ValueError(f"Following group_ids are not assigned to a validation fold: {unassigned_groups}")

#creo train/val per ogni fold
for fold_idx, val_groups in folds.items():
    train_groups = all_dataset_groups - val_groups

    fold_train_df = metadata_df[metadata_df['group_id'].isin(train_groups)].copy()
    fold_val_df = metadata_df[metadata_df['group_id'].isin(val_groups)].copy()

    leakage = set(fold_train_df['group_id']).intersection(set(fold_val_df['group_id']))

    if leakage:
        raise ValueError(f"Leakage nel fold {fold_idx}: {leakage}")
    
    train_labels = set(fold_train_df['label'].unique())
    missing_in_train = expected_labels - train_labels

    if missing_in_train:
        raise ValueError(f"Fold {fold_idx}: missing groups in train: {missing_in_train}")
    
    train_file = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_train.csv")
    val_file = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}_val.csv")

    fold_train_df.to_csv(train_file, index=False)
    fold_val_df.to_csv(val_file, index=False)

    #report terminale
    print(f"\nFOLD {fold_idx}")
    print("Train label distribution:")
    print(fold_train_df["label"].value_counts())

    print("Validation label distribution:")
    print(fold_val_df["label"].value_counts())

    print("Train groups:", fold_train_df["group_id"].nunique())
    print("Validation groups:", fold_val_df["group_id"].nunique())

print("\nCreazione dei 5 fold completata correttamente.")


"""Fold 1 e Fold 2 validano tutte le classi
Fold 3 valida solo Falciparum + Malariae
Fold 4 Fold 5 validano solo Falciparum"""

for fold in range(1, 6):
    train = pd.read_csv(os.path.join(OUTPUT_DIR, f"fold{fold}_train.csv"))
    val = pd.read_csv(os.path.join(OUTPUT_DIR, f"fold_{fold}_val.csv"))

    print(f"\nFOLD {fold}")
    print("TRAIN labels:")
    print(train["label"].value_counts())
    print("VAL labels:")
    print(val["label"].value_counts())
    print("TRAIN groups:", train["group_id"].nunique())
    print("VAL groups:", val["group_id"].nunique())




