"""
Classi rare:

OVALE
Fold 1 val → 1707180816
Fold 2 val → 1708161076
Fold 3 val → nessun Ovale
Fold 4 val → nessun Ovale
Fold 5 val → nessun Ovale

VIVAX
Fold 1 val → 1703121298
Fold 2 val → 1709041080
Fold 3 val → nessun Vivax
Fold 4 val → nessun Vivax
Fold 5 val → nessun Vivax

in ogni fold il train deve contenere sempre almeno 1 gruppo Ovale e 1 Vivax

MALARIAE
Fold 1 val → 1312132815
Fold 2 val → 1401063467
Fold 3 val → 1401080976
Fold 4 val → nessun Malariae
Fold 5 val → nessun Malariae

FALCIPARUM
Fold 1 val → 1305121398, 1307210661
Fold 2 val → 1405022890, 1408161544
Fold 3 val → 1408290968, 1409171742
Fold 4 val → 1409191647, 1603223711
Fold 5 val → 1701151546, 1704282807


val_groups = gruppi assegnati manualmente al fold
train_groups = tutti i trainval_groups - val_groups

ho provato a tenere un test separato ma ho ottenuto un CV poco affidabile con classe rare instabili
"""

import pandas as pd
import random
import os


N_FOLDS = 5
TEST_RATIO = 0.2
TEST_SEED = 42
seeds_list = [random.randint(0, 10000) for _ in range(N_FOLDS)]
labels = {"Falciparum", "Malariae", "Ovale", "Vivax"}
test_groups= list()
trainval_groups = list()


metadata_path = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\mpidb_metadata.csv"
labels = {"Falciparum", "Malariae", "Ovale", "Vivax"}
df = pd.read_csv(metadata_path)

output_dir = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\kfold"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if df.empty:
    raise ValueError("The CSV is empty!")

required_columns = {"filepath", "label", "group_id", "filename"}    
if not required_columns.issubset(df.columns):
    raise ValueError("The metadata CSV must contain 'filepath', 'label', 'group_id', and 'filename' columns.")
    
unique_groups = df["group_id"].unique()
n_groups = len(unique_groups) #array di valori unici

if n_groups < N_FOLDS:
    raise ValueError("Number of groups smaller than the number of folds!")

#controllo dei gruppi e delle etichette
summary_rows = []
for group_id in unique_groups:
    group_rows = df[df["group_id"] == group_id]
    total_samples = len(group_rows)
    label_counts = group_rows["label"].value_counts()

    row = {
        "group_id": group_id,
        "n_samples": total_samples
    }
    for label_name, count in label_counts.items():
        row[label_name] = count
    summary_rows.append(row)
group_summary_df = pd.DataFrame(summary_rows)
group_summary_df = group_summary_df.fillna(0)
group_summary_file = os.path.join(output_dir, "group_summary.csv")
group_summary_df.to_csv(group_summary_file, index=False)

#creo lo split sui gruppi
random.seed(TEST_SEED)

for label in labels:
    label_df = df[df["label"] == label]
    label_groups = list(label_df["group_id"].unique())

    random.shuffle(label_groups)

    test_group = label_groups[0]
    trainval_groups_for_label = label_groups[1:]

    test_groups.append(test_group)
    trainval_groups.extend(trainval_groups_for_label)

test_df = df[df["group_id"].isin(test_groups)]
trainval_df = df[df["group_id"].isin(trainval_groups)]

test_file = os.path.join(output_dir, "test.csv")
trainval_file = os.path.join(output_dir, "trainval.csv")

test_df.to_csv(test_file, index=False)
trainval_df.to_csv(trainval_file, index=False)

print("TEST LABEL DISTRIBUTION:")
print(test_df["label"].value_counts())

print("TRAINVAL LABEL DISTRIBUTION:")
print(trainval_df["label"].value_counts())

print("TEST GROUPS:")
print(test_df.groupby("label")["group_id"].unique())

"""Dato che il trainval contiene:

Falciparum: molti gruppi
Malariae: 2 gruppi rimasti
Ovale: 1 gruppo rimasto
Vivax: 1 gruppo rimasto

c’è un limite importante: non puoi avere ogni validation fold con tutte e 4 le classi, perché Ovale e Vivax nel trainval hanno solo 1 gruppo ciascuno.

Quindi devi accettare che alcuni fold di validation non avranno tutte le classi. Va scritto nel report come limite dovuto alla bassa numerosità di gruppi."""


#creazione dei fold
trainval_unique_groups = list(trainval_df["group_id"].unique())

random.seed(TEST_SEED)
random.shuffle(trainval_unique_groups)

fold_groups = [[] for _ in range(N_FOLDS)]

#divido i fold in 5 gruppi
for i, group_id in enumerate(trainval_unique_groups):
    
    fold_index = i % N_FOLDS
    fold_groups[fold_index].append(group_id)


for fold_idx in range(N_FOLDS):
    val_groups = fold_groups[fold_idx]

    train_groups = []

    for i in range(N_FOLDS):
        if i != fold_idx:
            train_groups.extend(fold_groups[i])
    
    #righe con group_id in val/train group
    fold_train_df = trainval_df[trainval_df["group_id"].isin(train_groups)]
    fold_val_df = trainval_df[trainval_df["group_id"].isin(val_groups)]

    #controllo leakage
    overlap = set(fold_train_df["group_id"]).intersection(set(fold_val_df["group_id"]))

    if len(overlap) > 0:
        raise ValueError(f"Leakage detected in fold {fold_idx + 1}: {overlap}")
    
    #creazione e salvataggio file
    train_file = os.path.join(output_dir, f"fold_{fold_idx+1}_train.csv")
    val_file = os.path.join(output_dir, f"fold_{fold_idx + 1}_val.csv")

    fold_train_df.to_csv(train_file, index=False)
    fold_val_df.to_csv(val_file, index=False)

    print(f"\nFOLD {fold_idx + 1}")
    print("Train distribution:")
    print(fold_train_df["label"].value_counts())
    print("Validation distribution:")
    print(fold_val_df["label"].value_counts())