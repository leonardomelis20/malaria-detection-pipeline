import pandas as pd
import os
"""Utilizzando un test set heldout Ovale e Vivax hanno un solo gruppo nel trainval.
Di conseguenza:
Ovale -> sempre train
Vivax -> sempre train.
Ne consegue che i fold validation utilizzeranno solo Falciparum e Malariae"""

TRAINVAL_CSV = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits_heldout\trainval_metadata.csv"
OUTPUT_DIR = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\kfold_heldout"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
trainval_df = pd.read_csv(TRAINVAL_CSV)
"""fold1_val = {
    "Falciparum": {"1305121398"},
    "Malariae": {"1312132815"}
}

fold2_val = {
    "Falciparum" : {"1307210661", "1405022890"},
    "Malariae" : {"1401080976"}
}

fold3_val = {
    "Falciparum": {"1408161544", "1408290968"}
}

fold4_val = {
    "Falciparum": {"1409171742", "1409191647"}
}

fold5_val = {
    "Falciparum": {"1603223711", "1701151546"}
}
"""

folds = {
    1: {"1305121398","1312132815"},
    2: {"1307210661", "1405022890", "1401080976"}, 
    3: {"1408161544", "1408290968"},
    4: {"1409171742", "1409191647"},
    5: {"1603223711", "1701151546"}
}

#Ovale 1707180816 -> sempre train
#Vivax 1709041080 -> sempre train

trainval_df["group_id"] = trainval_df["group_id"].astype(str)

for fold, group_ids in folds.items():

    group_ids = {str(g) for g in group_ids}
    val_mask = trainval_df["group_id"].isin(group_ids)

    val_df = trainval_df[val_mask].copy()

    train_df = trainval_df[~val_mask].copy()

    train_file = os.path.join(OUTPUT_DIR, f"fold{fold}_train.csv")
    val_file = os.path.join(OUTPUT_DIR, f"fold_{fold}_val.csv")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    overlap = train_groups.intersection(val_groups)

    print(f"\n===== FOLD {fold} =====")
    print("TRAIN LABEL DISTRIBUTION")
    print(train_df["label"].value_counts())
    print("VAL LABEL DISTRIBUTION")
    print(val_df["label"].value_counts())

    print("TRAIN GROUPS")
    print(train_df["group_id"].nunique())
    print("VAL GROUPS")
    print(val_df["group_id"].nunique())


    print("OVERLAP:", overlap)

    print(f"Saved: {train_file}")
    print(f"Saved: {val_file}")
