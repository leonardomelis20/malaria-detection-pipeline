"""Provo a portare il numero di campioni delle classi minoritarie allo stesso
livello o simili al numero di campioni di Falciparum.
Fold1_train -> fold1_train_oversampled -> training

IN questo file duplico le righe delle classi minoritarie"""

import pandas as pd
import os
import numpy as np


INPUT_DIR = r"MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\kfold_heldout"
OUTPUT_DIR = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\csvs\oversampled_folds"

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS = 5
LABEL_COL = "label"
MAJORITY_CLASS = "Falciparum"

for fold in range(1, N_FOLDS+1):
    input_csv = os.path.join(INPUT_DIR, f"fold{fold}_train.csv")
    output_csv = os.path.join(OUTPUT_DIR, f"fold{fold}_train_oversampled.csv")
    
    df = pd.read_csv(input_csv)

    if LABEL_COL not in df.columns:
        raise ValueError("Errore nella lettura delle labels")
    
    class_count = df[LABEL_COL].value_counts()

    if MAJORITY_CLASS not in class_count.index:
        raise ValueError(f"Errore: {MAJORITY_CLASS} non presente nel fold {fold}")

    target_count = class_count[MAJORITY_CLASS]

    oversampled_parts = []

    for label in class_count.index:

        class_df = df[df[LABEL_COL] == label]
        n_class = class_df.shape[0]
        
        if n_class < target_count:
            sampled_df = class_df.sample(
                n=target_count,
                replace=True,
                random_state=fold
            )
        else:
            sampled_df = class_df.copy()
        oversampled_parts.append(sampled_df)

    final_df = pd.concat(oversampled_parts, ignore_index=True)
    #mescolo le righe del dataframe e resetto indici
    final_df = final_df.sample(
        frac=1,
        random_state=fold
    ).reset_index(drop=True)

        
    final_df["source_fold"] = fold
    final_df["oversampling_target"] = target_count

    final_df.to_csv(output_csv, index=False)


    print(f"\n===== FOLD {fold} =====")
    print("Original distribution:")
    print(class_count)
    print("\nOversampled distribution:")
    print(final_df[LABEL_COL].value_counts())
    print(f"\nOriginal samples: {len(df)}")
    print(f"Oversampled samples: {len(final_df)}")
    print(f"Saved in: {output_csv}")
