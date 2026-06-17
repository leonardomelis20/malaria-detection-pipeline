import pandas as pd
import os

TRAINVAL_CSV = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits_heldout\trainval_metadata.csv"
OUTPUT_CSV = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits_heldout\trainval_metadata_oversampled.csv"

LABEL_COL = "label"
MAJORITY_CLASS = "Falciparum"
RANDOM_STATE = 42

trainval_df = pd.read_csv(TRAINVAL_CSV)

class_counts = trainval_df[LABEL_COL].value_counts()
if MAJORITY_CLASS not in class_counts.index:
    raise ValueError("Falciparum non presente nel trainval")

target_count = class_counts[MAJORITY_CLASS]

oversampled = []
for label in class_counts.index:

    class_df = trainval_df[trainval_df[LABEL_COL] == label]
    n_class = len(class_df)

    if n_class < target_count:
         sampled_df = class_df.sample(
                n=target_count,
                replace=True,
                random_state=RANDOM_STATE
            )
    else:
        sampled_df = class_df.copy()
    oversampled.append(sampled_df)
final_df = pd.concat(oversampled, ignore_index =True)

final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

final_df["oversampling_target"] = target_count
final_df.to_csv(OUTPUT_CSV, index = False)

print("Original_distribution:")
print(class_counts)

print("\nOversampled distribution:")
print(final_df[LABEL_COL].value_counts())

print("\nOriginal samples:", len(trainval_df))
print("Oversampled samples:", len(final_df))
print("Saved in:", OUTPUT_CSV)




         