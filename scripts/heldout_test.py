import pandas as pd
import os
METADATA = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\mpidb_metadata_nodiag.csv"
OUTPUT_DIR = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits_holdout"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(METADATA)
df["group_id"] = df["group_id"].astype(str)

#definizione manuale dei gruppi da mettere nel test

TEST_GROUPS = {
    "Falciparum": ["1704282807"],
    "Malariae": ["1401063467"],
    "Ovale": ["1708161076"],
    "Vivax": ["1703121298"]
}

#maschera booleana
is_test = df.apply(
    lambda row: row["group_id"] in TEST_GROUPS.get(row["label"], []),
    axis = 1)

test_df = df[is_test].copy()
trainval_df = df[~is_test].copy()

test_df.to_csv(f"{OUTPUT_DIR}/test_heldout.csv", index=False)
trainval_df.to_csv(f"{OUTPUT_DIR}/trainval_metadata.csv", index=False)

print("\nTEST DISTRIBUTION")
print(test_df["label"].value_counts())

print("\nTRAINVAL DISTRIBUTION")
print(trainval_df["label"].value_counts())

print("\nTEST GROUPS")
print(test_df.groupby("label")["group_id"].unique())

test_groups = set(test_df["group_id"].astype(str))
trainval_groups = set(trainval_df["group_id"].astype(str))

overlap = test_groups.intersection(trainval_groups)

print("Overlap:", overlap)