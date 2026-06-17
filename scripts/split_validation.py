import pandas as pd
import os

# cartella dove si trovano gli split generati
splits_dir = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits"

# file finale di report
output_csv = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\split_validation_report.csv"

expected_labels = {"Falciparum", "Malariae", "Ovale", "Vivax"}

results = []

for split_file in os.listdir(splits_dir):

    if not split_file.endswith(".csv"):
        continue

    split_path = os.path.join(splits_dir, split_file)
    df = pd.read_csv(split_path)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_groups = set(train_df["group_id"])
    test_groups = set(test_df["group_id"])

    overlap_groups = train_groups.intersection(test_groups)

    train_labels = set(train_df["label"])
    test_labels = set(test_df["label"])

    missing_train = expected_labels - train_labels
    missing_test = expected_labels - test_labels

    dup_filename = df["filename"].duplicated().sum()
    dup_filepath = df["filepath"].duplicated().sum()

    row = {
        "split_file": split_file,

        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),

        "row_sum_ok": (len(train_df) + len(test_df) == len(df)),

        "train_groups_n": len(train_groups),
        "test_groups_n": len(test_groups),
        "group_overlap_n": len(overlap_groups),
        "group_overlap_ids": ",".join(map(str, sorted(overlap_groups))),

        "missing_train_labels_n": len(missing_train),
        "missing_train_labels": ",".join(sorted(missing_train)),

        "missing_test_labels_n": len(missing_test),
        "missing_test_labels": ",".join(sorted(missing_test)),

        "duplicate_filename_n": dup_filename,
        "duplicate_filepath_n": dup_filepath,

        "falciparum_train": (train_df["label"] == "Falciparum").sum(),
        "malariae_train": (train_df["label"] == "Malariae").sum(),
        "ovale_train": (train_df["label"] == "Ovale").sum(),
        "vivax_train": (train_df["label"] == "Vivax").sum(),

        "falciparum_test": (test_df["label"] == "Falciparum").sum(),
        "malariae_test": (test_df["label"] == "Malariae").sum(),
        "ovale_test": (test_df["label"] == "Ovale").sum(),
        "vivax_test": (test_df["label"] == "Vivax").sum(),
    }

    results.append(row)

report_df = pd.DataFrame(results)

report_df = report_df.sort_values("split_file").reset_index(drop=True)

report_df.to_csv(output_csv, index=False)

print("Report salvato in:")
print(output_csv)

print(report_df)