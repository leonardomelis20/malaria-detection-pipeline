import pandas as pd

METADATA = r"MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis/csvs/mpidb_metadata_nodiag.csv"

df = pd.read_csv(METADATA)

summary = (
    df.groupby(["label", "group_id"])
    .size()
    .reset_index(name="n_images")
    .sort_values(["label", "group_id"])
)

print(summary.to_string(index=False))