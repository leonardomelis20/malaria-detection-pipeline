#estrazione del group_id dal prefisso del filename
#e creazione file csv da usare per split del dataset
import os
import pandas as pd
import re

base_dir = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis" 


classes = ["Falciparum", "Malariae", "Ovale", "Vivax"]

phases = ["G", "R", "S", "T"]

rows = []
seen_base_names= set()

for label in classes:
    for phase in phases:
        phase_dir = os.path.join(base_dir, label, "crops", phase)

        if not os.path.isdir(phase_dir):
            print(f"Directory {phase_dir} non trovata")
            continue

        for filename in os.listdir(phase_dir):

            if "diag" in filename.lower():
                continue
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            name_without_ext = os.path.splitext(filename)[0]

            #rimuovo suffisso finale
            base_name = re.sub(r'_\d+$', '', name_without_ext)

            if base_name in seen_base_names:
                print(f"Duplicato saltato: {base_name} in {phase_dir}")
                continue

            seen_base_names.add(base_name)

            parts = base_name.split("-")
            if len(parts) < 2:
                print(f"Filename non conforme: {filename} in {phase_dir}")
                continue
            
            group_id = parts[0]
            file_path = os.path.join(phase_dir, filename)

            row = {
            "filepath": file_path,
            "label": label,
            "phase": phase,
            "group_id": group_id,
            "filename": filename
        }
            rows.append(row)

print(len(rows))
print(rows[:5])

df = pd.DataFrame(rows)
df = df.sort_values(by=["label", "group_id", "filename"]).reset_index(drop=True)
df.to_csv(os.path.join(base_dir, "mpidb_metadata_nodiag.csv"), index=False)

print(df["label"].value_counts())
print(df.groupby("label")["group_id"].nunique())
print(df.groupby(["label", "phase"]).size())
print(f"Totale righe: {len(df)}, conteggio per classe: {df['label'].value_counts()},Gruppi unici per classe: {df.groupby('label')['group_id'].nunique()}")

