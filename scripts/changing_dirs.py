import os
import shutil

base_dir = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis"

classes = ["Ovale", "Malariae"]

phases = ["G", "R", "S", "T"]

for label in classes:
    crops_dir = os.path.join(base_dir, label, "crops")
    diagnostics_dir = os.path.join(crops_dir, "diagnostics")

    if not os.path.exists(diagnostics_dir):
        print(f"Directory {diagnostics_dir} non trovata")
        continue

    for phase in phases:
        if not os.path.exists(os.path.join(crops_dir, phase)):
            os.makedirs(os.path.join(crops_dir, phase))
    
    files = os.listdir(diagnostics_dir)

    for filename in files:
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split("-")

        if len(parts) < 3:
            print("Filename non conforme:", filename)
            continue

        stage_string = parts[2]
        tokens_stage = stage_string.split("_")

        present_stages = []

        for token in tokens_stage:
            if token in phases and token not in present_stages:
                present_stages.append(token)
        if not present_stages:
            print("Nessuna fase riconosciuta in:", filename)
            continue

        source_path = os.path.join(diagnostics_dir, filename)
        for single_stage in present_stages:
            destination_dir = os.path.join(crops_dir, single_stage, filename)
            
            shutil.copy2(source_path, destination_dir)
            print(f"Copiato {filename} -> {single_stage}")
