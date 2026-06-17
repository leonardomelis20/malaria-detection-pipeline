import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from utils import load_features, get_classifiers, save_results, run_experiment, run_experiment_ood

BASE_ROOT = Path(r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis")
FEATURES_DIR = BASE_ROOT / "results" / "features"
RESULTS_PATH = BASE_ROOT / "results" / "classification"
SPECIES = ["Falciparum", "Vivax", "Ovale", "Malariae"]
all_results = []



def filter_by_species(x, y, species_array, target_species):
    mask = (species_array == target_species)

    return x[mask], y[mask]

def encode_phases(y_bytes):
    #bytes -> stringhe
    y_str = [phase.decode() for phase in y_bytes]

    #creazione encoder
    encoder = LabelEncoder()

    #fit e trasformazione
    y_encoded = encoder.fit_transform(y_str)

    #restituisco array codificato + encoder
    return y_encoded, encoder


#usa encoder già fittato per test set
def encode_phases_with_existing(y_bytes, encoder):
    y_str = [p.decode() for p in y_bytes]

    try:
        y_encoded = encoder.transform(y_str)
        return y_encoded, True
    except Exception:
        return None, False #classe non vista in training



#LOOP PRINCIPALE OOD
for model_dir in FEATURES_DIR.iterdir():

    if not model_dir.is_dir():
        continue
    model_name = model_dir.name

    for source_species in SPECIES:
        X_parts = []
        y_parts = []

        for fold_dir in sorted(model_dir.iterdir()):
            for split in ["train", "val", "test"]:
                h5_path = fold_dir / f"{split}.h5"

                if not h5_path.exists():
                    continue

                x, y , species, _ = load_features(h5_path, target="phases")
                x_filt, y_filt = filter_by_species(x, y, species, source_species.encode())

                if len(x_filt) > 0:
                    X_parts.append(x_filt)
                    y_parts.append(y_filt)
        if len(X_parts) == 0:
            print(f"WARNING: nessun campione per {source_species}, skip")
            continue

        X_train_ood = np.concatenate(X_parts, axis=0)
        y_train_ood = np.concatenate(y_parts, axis=0)


        #codifico fasi in interi
        y_train_encoded, le = encode_phases(y_train_ood)

        for target_species in SPECIES:
            if target_species == source_species:
                continue

            test_h5 = model_dir / "fold1" / "test.h5"
            if not test_h5.exists():
                print(f"WARNING: missing test for {model_name}, skip")
                continue

            x_test_all, y_test_all, species_test, _ = load_features(
                test_h5, target="phases"
            )
            x_test_ood, y_test_bytes = filter_by_species(
                x_test_all, y_test_all, species_test, target_species.encode()
            )

            if len(x_test_ood) == 0:
                print(f"WARNING: no sample target {target_species}, skip")
                continue

            y_test_encoded, ok = encode_phases_with_existing(y_test_bytes, le)

            if not ok:
                print(f"WARNING: classes not seen in training for {target_species}, skip")
                continue

            output_dir = RESULTS_PATH / "ood" / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            classifiers = get_classifiers()
            for clf_name, clf in classifiers.items():
                results, cm = run_experiment_ood(
                    X_train_ood, y_train_encoded,
                    x_test_ood, y_test_encoded,
                    clf, clf_name, model_name,
                    source_species, target_species
                )
                all_results.append(results)
                
                cm_name = f"{source_species}_to_{target_species}_{clf_name}.npy"
                np.save(output_dir / cm_name, cm)

save_results(all_results, RESULTS_PATH / "ood")

        