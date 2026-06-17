import pandas as pd
from pathlib import Path
import numpy as np
import h5py
from utils import load_features, save_results, get_classifiers, run_experiment



BASE_ROOT = Path(r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis")
FEATURES_PATH = BASE_ROOT / "results" / "features"
RANDOM_STATE = 42


#LOOP PRINCIPALE INTRA-DATASET
RESULTS_PATH = BASE_ROOT / "results" / "classification"

all_results = []

for model_dir in FEATURES_PATH.iterdir():
    if not model_dir.is_dir():
        continue

    model_name = model_dir.name

    
    for fold_dir in model_dir.iterdir():
        if not fold_dir.is_dir():
            continue

        fold_name = fold_dir.name
        train_h5 = fold_dir / "train.h5"
        val_h5 = fold_dir / "val.h5"
        test_h5 = fold_dir / "test.h5"

        if not train_h5.exists():
            print(f"WARNING: file mancante in {fold_dir}, skip")
            continue
        if not val_h5.exists():
            print(f"WARNING: file mancante in {fold_dir}, skip")
            continue
        if not test_h5.exists():
            print(f"WARNING: file mancante in {fold_dir}, skip")
            continue
        
        output_dir = RESULTS_PATH / "intra" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        classifiers = get_classifiers()
        """classificatore ricreato per ogni fold perchè devo partire da un 
        classificatore non addestrato"""
       

        for clf_name, clf in classifiers.items():
            result_val, cm_val = run_experiment(
                train_h5, 
                val_h5,
                clf,
                clf_name, 
                model_name,
                fold_name)
            
            #runno esperimento su val
            result_val["split"] = "val"
            all_results.append(result_val)
            

            #salva confusion matrix di validation
            np.save(output_dir / f"{fold_name}_cm_{clf_name}_val.npy", cm_val)

            result_test, cm_test = run_experiment(
                train_h5,
                test_h5,
                clf,
                clf_name,
                model_name,
                fold_name
            )
            result_test["split"] = "test"
            all_results.append(result_test)

            np.save(output_dir / f"{fold_name}_cm_{clf_name}_test.npy", cm_test)

save_results(all_results, RESULTS_PATH / "intra")

        
            


