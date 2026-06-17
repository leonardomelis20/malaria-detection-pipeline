"""Aprire file .h5 prodotto dalla feature extraction e controllare:
- dataset contenenti
- shape delle features
- numero di label
- assenza di NaN
- corrispondenza feature-label-filepath
- prime righe"""

from pathlib import Path
import h5py
import numpy as np


H5_PATH = Path(
    r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\results\features\reddino_base\fold1\train.h5"
)


def inspect_h5(h5_path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5f:
        print("Keys presenti nel file:")
        print(list(h5f.keys()))

        features = h5f["features"][:] #[:] converter in array numpy
        label_ids = h5f["label_ids"][:]
        labels = h5f["labels"][:]
        phases = h5f["phases"][:]
        group_ids = h5f["group_ids"][:]
        filenames = h5f["filenames"][:]
        filepaths = h5f["filepaths"][:]

        #con x.decode("utf-8") trasformo da bytes a stringhe
        labels = np.array([x.decode("utf-8") for x in labels])
        phases = np.array([x.decode("utf-8") for x in phases])
        group_ids = np.array([x.decode("utf-8") for x in group_ids])
        filenames = np.array([x.decode("utf-8") for x in filenames])
        filepaths = np.array([x.decode("utf-8") for x in filepaths])

        print("\nInformazioni generali:")
        print("Features shape:", features.shape)
        print("Label IDs shape:", label_ids.shape)
        print("Labels shape:", labels.shape)
        print("Phases shape:", phases.shape)
        print("Group IDs shape:", group_ids.shape)
        print("Filenames shape:", filenames.shape)
        print("Filepaths shape:", filepaths.shape)

        N = features.shape[0]

        assert len(label_ids) == N, "Errore: label_ids non corrisponde a features"
        assert len(labels) == N, "Errore: labels non corrisponde a features"
        assert len(phases) == N, "Errore: phases non corrisponde a features"
        assert len(group_ids) == N, "Errore: group_ids non corrisponde a features"
        assert len(filenames) == N, "Errore: filenames non corrisponde a features"
        assert len(filepaths) == N, "Errore: filepaths non corrisponde a features"

        assert features.shape[1] == 768, f"Errore: feature dim = {features.shape[1]}, atteso 768"

        has_nan = np.isnan(features).any()
        print("Contiene NaN", has_nan)
        assert not has_nan, "Errore: features contiene NaN"

        has_inf = np.isinf(features).any()
        print("Contiene infiniti:", has_inf)
        assert not has_inf, "Errore: valori infiniti trovati"

        print("\nPrimi 5 campioni:")
        for i in range(min(5, N)):
            print(f"\nSample {i}")
            print("label_id:", label_ids[i])
            print("label:", labels[i])
            print("phase:", phases[i])
            print("group_id:", group_ids[i])
            print("filename:", filenames[i])
            print("filepath:", filepaths[i])
            print("prime 5 features:", features[i][:5])

        unique_labels, label_counts= np.unique(labels, return_counts=True)
        print("\nDistribuzione classi:")
        for label, count in zip(unique_labels, label_counts):
            print(label, count)
        
        unique_phases, phase_counts = np.unique(phases, return_counts=True)
        print("\nDistribuzione delle phases:")
        for phase, count in zip(unique_phases, phase_counts):
            print(phase, count)
        
        print("\nStatistiche features:")
        print("Min:", features.min())
        print("Max:", features.max())
        print("Mean:", features.mean())
        print("Std:", features.std())


        print("\nInspection completata")

if __name__ == "__main__":
    inspect_h5(H5_PATH)