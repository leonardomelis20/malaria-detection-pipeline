"""Controllo file .h5 DinoBloom"""

from pathlib import Path
import h5py
import numpy as np


H5_PATH = Path(
    r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\results\features\dinobloom_base\fold1\train.h5"
)


def inspect_h5(h5_path):

    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as h5f:

        print("Keys:")
        print(list(h5f.keys()))

        features = h5f["features"][:]
        labels = h5f["labels"][:]

        labels = np.array([
            x.decode("utf-8")
            for x in labels
        ])

        print("\nINFO")
        print("Features shape:", features.shape)
        print("Labels:", len(labels))

        has_nan = np.isnan(features).any()
        has_inf = np.isinf(features).any()

        print("Contains NaN:", has_nan)
        print("Contains inf:", has_inf)

        print("\nSTATISTICHE")
        print("Min :", features.min())
        print("Max :", features.max())
        print("Mean:", features.mean())
        print("Std :", features.std())

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        print("\nDistribuzione classi:")
        for label, count in zip(unique_labels, counts):
            print(label, count)

        print("\nPrime 5 features sample 0:")
        print(features[0][:5])

        print("\nInspection completata")


if __name__ == "__main__":
    inspect_h5(H5_PATH)