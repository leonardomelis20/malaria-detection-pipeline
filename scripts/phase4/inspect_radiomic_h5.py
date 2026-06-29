"""
Ispezione del file .h5 prodotto da extract_radiomic_features.py.
Da eseguire dopo l'estrazione per verificare shape, consistenza delle chiavi,
assenza di NaN/Inf, distribuzione delle classi e statistiche di base.
"""

from pathlib import Path
import h5py
import numpy as np

_BASE_ROOT = Path(__file__).resolve().parent.parent.parent

H5_PATH = _BASE_ROOT / "results" / "features" / "radiomics" / "fold1" / "train.h5"


def inspect_h5(h5_path: Path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5f:
        print("Chiavi presenti nel file:")
        print(list(h5f.keys()))

        features  = h5f["features"][:]
        label_ids = h5f["label_ids"][:]
        labels    = np.array([x.decode() for x in h5f["labels"][:]])
        phases    = np.array([x.decode() for x in h5f["phases"][:]])
        group_ids = np.array([x.decode() for x in h5f["group_ids"][:]])
        filenames = np.array([x.decode() for x in h5f["filenames"][:]])
        filepaths = np.array([x.decode() for x in h5f["filepaths"][:]])

    N = features.shape[0]
    # A differenza degli script deep, non c'è un assert sulla dimensione fissa
    # delle feature: PyRadiomics può restituire un numero variabile di feature
    # a seconda delle impostazioni (tipicamente 100-120 per analisi 2D completa)
    print(f"\nShape features: {features.shape}  ({features.shape[1]} feature radiomiche per immagine)")
    print(f"Campioni totali: {N}")

    for name, arr in [("label_ids", label_ids), ("labels", labels),
                      ("phases", phases), ("group_ids", group_ids),
                      ("filenames", filenames), ("filepaths", filepaths)]:
        assert len(arr) == N, f"Lunghezza {name} ({len(arr)}) != N ({N})"
    print("Consistenza chiavi: OK")

    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()
    print(f"Contiene NaN: {has_nan}")
    print(f"Contiene Inf: {has_inf}")
    assert not has_nan, "Errore: NaN trovati nelle feature"
    assert not has_inf, "Errore: valori infiniti trovati nelle feature"

    print("\nPrimi 3 campioni:")
    for i in range(min(3, N)):
        print(f"  [{i}] {labels[i]} | phase={phases[i]} | group={group_ids[i]}")
        print(f"       feat[:5] = {features[i][:5].round(4)}")

    uniq_lbl, cnt_lbl = np.unique(labels, return_counts=True)
    print("\nDistribuzione classi:")
    for lbl, cnt in zip(uniq_lbl, cnt_lbl):
        print(f"  {lbl}: {cnt}")

    uniq_ph, cnt_ph = np.unique(phases, return_counts=True)
    print("\nDistribuzione phases:")
    for ph, cnt in zip(uniq_ph, cnt_ph):
        print(f"  {ph}: {cnt}")

    print(f"\nStatistiche feature:")
    print(f"  min={features.min():.4f}  max={features.max():.4f}")
    print(f"  mean={features.mean():.4f}  std={features.std():.4f}")

    print("\nIspezione completata.")


if __name__ == "__main__":
    inspect_h5(H5_PATH)
