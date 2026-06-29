"""
Estrazione feature radiomiche dai crop RBC di Fase 1.
Usa scikit-image (già installato) al posto di PyRadiomics.

Motivazione: PyRadiomics 3.0.1/3.1.0 usa configparser.SafeConfigParser nel proprio
sistema di build (versioneer.py); questa funzione è stata rimossa in Python 3.12.
Non esistono wheel pre-compilate per Python 3.12 su PyPI né nel repository GitHub.
Python non può essere retrocesso perché PyTorch richiede 3.12.

Feature estratte per immagine (~61 totali):
  - First-order  (15): statistiche sui pixel della ROI
                       (media, mediana, std, varianza, asimmetria, curtosi,
                        energia, entropia, max, min, range, rms, IQR, P10, P90)
  - GLCM         (36): Gray Level Co-occurrence Matrix
                       4 direzioni × 3 distanze × 6 proprietà
                       (contrasto, dissimilarità, omogeneità, energia, correlazione, ASM)
                       → per ogni coppia (distanza, proprietà): media e std across
                         le 4 direzioni, per una rappresentazione rotationally invariant
  - Shape 2D     (10): proprietà geometriche della ROI
                       (area, perimetro, eccentricità, solidità, extent,
                        assi maggiore/minore, diametro equivalente,
                        orientazione, numero di Eulero)
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from PIL import Image
from scipy import stats as scipy_stats
from skimage import feature as skfeature, measure as skmeasure

# --- Radice del progetto ricavata dalla posizione di questo file ---
_BASE_ROOT = Path(__file__).resolve().parent.parent.parent
_KNOWN_SPECIES = {"Falciparum", "Malariae", "Ovale", "Vivax"}

# --- Percorsi ---
KFOLD_DIR   = _BASE_ROOT / "csvs" / "kfold_heldout"
HELDOUT_DIR = _BASE_ROOT / "csvs" / "splits_heldout"
TEST_CSV    = HELDOUT_DIR / "test_heldout.csv"
OUTPUT_DIR  = _BASE_ROOT / "results" / "features" / "radiomics"

N_FOLDS = 5

LABEL_TO_ID = {"Falciparum": 0, "Malariae": 1, "Ovale": 2, "Vivax": 3}
REQUIRED_COLUMNS = {"filepath", "label", "phase", "group_id", "filename"}

# Parametri GLCM
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS    = 16   # stesso default di PyRadiomics
GLCM_PROPS     = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def _relocate_path(path_str: str) -> str:
    """
    Riporta un path assoluto da altra macchina sotto _BASE_ROOT locale.
    Stessa strategia di phase 6/data/dataset.py.
    """
    parts = Path(path_str).parts
    for i, part in enumerate(parts):
        if part in _KNOWN_SPECIES:
            return str(_BASE_ROOT.joinpath(*parts[i:]))
    return path_str


def build_roi_mask(gray_array: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crea una maschera binaria per la ROI.
    Pixel con intensità > threshold → 1 (incluso nella ROI).
    Threshold=10 esclude solo lo sfondo nero puro includendo tutta la cellula.
    Fallback: se la maschera fosse vuota, usa l'intera immagine come ROI.
    """
    mask = (gray_array > threshold).astype(np.uint8)
    if mask.sum() == 0:
        mask = np.ones_like(gray_array, dtype=np.uint8)
    return mask


def _first_order_features(roi_pixels: np.ndarray) -> dict:
    """
    Statistiche di primo ordine sui pixel della ROI.
    Equivalenti alla classe 'firstorder' di PyRadiomics.
    L'entropia è calcolata sull'istogramma a 256 bin (Shannon, log base 2).
    """
    p = roi_pixels.astype(np.float64)
    hist, _ = np.histogram(p, bins=256, range=(0, 256), density=True)
    h = hist[hist > 0]

    return {
        "fo_mean":     float(np.mean(p)),
        "fo_median":   float(np.median(p)),
        "fo_std":      float(np.std(p)),
        "fo_variance": float(np.var(p)),
        "fo_skewness": float(scipy_stats.skew(p)),
        "fo_kurtosis": float(scipy_stats.kurtosis(p)),
        "fo_energy":   float(np.sum(p ** 2) / len(p)),
        "fo_entropy":  float(-np.sum(h * np.log2(h))),
        "fo_max":      float(np.max(p)),
        "fo_min":      float(np.min(p)),
        "fo_range":    float(np.max(p) - np.min(p)),
        "fo_rms":      float(np.sqrt(np.mean(p ** 2))),
        "fo_iqr":      float(np.percentile(p, 75) - np.percentile(p, 25)),
        "fo_p10":      float(np.percentile(p, 10)),
        "fo_p90":      float(np.percentile(p, 90)),
    }


def _glcm_features(gray_array: np.ndarray, mask: np.ndarray) -> dict:
    """
    Feature GLCM (Gray Level Co-occurrence Matrix) per 4 direzioni e 3 distanze.
    Per ogni coppia (distanza, proprietà): media e std across le 4 direzioni
    producono una rappresentazione rotationally invariant che cattura anche
    la variabilità direzionale della texture.
    I pixel fuori maschera vengono azzerati prima della quantizzazione.
    NaN sulla correlazione (patch costante) → sostituiti con 0.
    """
    roi_masked = gray_array.copy()
    roi_masked[mask == 0] = 0
    quantized = (roi_masked / 255.0 * (GLCM_LEVELS - 1)).clip(0, GLCM_LEVELS - 1).astype(np.uint8)

    features = {}
    for d in GLCM_DISTANCES:
        glcm = skfeature.graycomatrix(
            quantized,
            distances=[d],
            angles=GLCM_ANGLES,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )
        for prop in GLCM_PROPS:
            # graycoprops → shape (n_distances, n_angles); [0] → (n_angles,)
            vals = skfeature.graycoprops(glcm, prop)[0]
            vals = np.nan_to_num(vals, nan=0.0)
            features[f"glcm_d{d}_{prop}_mean"] = float(np.mean(vals))
            features[f"glcm_d{d}_{prop}_std"]  = float(np.std(vals))

    return features


def _shape_features(mask: np.ndarray) -> dict:
    """
    Feature geometriche della ROI tramite regionprops.
    Se la maschera è degenere, restituisce zeri.
    Usa la regione più grande (in area) come proxy per la cellula principale.
    """
    _zero = {
        "shape_area": 0.0, "shape_perimeter": 0.0, "shape_eccentricity": 0.0,
        "shape_solidity": 0.0, "shape_extent": 0.0, "shape_major_axis_length": 0.0,
        "shape_minor_axis_length": 0.0, "shape_equivalent_diameter": 0.0,
        "shape_orientation": 0.0, "shape_euler_number": 0.0,
    }

    label_img = skmeasure.label(mask)
    if label_img.max() == 0:
        return _zero

    regions = skmeasure.regionprops(label_img)
    region  = max(regions, key=lambda r: r.area)

    try:
        return {
            "shape_area":                float(region.area),
            "shape_perimeter":           float(region.perimeter),
            "shape_eccentricity":        float(region.eccentricity),
            "shape_solidity":            float(region.solidity),
            "shape_extent":              float(region.extent),
            "shape_major_axis_length":   float(region.major_axis_length),
            "shape_minor_axis_length":   float(region.minor_axis_length),
            "shape_equivalent_diameter": float(region.equivalent_diameter_area),
            "shape_orientation":         float(region.orientation),
            "shape_euler_number":        float(region.euler_number),
        }
    except Exception:
        return _zero


def extract_single_image(image_path: str) -> tuple:
    """
    Estrae le feature da una singola immagine.
    Ritorna (feature_array: np.ndarray, feature_names: list)
    oppure (None, None) in caso di errore.

    Conversione RGB→grayscale con PIL convert("L"):
    Y = 0.299R + 0.587G + 0.114B (luminanza percettiva standard).
    """
    try:
        gray_np  = np.array(Image.open(image_path).convert("L"))
        mask_np  = build_roi_mask(gray_np)
        roi_pix  = gray_np[mask_np == 1]

        all_feats = {
            **_first_order_features(roi_pix),
            **_glcm_features(gray_np, mask_np),
            **_shape_features(mask_np),
        }

        feature_names  = list(all_feats.keys())
        feature_values = np.array(list(all_feats.values()), dtype=np.float64)
        # Sanificazione finale: NaN/Inf residui → 0
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_values, feature_names

    except Exception as e:
        print(f"    ERRORE su {image_path}: {e}")
        return None, None


def process_split(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonne mancanti in {csv_path}: {missing_cols}")

    df["filepath"] = df["filepath"].apply(_relocate_path)

    all_features, all_label_ids, all_labels = [], [], []
    all_phases, all_group_ids, all_filenames, all_filepaths = [], [], [], []

    feature_names = None
    n_errors = 0

    for _, row in df.iterrows():
        feat_values, feat_names = extract_single_image(row["filepath"])

        if feat_values is None:
            n_errors += 1
            continue

        if feature_names is None:
            feature_names = feat_names

        all_features.append(feat_values)
        all_label_ids.append(LABEL_TO_ID[row["label"]])
        all_labels.append(row["label"])
        all_phases.append(row["phase"])
        all_group_ids.append(str(row["group_id"]))
        all_filenames.append(row["filename"])
        all_filepaths.append(row["filepath"])

    if n_errors > 0:
        print(f"    Immagini saltate per errore: {n_errors}/{len(df)}")

    return {
        "features":  np.array(all_features),
        "label_ids": np.array(all_label_ids, dtype=np.int64),
        "labels":    np.array(all_labels),
        "phases":    np.array(all_phases),
        "group_ids": np.array(all_group_ids),
        "filenames": np.array(all_filenames),
        "filepaths": np.array(all_filepaths),
    }


def save_h5(data: dict, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("features",  data=data["features"])
        h5f.create_dataset("label_ids", data=data["label_ids"])
        h5f.create_dataset("labels",    data=data["labels"].astype("S"))
        h5f.create_dataset("phases",    data=data["phases"].astype("S"))
        h5f.create_dataset("group_ids", data=data["group_ids"].astype("S"))
        h5f.create_dataset("filenames", data=data["filenames"].astype("S"))
        h5f.create_dataset("filepaths", data=data["filepaths"].astype("S"))
    print(f"    Salvato: {output_path}")


def validate_h5(output_path: Path):
    with h5py.File(output_path, "r") as h5f:
        features = h5f["features"][:]
        labels   = h5f["labels"][:]
    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()
    print(f"    Validazione {output_path.name}: shape={features.shape}  NaN={has_nan}  Inf={has_inf}")
    if has_nan or has_inf:
        raise ValueError(f"NaN/Inf trovati in {output_path}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Mismatch features/labels")
    print("    Validazione ok")


def main():
    for fold in range(1, N_FOLDS + 1):
        print(f"\n========== FOLD {fold} ==========")

        split_csvs = {
            "train": KFOLD_DIR / f"fold{fold}_train.csv",
            "val":   KFOLD_DIR / f"fold_{fold}_val.csv",
            "test":  TEST_CSV,
        }
        fold_output_dir = OUTPUT_DIR / f"fold{fold}"

        for split_name, csv_path in split_csvs.items():
            output_path = fold_output_dir / f"{split_name}.h5"
            if output_path.exists():
                print(f"  {split_name}: già presente, skip")
                continue
            print(f"\n  Split: {split_name} — {csv_path}")
            data = process_split(csv_path)
            print(f"    Campioni estratti: {data['features'].shape[0]}")
            save_h5(data, output_path)
            validate_h5(output_path)


if __name__ == "__main__":
    main()
