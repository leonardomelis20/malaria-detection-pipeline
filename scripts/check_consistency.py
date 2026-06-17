from pathlib import Path
import pandas as pd 


# =========================
# CONFIGURAZIONE
# =========================
DATASET_ROOT = Path(
    r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis"
)

OUTPUT_DETAILED_CSV = DATASET_ROOT / "file_consistency_detailed.csv"
OUTPUT_SUMMARY_CSV = DATASET_ROOT / "file_consistency_summary.csv"

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# FUNZIONI UTILI
# =========================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMG_EXTS


def list_image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if is_image_file(p)])


def expected_crop_name(img_name: str) -> str:
    """
    Da '1312132815-0001-G.jpg' produce '1312132815-0001-G_diag.png'
    """
    stem = Path(img_name).stem
    return f"{stem}_diag.png"


# =========================
# ANALISI
# =========================
detailed_rows = []
summary_rows = []

# Considera come subset tutte le sottocartelle immediate del dataset root
subset_dirs = [p for p in DATASET_ROOT.iterdir() if p.is_dir()]

for subset_dir in sorted(subset_dirs):
    img_dir = subset_dir / "img"
    gt_dir = subset_dir / "gt"
    crops_dir = subset_dir / "crops" / "diagnostics"

    # Se non c'è almeno img o gt, salta
    if not img_dir.exists() and not gt_dir.exists() and not crops_dir.exists():
        continue

    img_files = list_image_files(img_dir)
    gt_files = list_image_files(gt_dir)
    crop_files = list_image_files(crops_dir)

    img_names = {p.name for p in img_files}
    gt_names = {p.name for p in gt_files}
    crop_names = {p.name for p in crop_files}

    # 1) Righe basate su img: per ogni immagine controllo gt e crop attesi
    for img_name in sorted(img_names):
        crop_name = expected_crop_name(img_name)

        img_present = True
        gt_present = img_name in gt_names
        crop_present = crop_name in crop_names

        if gt_present and crop_present:
            status = "OK"
        else:
            missing = []
            if not gt_present:
                missing.append("GT")
            if not crop_present:
                missing.append("CROP")
            status = "MISSING_" + "_".join(missing)

        detailed_rows.append({
            "subset": subset_dir.name,
            "image_name": img_name,
            "expected_gt_name": img_name,
            "expected_crop_name": crop_name,
            "img_present": img_present,
            "gt_present": gt_present,
            "crop_present": crop_present,
            "status": status
        })

    # 2) File in gt che non hanno immagine corrispondente
    gt_only = sorted(gt_names - img_names)
    for gt_name in gt_only:
        detailed_rows.append({
            "subset": subset_dir.name,
            "image_name": gt_name,
            "expected_gt_name": gt_name,
            "expected_crop_name": expected_crop_name(gt_name),
            "img_present": False,
            "gt_present": True,
            "crop_present": False,
            "status": "GT_WITHOUT_IMG"
        })

    # 3) File crop che non corrispondono a nessuna immagine
    expected_crop_names_from_img = {expected_crop_name(name) for name in img_names}
    crop_only = sorted(crop_names - expected_crop_names_from_img)
    for crop_name in crop_only:
        detailed_rows.append({
            "subset": subset_dir.name,
            "image_name": "",
            "expected_gt_name": "",
            "expected_crop_name": crop_name,
            "img_present": False,
            "gt_present": False,
            "crop_present": True,
            "status": "CROP_WITHOUT_IMG"
        })

    # 4) Riassunto per subset
    n_img = len(img_names)
    n_gt = len(gt_names)
    n_crop = len(crop_names)

    n_ok = sum(
        1 for img_name in img_names
        if (img_name in gt_names) and (expected_crop_name(img_name) in crop_names)
    )
    n_missing_gt = sum(1 for img_name in img_names if img_name not in gt_names)
    n_missing_crop = sum(1 for img_name in img_names if expected_crop_name(img_name) not in crop_names)
    n_gt_without_img = len(gt_only)
    n_crop_without_img = len(crop_only)

    if (
        n_img == n_gt == n_crop
        and n_missing_gt == 0
        and n_missing_crop == 0
        and n_gt_without_img == 0
        and n_crop_without_img == 0
    ):
        subset_status = "OK"
    else:
        subset_status = "MISMATCH"

    summary_rows.append({
        "subset": subset_dir.name,
        "n_img_files": n_img,
        "n_gt_files": n_gt,
        "n_crop_files": n_crop,
        "n_ok_matches": n_ok,
        "n_missing_gt_for_img": n_missing_gt,
        "n_missing_crop_for_img": n_missing_crop,
        "n_gt_without_img": n_gt_without_img,
        "n_crop_without_img": n_crop_without_img,
        "subset_status": subset_status
    })


# =========================
# SALVATAGGIO CSV
# =========================
detailed_df = pd.DataFrame(detailed_rows)
summary_df = pd.DataFrame(summary_rows)

# Ordine colonne
detailed_cols = [
    "subset",
    "image_name",
    "expected_gt_name",
    "expected_crop_name",
    "img_present",
    "gt_present",
    "crop_present",
    "status"
]
summary_cols = [
    "subset",
    "n_img_files",
    "n_gt_files",
    "n_crop_files",
    "n_ok_matches",
    "n_missing_gt_for_img",
    "n_missing_crop_for_img",
    "n_gt_without_img",
    "n_crop_without_img",
    "subset_status"
]

if not detailed_df.empty:
    detailed_df = detailed_df[detailed_cols].sort_values(by=["subset", "image_name", "expected_crop_name"])
else:
    detailed_df = pd.DataFrame(columns=detailed_cols)

if not summary_df.empty:
    summary_df = summary_df[summary_cols].sort_values(by=["subset"])
else:
    summary_df = pd.DataFrame(columns=summary_cols)

detailed_df.to_csv(OUTPUT_DETAILED_CSV, index=False, encoding="utf-8-sig")
summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

print("Analisi completata.")
print(f"CSV dettagliato salvato in: {OUTPUT_DETAILED_CSV}")
print(f"CSV riassuntivo salvato in: {OUTPUT_SUMMARY_CSV}")