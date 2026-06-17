from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# =========================
# CONFIGURAZIONE
# =========================
DATASET_ROOT = Path(
    r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis"
)

MALARIAE_ROOT = DATASET_ROOT / "Malariae"
OVALE_ROOT = DATASET_ROOT / "Ovale"

OUTPUT_DETAILED_CSV = DATASET_ROOT / "geometry_check_detailed.csv"
OUTPUT_SUMMARY_CSV = DATASET_ROOT / "geometry_check_summary.csv"

VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# =========================
# FUNZIONI UTILI
# =========================
def find_gt_file(gt_dir: Path, sample: str):
    for ext in VALID_EXTS:
        p = gt_dir / f"{sample}{ext}"
        if p.exists():
            return p
    return None


def load_gt_mask(gt_path: Path):
    """
    Legge la maschera GT in scala di grigi e la binarizza (pixel > 0 -> 1)
    """
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    mask = (gt > 0).astype(np.uint8)
    return mask


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def classify_coverage(coverage):
    if coverage is None:
        return "EMPTY_GT"
    if coverage == 1.0:
        return "FULL_COVERAGE"
    if coverage == 0.0:
        return "MISSING"
    return "PARTIAL"


# =========================
# ANALISI
# =========================
detailed_rows = []
summary_rows = []

subset_dirs = [MALARIAE_ROOT, OVALE_ROOT]

for subset_dir in subset_dirs:
    gt_dir = subset_dir / "gt"
    report_path = subset_dir / "crops" / "report.csv"

    # Salto i subset che non hanno gt o report
    if not gt_dir.exists() or not report_path.exists():
        summary_rows.append({
            "subset": subset_dir.name,
            "n_rows": 0,
            "n_ok": 0,
            "n_partial": 0,
            "n_missing": 0,
            "n_empty_gt": 0,
            "n_missing_gt_file": 0,
            "n_invalid_box": 0,
            "mean_coverage": np.nan,
            "status": "MISSING_GT_OR_REPORT"
        })
        continue

    try:
        df = pd.read_csv(report_path)
    except Exception as e:
        summary_rows.append({
            "subset": subset_dir.name,
            "n_rows": 0,
            "n_ok": 0,
            "n_partial": 0,
            "n_missing": 0,
            "n_empty_gt": 0,
            "n_missing_gt_file": 0,
            "n_invalid_box": 0,
            "mean_coverage": np.nan,
            "status": f"REPORT_READ_ERROR: {e}"
        })
        continue

    required_cols = {"sample", "x1", "y1", "x2", "y2"}

    if not required_cols.issubset(df.columns):
        summary_rows.append({
            "subset": subset_dir.name,
            "n_rows": len(df),
            "n_ok": 0,
            "n_partial": 0,
            "n_missing": 0,
            "n_empty_gt": 0,
            "n_missing_gt_file": 0,
            "n_invalid_box": 0,
            "mean_coverage": np.nan,
            "status": f"MISSING_COLUMNS: {required_cols - set(df.columns)}"
        })
        continue

    n_ok = 0
    n_partial = 0
    n_missing = 0
    n_empty_gt = 0
    n_missing_gt_file = 0
    n_invalid_box = 0
    coverages = []

    for _, row in df.iterrows():
        sample = str(row["sample"]).strip()

        gt_path = find_gt_file(gt_dir, sample)
        if gt_path is None:
            detailed_rows.append({
                "subset": subset_dir.name,
                "sample": sample,
                "gt_file": "",
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "gt_pixels_total": np.nan,
                "gt_pixels_inside_crop": np.nan,
                "coverage": np.nan,
                "status": "MISSING_GT_FILE"
            })
            n_missing_gt_file += 1
            continue

        mask = load_gt_mask(gt_path)
        if mask is None:
            detailed_rows.append({
                "subset": subset_dir.name,
                "sample": sample,
                "gt_file": gt_path.name,
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "gt_pixels_total": np.nan,
                "gt_pixels_inside_crop": np.nan,
                "coverage": np.nan,
                "status": "GT_READING_ERROR"
            })
            continue

        h, w = mask.shape[:2]
        x1, y1, x2, y2 = clamp_box(row["x1"], row["y1"], row["x2"], row["y2"], w, h)

        if x2 <= x1 or y2 <= y1:
            detailed_rows.append({
                "subset": subset_dir.name,
                "sample": sample,
                "gt_file": gt_path.name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "gt_pixels_total": np.nan,
                "gt_pixels_inside_crop": np.nan,
                "coverage": np.nan,
                "status": "INVALID_BOX"
            })
            n_invalid_box += 1
            continue

        gt_pixels_total = int(mask.sum())

        if gt_pixels_total == 0:
            coverage = None
            gt_pixels_inside = 0
            status = "EMPTY_GT"
            n_empty_gt += 1
        else:
            crop_region = mask[y1:y2, x1:x2]
            gt_pixels_inside = int(crop_region.sum())
            coverage = gt_pixels_inside / gt_pixels_total
            status = classify_coverage(coverage)

            if status == "FULL_COVERAGE":
                n_ok += 1
            elif status == "PARTIAL":
                n_partial += 1
            elif status == "MISSING":
                n_missing += 1

            coverages.append(coverage)

        detailed_rows.append({
            "subset": subset_dir.name,
            "sample": sample,
            "gt_file": gt_path.name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "gt_pixels_total": gt_pixels_total,
            "gt_pixels_inside_crop": gt_pixels_inside,
            "coverage": coverage,
            "status": status
        })

    mean_coverage = float(np.mean(coverages)) if len(coverages) > 0 else np.nan

    if n_partial == 0 and n_missing == 0 and n_missing_gt_file == 0 and n_invalid_box == 0:
        overall_status = "OK"
    else:
        overall_status = "CHECK_ISSUES"

    summary_rows.append({
        "subset": subset_dir.name,
        "n_rows": len(df),
        "n_ok": n_ok,
        "n_partial": n_partial,
        "n_missing": n_missing,
        "n_empty_gt": n_empty_gt,
        "n_missing_gt_file": n_missing_gt_file,
        "n_invalid_box": n_invalid_box,
        "mean_coverage": mean_coverage,
        "status": overall_status
    })

# =========================
# SALVATAGGIO CSV
# =========================
detailed_df = pd.DataFrame(detailed_rows)
summary_df = pd.DataFrame(summary_rows)

if not detailed_df.empty:
    detailed_df = detailed_df.sort_values(by=["subset", "sample"])

if not summary_df.empty:
    summary_df = summary_df.sort_values(by=["subset"])

detailed_df.to_csv(OUTPUT_DETAILED_CSV, index=False, encoding="utf-8-sig")
summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

print("Geometric check completed")
print(f"Detailed report saved to: {OUTPUT_DETAILED_CSV}")
print(f"Summary report saved to: {OUTPUT_SUMMARY_CSV}")