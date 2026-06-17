"""
RBC Crop Extractor — MP-IDB Malaria Dataset

Estrazione automatica dei crop dei globuli rossi infetti (RBC)
partendo dalle maschere ground-truth del parassita.


Output :
    - diagnostics/  ← una figura per ogni campione (4)
    - report.csv    ← coordinate, metodo e raggio per ogni campione
    - stampa a terminale con statistiche aggregate

"""

import cv2
import numpy as np
import os
import csv
import time
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")           # backend non-interattivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════════
#  CONFIGURAZIONE 
# ═══════════════════════════════════════════════════════════════

DEFAULT_DATASET_DIR  = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Malariae"   # cartella con img/ e gt/
DEFAULT_OUTPUT_DIR   = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Malariae\crops_masked"     # dove salvare i risultati

# Ricerca LAB
SEARCH_RADIUS        = 130           # px attorno al centroide parassita

# Padding finale del crop
PADDING_FACTOR       = 1.18          # margine extra attorno al RBC trovato

# Hough Circles
HOUGH_RADIUS_MIN     = 18            # raggio minimo RBC atteso (px)
HOUGH_RADIUS_MAX     = 85            # raggio massimo RBC atteso (px)

# Fallback 
MEDIAN_RBC_RADIUS_PX = 142        # raggio mediano stimato del dataset

# ── Validazione plausibilità contorno LAB ──────────────────────
# Il raggio derivato dall'ellisse LAB deve stare in questo range
# rispetto al raggio mediano atteso; fuori range → rigetta e vai a Hough.
LAB_RADIUS_MIN_FACTOR = 0.35  
LAB_RADIUS_MAX_FACTOR = 1.75

# Compattezza minima del contorno LAB accettato (4π·A/P²).
# Un blob enorme di background ha compattezza molto bassa.
# Cerchio perfetto = 1.0; un RBC tipico ≈ 0.55–0.85.
LAB_MIN_COMPACTNESS = 0.35

# ── Validazione raggio Hough ───────────────────────────────────
# Il cerchio Hough deve avere raggio ≥ questa frazione del mediano
# per evitare di selezionare il parassita stesso (caso 0007-S).
HOUGH_MIN_RADIUS_FACTOR = 0.45  # sotto 0.45 * MEDIAN → probabilmente è il parassita

# Output
SAVE_DIAGNOSTICS     = True          # salva diagnostica PNG per ogni campione
SHOW_DIAGNOSTICS     = False         # mostra interattivo (lento su dataset grandi)

IMG_EXTENSIONS       = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ═══════════════════════════════════════════════════════════════
# Analisi maschera GT (centroide + area parassita)
# ═══════════════════════════════════════════════════════════════

def analyze_gt_mask(gt_mask: np.ndarray):
    """
    Estrae centroide, bounding box e area del parassita dalla maschera GT.
    Se la maschera contiene più RBC, prende quello con area maggiore.

    Returns:
        centroid (cx, cy), bbox (x,y,w,h), area float, contour ndarray
        Tutti None se la maschera è vuota.
    """
    _, binary = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5:
        return None, None, None, None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None, None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), cv2.boundingRect(c), area, c


# ═══════════════════════════════════════════════════════════════
# Segmentazione RBC via canale A (spazio LAB)
# ═══════════════════════════════════════════════════════════════

def segment_rbc_lab(img: np.ndarray, centroid: tuple, search_radius: int):
    """
    Segmenta il RBC in una ROI attorno al centroide del parassita.
    Il canale A dello spazio LAB separa efficacemente i globuli rossi
    (tonalità rosa-rossa) dal background (tonalità verde-grigia).

   Strategia:
      1. Ritaglia ROI quadrata di lato 2*search_radius centrata sul parassita
      2. Converte in LAB estraendo canale A
      3. Threshold di Otsu + morfologia (close riempie pallore centrale;
         open rimuove rumore)
      4. Seleziona il contorno che contiene il centroide del parassita;
         se nessuno lo contiene, prende il più vicino geometricamente

    Returns:
        best_contour  : contorno in coordinate locali alla ROI (o None)
        offset        : (x1, y1) per convertire in coordinate globali
        roi           : immagine della ROI (BGR)
        debug_binary  : maschera binaria per diagnostica
    """
    cx, cy = centroid
    H, W = img.shape[:2]
    r = search_radius

    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(W, cx + r)
    y2 = min(H, cy + r)

    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None, (x1, y1), roi, None

    lab  = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1]

    _, binary = cv2.threshold(a_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfologia
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=3)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open,  iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, (x1, y1), roi, binary

    center_local = (cx - x1, cy - y1)

    # Calcolo metriche per ogni contorno candidato
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        perim = cv2.arcLength(cnt, True)
        compactness = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0
        inside = cv2.pointPolygonTest(cnt, center_local, False) >= 0
        candidates.append((cnt, area, compactness, inside))

    if not candidates:
        return None, (x1, y1), roi, binary

    # Contorni che contengono il centroide (prende il più compatto)
    containing = [(c, a, comp) for c, a, comp, ins in candidates if ins]
    if containing:
        best = max(containing, key=lambda x: x[2])[0]
        return best, (x1, y1), roi, binary

    # Altrimenti contorno più vicino al centroide
    def dist_to_center(item):
        cnt = item[0]
        M2 = cv2.moments(cnt)
        if M2["m00"] == 0:
            return 1e9
        return np.hypot(M2["m10"]/M2["m00"] - center_local[0],
                        M2["m01"]/M2["m00"] - center_local[1])

    best = min(candidates, key=dist_to_center)[0]
    return best, (x1, y1), roi, binary


# ═══════════════════════════════════════════════════════════════
#  Circular Hough Transform (fallback geometrico)
# ═══════════════════════════════════════════════════════════════

def find_rbc_hough(img: np.ndarray, centroid: tuple,
                   r_min: int = HOUGH_RADIUS_MIN,
                   r_max: int = HOUGH_RADIUS_MAX):
    """
    Cerca cerchi nella ROI allargata (3x r_max) attorno al centroide.
    Prova parametri progressivamente più permissivi (param2: 30 → 22 → 15)
    per aumentare la robustezza su immagini con contrasto variabile.

    Returns:
        (cx_globale, cy_globale, raggio) oppure None
    """
    cx, cy = centroid
    H, W = img.shape[:2]
    search_r = r_max * 3

    x1 = max(0, cx - search_r)
    y1 = max(0, cy - search_r)
    x2 = min(W, cx + search_r)
    y2 = min(H, cy + search_r)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    for param2 in [30, 22, 15]:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=r_min * 2,
            param1=50, param2=param2,
            minRadius=r_min, maxRadius=r_max
        )
        if circles is not None:
            break

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    cloc = (cx - x1, cy - y1)
    best = min(circles, key=lambda c: np.hypot(c[0] - cloc[0], c[1] - cloc[1]))
    return (int(best[0] + x1), int(best[1] + y1), int(best[2]))


# ═══════════════════════════════════════════════════════════════
# Costruzione crop adattivo
# ═══════════════════════════════════════════════════════════════

def build_crop(img: np.ndarray,
               rbc_contour=None, offset=(0, 0),
               rbc_circle=None,
               padding_factor: float = PADDING_FACTOR):
    """
    Costruisce il crop quadrato attorno al RBC.

    Se il contorno ha ≥ 5 punti, adatta un'ellisse (più preciso su
    forme irregolari e RBC deformati). Altrimenti usa la bounding box.
    Se viene fornito il cerchio (da Hough), usa il raggio direttamente.

    Returns:
        crop          : immagine ritagliata (BGR)
        coords        : (x1, y1, x2, y2) globali
        radius        : raggio effettivo del crop (px)
        method_detail : stringa descrittiva per il report
    """
    H, W = img.shape[:2]

    if rbc_contour is not None:
        ox, oy = offset
        if len(rbc_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(rbc_contour)
                (ex, ey), (axis_a, axis_b), _ = ellipse
                ex, ey = ex + ox, ey + oy
                radius = int((max(axis_a, axis_b) / 2) * padding_factor)
                detail = "lab+ellipse"
            except Exception:
                x, y, w, h = cv2.boundingRect(rbc_contour)
                ex, ey = x + ox + w // 2, y + oy + h // 2
                radius = int(max(w, h) / 2 * padding_factor)
                detail = "lab+bbox_fallback"
        else:
            x, y, w, h = cv2.boundingRect(rbc_contour)
            ex, ey = x + ox + w // 2, y + oy + h // 2
            radius = int(max(w, h) / 2 * padding_factor)
            detail = "lab+bbox"

    elif rbc_circle is not None:
        ex, ey, r = rbc_circle
        radius = int(r * padding_factor)
        detail = "hough"

    else:
        return None, None, None, None

    x1 = max(0, int(ex - radius))
    y1 = max(0, int(ey - radius))
    x2 = min(W, int(ex + radius))
    y2 = min(H, int(ey + radius))

    if x2 <= x1 or y2 <= y1:
        return None, None, None, None

    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), radius, detail


def build_crop_mask(img_shape, coords, contour=None, offset=(0,0)):
    x1, y1, x2, y2 = coords
    H, W = img_shape[:2]

    full_mask = np.zeros((H,W), dtype=np.uint8)

    if contour is None:
        return None
    
    ox, oy = offset
    contour_global = contour.copy()
    contour_global[:, 0, 0] += ox
    contour_global[:, 0 , 1] += oy

    cv2.drawContours(full_mask, [contour_global], -1, 255, thickness=-1)

    crop_mask = full_mask[y1:y2, x1:x2].copy()
    return crop_mask


# ═══════════════════════════════════════════════════════════════
#  Pipeline con fallback a cascata
# ═══════════════════════════════════════════════════════════════

def _validate_lab_contour(contour, offset, parasite_area: float, parasite_centroid) -> tuple[bool, str]:
    """
    Valida la plausibilità del contorno LAB prima di accettarlo.

    Controlli:
      1. Area > 1.8x parasite_area  (deve essere più grande del parassita)
      2. Compattezza ≥ LAB_MIN_COMPACTNESS  (blob di background → bassa compattezza)
      3. Raggio ellisse in [LAB_RADIUS_MIN_FACTOR, LAB_RADIUS_MAX_FACTOR] * MEDIAN

    Returns:
        (True, "ok")  oppure  (False, motivo_rifiuto)
    """
    area = cv2.contourArea(contour)

    # 1. Dimensione minima
    if area <= parasite_area * 1.2:
        return False, f"area troppo piccola ({area:.0f} ≤ {parasite_area*1.2:.0f})"

    # 2. Compattezza — filtra blob fusionati e forme aberranti
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area > 0 else 0

    if solidity < 0.68:
        return False, f"solidity troppo bassa ({solidity:.3f} < 0.68)"
    
    
    # 3. Raggio plausibile — stima dal contorno in coordinate globali
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (axis_a, axis_b), _ = ellipse
            est_radius = max(axis_a, axis_b) / 2
        except cv2.error:
            x, y, w, h = cv2.boundingRect(contour)
            est_radius = max(w, h) / 2
    else:
        x, y, w, h = cv2.boundingRect(contour)
        est_radius = max(w, h) / 2
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (_,_), (axis_a, axis_b), _ = ellipse

            major = max(axis_a, axis_b)
            minor = min(axis_a, axis_b)

            axis_ratio = major / minor if minor > 0 else 999
            est_radius = major / 2
        except cv2.error:
            x, y, w, h = cv2.boundingRect(contour)

            axis_ratio = max(w, h) / min(w, h)
            est_radius = max(w, h) / 2
    else:
        x, y, w, h = cv2.boundingRect(contour)

        axis_ratio = max(w, h) / min(w, h)
        est_radius = max(w, h) / 2

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return False, "momenti nulli"
    
    ccx = M["m10"] / M["m00"]
    ccy = M["m01"] / M["m00"]

    pcx, pcy = parasite_centroid
    ox, oy = offset
    ccx_global = ccx + ox
    ccy_global = ccy + oy
    
    centroid_dist = np.hypot(pcx - ccx_global, pcy - ccy_global)
    relative_centroid_dist = centroid_dist / est_radius if est_radius > 0 else 999

    if relative_centroid_dist > 0.55:
        return False, f"centroid_dist troppo alta ({relative_centroid_dist:.2f} > 0.55)"
    if axis_ratio > 2.2:
        return False, f"axis_ration troppo alto ({axis_ratio:.2f}>2.2)"

    r_min = LAB_RADIUS_MIN_FACTOR * MEDIAN_RBC_RADIUS_PX
    r_max = LAB_RADIUS_MAX_FACTOR * MEDIAN_RBC_RADIUS_PX
    if not (r_min <= est_radius <= r_max):
        return False, (f"raggio stimato fuori range "
                       f"({est_radius:.0f}px, atteso {r_min:.0f}–{r_max:.0f}px)")

    return True, "ok"


def _validate_hough_circle(circle: tuple) -> tuple[bool, str]:
    """
    Valida il cerchio Hough: il raggio deve essere ≥ HOUGH_MIN_RADIUS_FACTOR
    volte il raggio mediano del dataset, per evitare di selezionare il
    parassita stesso invece del RBC (caso 0007-S).

    Returns:
        (True, "ok")  oppure  (False, motivo_rifiuto)
    """
    _, _, r = circle
    r_min = HOUGH_MIN_RADIUS_FACTOR * MEDIAN_RBC_RADIUS_PX
    if r < r_min:
        return False, f"raggio Hough troppo piccolo ({r}px < {r_min:.0f}px, probabilmente è il parassita)"
    return True, "ok"


def extract_rbc_crop(img: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Pipeline a cascata con validazione di plausibilità:
      1. LAB segmentation + fitEllipse   → validato (compattezza + range raggio)
      2. Circular Hough Transform        → validato (raggio minimo)
      3. Centroide GT + raggio mediano   → fallback finale

    Casi gestiti dalle validazioni:
      - 0006-T: contorno LAB enorme (blob fuso) → rifiutato per raggio > MAX_FACTOR
      - 0007-S: cerchio Hough piccolo (parassita) → rifiutato per raggio < MIN_FACTOR
                → entrambi ricadono sul fallback mediano che è calibrato a 142px

    Returns dict con chiavi:
        crop, coords, radius, method,
        parasite_area, centroid, lab_binary, rejection_log
    """
    result = {
        "crop": None, "crop_mask": None, "coords": None, "radius": None,
        "method": "failed", "parasite_area": None,
        "centroid": None, "lab_binary": None,
        "rejection_log": []          # traccia i motivi di rifiuto (utile per debug)
    }

    centroid, bbox, parasite_area, _ = analyze_gt_mask(gt_mask)
    if centroid is None:
        result["method"] = "no_parasite_found"
        return result

    result["centroid"]      = centroid
    result["parasite_area"] = parasite_area

    # ── Stadio 1: LAB ─────────────────────────────────────────────
    contour, offset, roi, binary = segment_rbc_lab(img, centroid, SEARCH_RADIUS)
    result["lab_binary"] = binary

    if contour is not None:
        valid, reason = _validate_lab_contour(contour, offset, parasite_area, centroid)
        if valid:
            crop, coords, radius, detail = build_crop(
                img, rbc_contour=contour, offset=offset)
            if crop is not None:
                crop_mask = build_crop_mask(
                    img_shape = img.shape,
                    coords=coords,
                    contour=contour,
                    offset=offset
                )
                masked_crop = None
                if crop_mask is not None and crop_mask.shape[:2] == crop.shape[:2]:
                    masked_crop = cv2.bitwise_and(crop, crop, mask=crop_mask)

                result.update(
                    crop=crop,
                    crop_mask = crop_mask,
                    masked_crop=masked_crop,
                    coords=coords,
                    radius=radius,
                    method=f"lab{detail}"
                )
                return result
        else:
            result["rejection_log"].append(f"LAB rifiutato: {reason}")

    # ── Stadio 2: Hough ───────────────────────────────────────────
    circle = find_rbc_hough(img, centroid)
    if circle is not None:
        valid, reason = _validate_hough_circle(circle)
        if valid:
            crop, coords, radius, _ = build_crop(img, rbc_circle=circle)
            if crop is not None:
                result.update(crop=crop, coords=coords, radius=radius,
                              method="hough")
                return result
        else:
            result["rejection_log"].append(f"Hough rifiutato: {reason}")

    # ── Stadio 3: Fallback mediana ─────────────────────────────────
    cx, cy = centroid
    H, W   = img.shape[:2]
    radius = int(MEDIAN_RBC_RADIUS_PX * PADDING_FACTOR)
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(W, cx + radius)
    y2 = min(H, cy + radius)
    result.update(
        crop=img[y1:y2, x1:x2].copy(),
        coords=(x1, y1, x2, y2),
        radius=radius,
        method="fallback (median radius)"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════

METHOD_COLORS = {
    "lab":      "#4CAF50",   # verde  → metodo principale riuscito
    "hough":    "#2196F3",   # blu    → fallback Hough
    "fallback": "#FF9800",   # arancio → fallback mediana
    "failed":   "#F44336",   # rosso  → nessun risultato
    "no_":      "#F44336",
}

def _method_color(method: str) -> str:
    for k, v in METHOD_COLORS.items():
        if k in method:
            return v
    return "#9E9E9E"


def save_diagnostic(img_bgr, gt_mask, result, sample_name, out_path):
    """
    Figura con 4 pannelli:
      [1] Immagine originale + overlay GT (giallo) + bbox crop (colorato)
      [2] Canale A (LAB) — mostra la separazione cromatica
      [3] Maschera binaria LAB post-morfologia
      [4] Crop finale con metodo e raggio
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    method  = result.get("method", "failed")
    color   = _method_color(method)

    fig = plt.figure(figsize=(16, 5), facecolor="#12121f")
    gs  = GridSpec(1, 4, figure=fig, wspace=0.06,
                   left=0.01, right=0.99, top=0.88, bottom=0.04)

    # ── Pannello 1: originale + overlay ──
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_rgb)

    gt_rgba = np.zeros((*gt_mask.shape, 4), dtype=np.uint8)
    gt_rgba[gt_mask > 0] = [255, 230, 0, 200]
    ax1.imshow(gt_rgba)

    coords = result.get("coords")
    if coords:
        x1, y1, x2, y2 = coords
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--"
        )
        ax1.add_patch(rect)

    cx, cy = result.get("centroid") or (0, 0)
    ax1.plot(cx, cy, "+", color="#FFD700", markersize=11, markeredgewidth=2)
    ax1.set_title("Originale + GT overlay", color="#cccccc", fontsize=8, pad=3)
    ax1.axis("off")

    # ── Pannello 2: canale A LAB ──
    ax2 = fig.add_subplot(gs[1])
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    ax2.imshow(lab[:, :, 1], cmap="RdYlGn_r")
    ax2.set_title("Canale A (LAB)", color="#cccccc", fontsize=8, pad=3)
    ax2.axis("off")

    # ── Pannello 3: maschera binaria ──
    ax3 = fig.add_subplot(gs[2])
    binary = result.get("lab_binary")
    if binary is not None:
        ax3.imshow(binary, cmap="gray")
    else:
        ax3.set_facecolor("#111")
        ax3.text(0.5, 0.5, "N/A", color="#555", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12)
    ax3.set_title("Maschera binaria LAB", color="#cccccc", fontsize=8, pad=3)
    ax3.axis("off")

    # ── Pannello 4: crop finale ──
    ax4 = fig.add_subplot(gs[3])
    crop = result.get("crop")
    if crop is not None and crop.size > 0:
        ax4.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        r = result.get("radius", "—")
        ax4.set_title(f"Crop — {method}\nr ≈ {r} px",
                      color=color, fontsize=8, pad=3)
    else:
        ax4.set_facecolor("#111")
        ax4.text(0.5, 0.5, "FAILED", color="#F44336",
                 ha="center", va="center", fontsize=14,
                 fontweight="bold", transform=ax4.transAxes)
        ax4.set_title("Crop finale", color="#F44336", fontsize=8, pad=3)
    ax4.axis("off")

    # Bordo colorato 
    for ax in [ax1, ax2, ax3, ax4]:
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(1.4)

    fig.suptitle(f"RBC Extractor  ·  {sample_name}  ·  {method}",
                 color="white", fontsize=9, y=0.98)

    fig.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Report statistico
# ═══════════════════════════════════════════════════════════════

def print_report(stats: list):
    total   = len(stats)
    failed  = sum(1 for s in stats if "fail" in s["method"] or "no_" in s["method"])
    methods = {}
    radii   = []

    for s in stats:
        m = s["method"].split(" ")[0]
        methods[m] = methods.get(m, 0) + 1
        if s.get("radius"):
            radii.append(s["radius"])

    W = 62
    print("\n" + "═" * W)
    print("  REPORT — RBC Extractor")
    print("═" * W)
    print(f"  Campioni processati : {total}")
    ok = total - failed
    print(f"  Successi            : {ok}  ({100*ok/total:.1f}%)")
    print(f"  Falliti             : {failed}")
    print()
    print("  Distribuzione metodi:")
    for m, count in sorted(methods.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, int(24 * count / total))
        print(f"    {m:<30} {count:>4}  {bar}")

    if radii:
        arr = np.array(radii)
        print()
        print("  Statistiche raggi RBC trovati (px):")
        print(f"    Mediana    : {np.median(arr):.1f}")
        print(f"    Media      : {np.mean(arr):.1f}  ±{np.std(arr):.1f}")
        print(f"    Min / Max  : {arr.min()} / {arr.max()}")
        suggested = int(np.median(arr))
        print()
        print(f"  ► SUGGERIMENTO: imposta MEDIAN_RBC_RADIUS_PX = {suggested}")
        print(f"    (riga ~35 dello script) per calibrare il fallback.")
    print("═" * W + "\n")


def save_csv_report(stats: list, out_path: str):
    fields = ["sample", "method", "radius_px", "crop_w", "crop_h",
              "parasite_area_px2", "cx", "cy", "x1", "y1", "x2", "y2",
              "rejection_log"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in stats:
            coords = s.get("coords") or (None, None, None, None)
            crop   = s.get("crop")
            ctr    = s.get("centroid")
            rej    = s.get("rejection_log", [])
            w.writerow({
                "sample"           : s["sample"],
                "method"           : s["method"],
                "radius_px"        : s.get("radius"),
                "crop_w"           : crop.shape[1] if (crop is not None and crop.size > 0) else None,
                "crop_h"           : crop.shape[0] if (crop is not None and crop.size > 0) else None,
                "parasite_area_px2": s.get("parasite_area"),
                "cx"               : ctr[0] if ctr else None,
                "cy"               : ctr[1] if ctr else None,
                "x1": coords[0], "y1": coords[1],
                "x2": coords[2], "y2": coords[3],
                "rejection_log"    : " | ".join(rej) if rej else "",
            })
    print(f"  CSV salvato: {out_path}")


# ═══════════════════════════════════════════════════════════════
#  UTILITY — pairing img ↔ gt
# ═══════════════════════════════════════════════════════════════

def find_pairs(img_dir: Path, gt_dir: Path):
    """
    Trova coppie (img, gt) con lo stesso stem.
    Tollerante sulle estensioni: img.png ↔ gt.png, img.jpg ↔ gt.png, ecc.
    """
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        gt_candidates = [g for g in gt_dir.glob(img_path.stem + ".*")
                         if g.suffix.lower() in IMG_EXTENSIONS]
        if not gt_candidates:
            print(f"  [WARN] GT mancante per: {img_path.name}")
            continue
        pairs.append((img_path, gt_candidates[0]))
    return pairs


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RBC Crop Extractor — MP-IDB Malaria Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python RBC_extractor_Malariae.py
  python RBC_extractor_Malariae.py --dataset /dati/MP-IDB --output /risultati
  python RBC_extractor_Malariae.py --limit 20          # test rapido su 20 campioni
  python RBC_extractor_Malariae.py --no-diag           # salta diagnostiche (più veloce)
  python RBC_extractor_Malariae.py --show              # mostra finestra per ogni campione
        """
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR,
                        help=f"Cartella radice del dataset (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT_DIR,
                        help=f"Cartella di output (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--no-diag", action="store_true",
                        help="Non salvare diagnostiche PNG")
    parser.add_argument("--show",    action="store_true",
                        help="Mostra diagnostica interattiva (richiede display)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Processa solo i primi N campioni")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    img_dir     = dataset_dir / "img"
    gt_dir      = dataset_dir / "gt"
    output_dir  = Path(args.output)
    diag_dir    = output_dir / "diagnostics"

 
    ok = True
    for d, label in [(img_dir, "img"), (gt_dir, "gt")]:
        if not d.exists():
            print(f"[ERRORE] Cartella '{label}' non trovata: {d}")
            print("  Struttura attesa: <dataset>/img/  e  <dataset>/gt/")
            ok = False
    if not ok:
        return

    save_diag = SAVE_DIAGNOSTICS and not args.no_diag
    if save_diag:
        diag_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(img_dir, gt_dir)
    if not pairs:
        print("[ERRORE] Nessuna coppia img/gt trovata. Controlla naming e estensioni.")
        return
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"\n  Dataset    : {dataset_dir.resolve()}")
    print(f"  Campioni   : {len(pairs)}")
    print(f"  Output     : {output_dir.resolve()}")
    print(f"  Diagnostica: {'sì → ' + str(diag_dir) if save_diag else 'no'}")
    print("─" * 62)

    stats   = []
    t_start = time.time()

    for i, (img_path, gt_path) in enumerate(pairs):
        sample = img_path.stem
        print(f"  [{i+1:>4}/{len(pairs)}] {sample:<36}", end=" ", flush=True)

        img = cv2.imread(str(img_path))
        gt  = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if img is None or gt is None:
            label = "load_error_img" if img is None else "load_error_gt"
            print(f"✗  [ERRORE lettura file]")
            stats.append({"sample": sample, "method": label,
                          "radius": None, "coords": None, "crop": None,
                          "parasite_area": None, "centroid": None,
                          "lab_binary": None})
            continue

        result = extract_rbc_crop(img, gt)
        result["sample"] = sample
        stats.append(result)

        crop = result.get("crop")
        crop_mask = result.get("crop_mask")
        masked_crop = result.get("masked_crop")

        if crop is not None and crop.size > 0:
            cv2.imwrite(str(output_dir / f"{sample}_crop.png"), crop)
        if crop_mask is not None and crop_mask.size > 0:
            cv2.imwrite(str(output_dir / f"{sample}_rbc_mask.png"), crop_mask)
        if masked_crop is not None and masked_crop.size > 0:
            cv2.imwrite(str(output_dir / f"{sample}_masked_crop.png"), masked_crop)


        method = result["method"]
        radius = result.get("radius", "—")
        icon   = "✓" if not any(k in method for k in ("fail","no_","error")) else "✗"
        rej    = result.get("rejection_log", [])
        rej_str = f"  [{'; '.join(rej)}]" if rej else ""
        print(f"{icon}  {method:<36}  r={radius}px{rej_str}")

        if save_diag:
            save_diagnostic(img, gt, result, sample,
                            str(diag_dir / f"{sample}_diag.png"))
        if args.show:
            diag_path = str(diag_dir / f"{sample}_diag.png")
            frame = cv2.imread(diag_path)
            if frame is not None:
                cv2.imshow("Diagnostica — premi un tasto per continuare", frame)
                cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n  Completato in {elapsed:.1f}s  "
          f"({elapsed/len(pairs):.2f}s per campione)")

    print_report(stats)
    csv_path = output_dir / "report.csv"
    save_csv_report(stats, str(csv_path))

    if save_diag:
        print(f"  Diagnostiche: {diag_dir}")
    print()


if __name__ == "__main__":
    main()