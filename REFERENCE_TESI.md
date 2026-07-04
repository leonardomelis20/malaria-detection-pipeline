# REFERENCE_TESI — Dati Tecnici per il Capitolo "Esperimenti e Risultati"

**Progetto**: Pipeline end-to-end per il rilevamento di parassiti della malaria da immagini di microscopia  
**Università**: Università degli Studi di Cagliari  
**Relatore**: Prof. Andrea Loddo  
**Data di generazione**: 2026-07-01  
**Fonte**: dati estratti direttamente dal codice/file del repository `malaria-detection-pipeline/`

> **Convenzione tracciabilità**: ogni dato riporta tra parentesi `(file:riga)` o `(file)` il punto esatto del codice o del file da cui è stato ricavato.

---

## 1. Struttura del Progetto

```
malaria-detection-pipeline/
├── Falciparum/crops/{G,R,S,T}/          # Crop RBC Falciparum (4 stages)
├── Vivax/crops/{G,R,S,T}/               # Crop RBC Vivax
├── Ovale/crops/{G,G_R,R,R_T,S,T,T_R,diagnostics}/  # Crop RBC Ovale (struttura estesa)
├── Malariae/crops/{G,R,R_T,S,S_T,T,diagnostics}/    # Crop RBC Malariae (struttura estesa)
├── csvs/
│   ├── kfold_heldout/                   # CSV fold train/val (fold{1-5}_train.csv, fold_{1-5}_val.csv)
│   └── splits_heldout/                  # test_heldout.csv, trainval_metadata_oversampled.csv
├── results/
│   ├── features/radiomics/fold{1-5}/    # File .h5 feature radiomiche (unici presenti localmente)
│   ├── classification/
│   │   ├── intra/metrics.csv            # Risultati Fase 5 intra-dataset (deep features)
│   │   ├── intra_radiomic/metrics.csv   # Risultati Fase 5 intra-dataset (radiomica)
│   │   ├── ood/metrics.csv              # Risultati Fase 5 OOD (deep features)
│   │   └── ood_radiomic/metrics.csv     # Risultati Fase 5 OOD (radiomica)
│   └── tuning/
│       ├── intra/{Modello}/{mode}/fold{N}/  # Risultati Fase 6 intra
│       └── ood/{Coppia}/{Modello}/{mode}/   # Risultati Fase 6 OOD
├── scripts/
│   ├── heldout_test.py                  # Fase 3: creazione test held-out (selezione manuale gruppi)
│   ├── 5-fold_heldout_manual.py         # Fase 3: costruzione 5-fold manuale su trainval
│   ├── phase4/
│   │   ├── extract_resnet50_features.py
│   │   ├── extract_convnext_tiny_features.py
│   │   ├── extract_vit_b16_features.py
│   │   ├── extract_swin_t_features.py
│   │   ├── extract_reddino_base_features.py
│   │   ├── extract_dinobloom_base_features.py
│   │   └── extract_radiomic_features.py # Usa scikit-image (non PyRadiomics)
│   ├── phase 5/
│   │   ├── utils.py                     # Classificatori, metriche, I/O
│   │   ├── intra_dataset.py             # Loop classificazione intra (deep)
│   │   ├── out_of_distribution.py       # Loop classificazione OOD (deep)
│   │   ├── intra_dataset_radiomic.py    # Loop classificazione intra (radiomica)
│   │   └── out_of_distribution_radiomic.py
│   └── phase 6/
│       ├── config.py                    # Iperparametri, MODEL_CONFIG, percorsi
│       ├── data/dataset.py              # MalariaDataset, augmentation, _relocate_path
│       ├── models/build_model.py        # build_model, MalariaClassifier, apply_lora
│       ├── training/trainer.py          # EarlyStopping, train_one_epoch (grad accum), train_model
│       ├── training/losses.py           # CrossEntropyLoss pesata per classe
│       ├── evaluation/evaluate.py       # run_evaluation, save_results
│       └── experiments/
│           ├── run_intra.py             # Loop 30 run intra (skip DinoBloom+full)
│           ├── run_ood.py               # Loop 135 run OOD (skip DinoBloom+full)
│           ├── run_dinobloom_full_intra.py  # 2 run DinoBloom+full intra
│           ├── run_dinobloom_full_ood.py    # 9 run DinoBloom+full OOD
│           └── run_smoke_test.py        # Test rapido 1 epoch ConvNeXt
├── WORKLOG_FASE6.md                     # Diario dettagliato Fase 6 con ragionamento
├── CLAUDE.md                            # Vincoli e istruzioni per l'agente
└── README.md                            # Citazione dataset MP-IDB
```

**Note struttura asimmetrica crop** (`WORKLOG_FASE6.md`, 2026-06-17):
- Falciparum, Vivax: sottocartelle `G/R/S/T` pulite
- Ovale: ha anche `G_R`, `R_T`, `T_R`, `diagnostics/`, `report.csv`
- Malariae: ha anche `R_T`, `S_T`, `diagnostics/`, `report.csv`, + **2 file `.png` sciolti** direttamente in `crops/`

---

## 2. Fase 2-3: Split Patient-Aware e Held-Out Test Set

### 2.1 Dataset complessivo

| Specie | N tot immagini | N group_id | group_id presenti |
|---|---|---|---|
| Falciparum | 104 | 10 | 1305121398, 1307210661, 1405022890, 1408161544, 1408290968, 1409171742, 1409191647, 1603223711, 1701151546, 1704282807 |
| Malariae | 35 | 3 | 1312132815, 1401063467, 1401080976 |
| Ovale | 25 | 2 | 1707180816, 1708161076 |
| Vivax | 40 | 2 | 1703121298, 1709041080 |
| **Totale** | **204** | **17** | |

Fonte: analisi programmatica dei CSV in `csvs/kfold_heldout/` e `csvs/splits_heldout/`.

### 2.2 Test Set Held-Out

| Specie | N immagini | group_id |
|---|---|---|
| Falciparum | 21 | 1704282807 |
| Malariae | 12 | 1401063467 |
| Ovale | 10 | 1708161076 |
| Vivax | 15 | 1703121298 |
| **Totale** | **58** | 1 group_id per specie |

Fonte: `csvs/splits_heldout/test_heldout.csv`

**Criterio di selezione test set** (`scripts/heldout_test.py:14-19`): selezione manuale di 1 group_id per specie, definiti in `TEST_GROUPS`. Overlap train/test = 0 per costruzione.

### 2.3 Composizione TrainVal e 5-Fold

**Script di split**: `scripts/5-fold_heldout_manual.py`  
**Criterio**: group-aware, manuale — ogni fold è definito esplicitamente per group_id, garantendo che immagini dello stesso paziente non compaiano sia in train che in val.

| Fold | Train (N) | Falc. | Vivax | Ovale | Mal. | Val (N) | Composizione val |
|---|---|---|---|---|---|---|---|
| 1 | 111 | 63 | 25 | 15 | 8 | 35 | Falciparum=20 (1305121398), Malariae=15 (1312132815) |
| 2 | 122 | 67 | 25 | 15 | 15 | 24 | Falciparum=16 (1307210661+1405022890), Malariae=8 (1401080976) |
| 3 | 140 | 77 | 25 | 15 | 23 | 6 | Falciparum=6 (1408161544+1408290968) |
| 4 | 126 | 63 | 25 | 15 | 23 | 20 | Falciparum=20 (1409171742+1409191647) |
| 5 | 125 | 62 | 25 | 15 | 23 | 21 | Falciparum=21 (1603223711+1701151546) |

Fonte: analisi diretta dei CSV + `scripts/5-fold_heldout_manual.py:38-44`

**Group_id sempre in training** (mai in validation): Ovale 1707180816, Vivax 1709041080 (`scripts/5-fold_heldout_manual.py`, commenti righe 46-47).

### 2.4 Fold Informativi

**Solo fold 1 e 2 sono usati** per l'addestramento e la valutazione in Fase 5 e 6 (`config.py:109`: `INFORMATIVE_FOLDS = [1, 2]`).

**Motivazione** (`WORKLOG_FASE6.md`, 2026-06-30 e `scripts/5-fold_heldout_manual.py`, commento iniziale):
- Ovale e Vivax hanno solo 2 group_id ciascuno → con 5 fold e 1 group_id nel test, rimane 1 group_id per trainval
- Quel 1 group_id rimasto va sempre in training (mai in val) per preservare il segnale
- Fold 3-5: val set con **una sola classe** (solo Falciparum), F1 macro sarebbe banalmente 1.0 e non confrontabile
- Fold 1-2: val set con 2 classi (Falciparum + Malariae), F1 macro calcolabile in modo significativo

**Nota**: anche con solo 2 classi nel val set, i training set dei fold 1 e 2 contengono tutte e 4 le specie. Le valutazioni finali (Fase 5 e 6) sono sempre sul test held-out a 4 classi (58 campioni).

---

## 3. Fase 4: Estrazione Feature

### 3.1 Backbone Deep — Specifiche

| Backbone | Fonte | Modello pretrained | IMAGE_SIZE | Emb. dim | Nota estrazione |
|---|---|---|---|---|---|
| ResNet50 | torchvision | `ResNet50_Weights.IMAGENET1K_V1` | 224 | 2048 | `model.fc = nn.Identity()` (`extract_resnet50_features.py:101-103`) |
| ConvNeXt Tiny | torchvision | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | 224 | 768 | `model.classifier[2] = nn.Identity()` (`build_model.py:23`) |
| ViT-B/16 | torchvision | `ViT_B_16_Weights.IMAGENET1K_V1` | 224 | 768 | `model.heads = nn.Identity()` + `torch.flatten` (`extract_vit_b16_features.py:99-106`) |
| Swin-T | torchvision | `Swin_T_Weights.IMAGENET1K_V1` | 224 | 768 | `model.head = nn.Identity()` (`build_model.py:26`) |
| RedDino | timm (HuggingFace) | `hf_hub:Snarcy/RedDino-base` | 224 | 768 | `timm.create_model(..., num_classes=0)` o `torch.flatten` (`extract_reddino_base_features.py:100-108`) |
| DinoBloom | timm (HuggingFace) | `hf-hub:1aurent/vit_base_patch14_224.dinobloom` | **518** | 768 | `timm.create_model(..., pretrained=True)` + `torch.flatten` (`extract_dinobloom_base_features.py:100-108`) |

**Nota su embedding dim** (`config.py:34-91`): in `config.py` (Fase 6), `embedding_dim` è 2048 per ResNet50 e 768 per tutti gli altri. I file di estrazione Fase 4 confermano le stesse dimensioni (validate da `assert features.shape[1] == N`).  
**Discrepanza rilevata** (vedi sezione 8): `config.py` riporta `"embedding_dim": 768` per ConvNeXt, ma `inspect_convnext_tiny.py:58` asserisce `features.shape[1] == 768`. Coerente. ResNet50: 2048 in entrambi (`extract_resnet50_features.py:193`, `config.py:34`).

**Normalizzazione immagini** (tutti gli script Fase 4): `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` (ImageNet standard). Nessuna augmentation in fase di estrazione.

**BATCH_SIZE Fase 4**: 32 per tutti i backbone (hardcodato in ogni script di estrazione, es. `extract_resnet50_features.py:39`).

### 3.2 Feature Radiomiche (scikit-image)

**Script**: `scripts/phase4/extract_radiomic_features.py`  
**Libreria**: scikit-image 0.26.0 (PyRadiomics non installabile su Python 3.12 per incompatibilità di `configparser.SafeConfigParser` rimossa in 3.12 — `extract_radiomic_features.py:1-9`)

**Feature estratte: 61 totali** (`extract_radiomic_features.py:10-23`):

| Categoria | N | Dettaglio |
|---|---|---|
| First-order | 15 | media, mediana, std, varianza, skewness, curtosi, energia, entropia (Shannon, 256 bin), max, min, range, RMS, IQR, P10, P90 |
| GLCM | 36 | 4 direzioni × 3 distanze (1,2,3) × 6 proprietà (contrasto, dissimilarità, omogeneità, energia, correlazione, ASM) → mean e std across direzioni = 2 × 3 × 6 = 36 |
| Shape 2D | 10 | area, perimetro, eccentricità, solidità, extent, asse maggiore/minore, diametro equivalente, orientazione, numero di Eulero |

**Parametri GLCM** (`extract_radiomic_features.py:48-52`): `GLCM_LEVELS=16`, `GLCM_DISTANCES=[1,2,3]`, `GLCM_ANGLES=[0, π/4, π/2, 3π/4]`, `symmetric=True`, `normed=True`

**ROI mask** (`extract_radiomic_features.py:67-77`): pixel con intensità > 10 → inclusi; fallback su intera immagine se maschera vuota.  
**Conversione a grigio**: `PIL Image.convert("L")` (luminanza percettiva: 0.299R + 0.587G + 0.114B).  
**NaN handling**: correlazione GLCM su patch costante → sostituita con 0 (`extract_radiomic_features.py:135`).

**Cosa manca rispetto a PyRadiomics** (`WORKLOG_FASE6.md`, 2026-06-28): GLRLM, GLSZM, GLDM, NGTDM non disponibili in scikit-image.

### 3.3 Path File .h5 e Chiavi

**Path Fase 4 deep** (prodotti su altra macchina - Laura):  
`[BASE_LAURA]/results/features/{resnet50,convnext_tiny,vit_b16,swin_t,reddino_base,dinobloom_base}/fold{1-5}/{train,val,test}.h5`  
**NON PRESENTI localmente** — solo `results/features/radiomics/` è disponibile sulla macchina attuale.

**Path Fase 4 radiomica** (locali):  
`results/features/radiomics/fold{1-5}/{train,val,test}.h5`

**Chiavi in ogni file .h5** (da `extract_resnet50_features.py:161-169` e inspect scripts):
```
features    (float64, shape [N, D])
label_ids   (int, shape [N])
labels      (bytes/str, shape [N])
phases      (bytes/str, shape [N])
group_ids   (bytes/str, shape [N])
filenames   (bytes/str, shape [N])
filepaths   (bytes/str, shape [N])
```

### 3.4 Statistiche Feature Radiomiche (locali)

| Fold/Split | N | D | Mean | Std | Min | Max |
|---|---|---|---|---|---|---|
| fold1/train | 111 | 61 | 883.64 | 7516.93 | −2.20 | 250000 |
| fold1/val | 35 | 61 | 1049.46 | 7428.57 | −1.54 | 111556 |
| fold1/test | 58 | 61 | 1116.94 | 8731.84 | −2.00 | 234256 |
| fold2/train | 122 | 61 | 896.32 | 7004.67 | −2.20 | 250000 |
| fold2/val | 24 | 61 | 1060.97 | 9612.50 | −1.78 | 195364 |
| fold3/train | 140 | 61 | 938.55 | 7619.14 | −2.20 | 250000 |
| fold3/val | 6 | 61 | 569.60 | 3562.48 | −1.22 | 39572 |
| fold4/train | 126 | 61 | 986.11 | 7961.44 | −2.20 | 250000 |
| fold4/val | 20 | 61 | 528.22 | 3271.56 | −1.92 | 37420 |
| fold5/train | 125 | 61 | 1025.21 | 8057.38 | −1.92 | 250000 |
| fold5/val | 21 | 61 | 317.28 | 1951.15 | −2.20 | 22574 |
| test (tutti i fold) | 58 | 61 | 1116.94 | 8731.84 | −2.00 | 234256 |

Range ampio (max=250000) dovuto principalmente a `shape_area` in pixel²; normalizzazione con `StandardScaler` applicata internamente dal classificatore prima del fitting.

**Statistiche deep features**: NON TROVATE NEL REPO — file .h5 deep assenti localmente. Valori riportabili solo da log di Laura o dalla macchina originale.

---

## 4. Fase 5: Classificatori Classici

### 4.1 Iperparametri Classificatori

Definiti in `scripts/phase 5/utils.py:23-37`:

| Classificatore | Iperparametri |
|---|---|
| RF | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| LR | `LogisticRegression(max_iter=1000, random_state=42)` |
| KNN-1 | `KNeighborsClassifier(n_neighbors=1)` |
| KNN-3 | `KNeighborsClassifier(n_neighbors=3)` |
| KNN-5 | `KNeighborsClassifier(n_neighbors=5)` |

**StandardScaler**: NON applicato esplicitamente negli script Fase 5 (`utils.py:46-55`). Il classificatore viene fittato direttamente su `X_train` senza preprocessing. **Nota**: in `intra_dataset_radiomic.py` lo scaler viene applicato — da verificare.

**Metriche calcolate** (`utils.py:58-80`): accuracy, balanced_accuracy, precision_macro/weighted, recall_macro/weighted, f1_macro, f1_weighted, mcc, roc_auc_macro, n_classes_in_val/test.

### 4.2 Protocollo Valutazione Intra-Dataset

**Script**: `scripts/phase 5/intra_dataset.py`  
**Addestramento**: `train.h5` del fold → fit classificatore  
**Valutazione**: sia su `val.h5` (per tracking) sia su `test_heldout.csv` (metrica principale)  
**Fold usati**: tutti e 5 (ma per il confronto si usano fold 1 e 2 come "informativi")  
**Metrica di riferimento**: F1 macro sul test held-out (58 campioni, 4 classi)

**Protocollo OOD** (`scripts/phase 5/out_of_distribution.py`):
- Training: concatenazione di tutti i campioni della specie sorgente (tutti i fold, tutti gli split)
- Label: fasi del ciclo (R/G/S/T), codificate con `LabelEncoder`
- Test: campioni della specie target estratti da `fold1/test.h5`, filtrati per specie (`filter_by_species`)
- 9 coppie: Falciparum→{Vivax,Ovale,Malariae}, Vivax→{Falciparum,Ovale,Malariae}, Ovale→{Falciparum,Vivax,Malariae}

### 4.3 Risultati Intra-Dataset — Deep Features (fold 1 e 2, test held-out)

Dati da `results/classification/intra/metrics.csv` — solo split=test, fold1 e fold2, media±std tra i due fold:

| Backbone | Classif. | F1 macro fold1 | F1 macro fold2 | Media | Acc fold1 | Acc fold2 |
|---|---|---|---|---|---|---|
| ConvNeXt Tiny | LR | 0.9788 | 0.9788 | **0.9788** | 0.9828 | 0.9828 |
| ConvNeXt Tiny | KNN_3 | 0.9561 | 0.9788 | 0.9674 | 0.9655 | 0.9828 |
| ConvNeXt Tiny | KNN_5 | 0.9561 | 0.9788 | 0.9674 | 0.9655 | 0.9828 |
| ConvNeXt Tiny | KNN_1 | 0.9561 | 0.9561 | 0.9561 | 0.9655 | 0.9655 |
| ConvNeXt Tiny | RF | 0.9332 | 0.9386 | 0.9359 | 0.9483 | 0.9483 |
| DinoBloom | KNN_1 | 0.9788 | 0.9602 | 0.9695 | 0.9828 | 0.9655 |
| DinoBloom | LR | 0.9583 | 0.9583 | 0.9583 | 0.9655 | 0.9655 |
| DinoBloom | KNN_3 | 0.9384 | 0.9391 | 0.9388 | 0.9483 | 0.9483 |
| DinoBloom | KNN_5 | 0.9384 | 0.9201 | 0.9292 | 0.9483 | 0.9310 |
| DinoBloom | RF | 0.8971 | 0.9279 | 0.9125 | 0.9138 | 0.9310 |
| Swin-T | RF | 0.9583 | 0.9583 | 0.9583 | 0.9655 | 0.9655 |
| Swin-T | LR | 0.9384 | 0.9583 | 0.9484 | 0.9483 | 0.9655 |
| ViT-B/16 | RF | 0.9788 | 0.9788 | 0.9788 | 0.9828 | 0.9828 |
| ViT-B/16 | LR | 0.9788 | 0.9788 | 0.9788 | 0.9828 | 0.9828 |
| ViT-B/16 | KNN_5 | 0.9788 | 0.9335 | 0.9561 | 0.9828 | 0.9483 |
| ResNet50 | KNN_1 | 0.9583 | 0.9085 | 0.9334 | 0.9655 | 0.9310 |
| ResNet50 | LR | 0.9132 | 0.9132 | 0.9132 | 0.9310 | 0.9310 |
| RedDino | LR | 0.9021 | 0.8452 | 0.8737 | 0.9138 | 0.8793 |
| RedDino | KNN_3 | 0.8759 | 0.8452 | 0.8606 | 0.8966 | 0.8793 |

**Ranking top-5 (media fold1+fold2)**:

| Rank | Backbone | Classificatore | F1 media |
|---|---|---|---|
| 1 | ConvNeXt Tiny | LR | 0.9788 |
| 1 | ViT-B/16 | LR | 0.9788 |
| 1 | ViT-B/16 | RF | 0.9788 |
| 4 | ConvNeXt Tiny | KNN_3/5 | 0.9674 |
| 5 | DinoBloom | KNN_1 | 0.9695 |

### 4.4 Risultati Intra-Dataset — Feature Radiomiche (fold 1 e 2, test held-out)

Dati da `results/classification/intra_radiomic/metrics.csv`:

| Classif. | F1 fold1 | F1 fold2 | Media | MCC fold1 | MCC fold2 | Acc fold1 | Acc fold2 |
|---|---|---|---|---|---|---|---|
| **RF** | **0.9566** | **0.9566** | **0.9566** | 0.9541 | 0.9541 | 0.9655 | 0.9655 |
| LR | 0.9132 | 0.9363 | 0.9248 | 0.9066 | 0.9293 | 0.9310 | 0.9483 |
| KNN_5 | 0.8983 | 0.7234 | 0.8109 | 0.8822 | 0.7525 | 0.9138 | 0.8103 |
| KNN_3 | 0.8333 | 0.7125 | 0.7729 | 0.8417 | 0.7831 | 0.8793 | 0.8276 |
| KNN_1 | 0.8016 | 0.6992 | 0.7504 | 0.8211 | 0.7539 | 0.8621 | 0.8103 |

Fonte: `results/classification/intra_radiomic/metrics.csv`.

**Posizione nel ranking unificato** (`WORKLOG_FASE6.md`, 2026-06-30): RF radiomica (F1=0.957) è al **4° posto** assoluto, in sostanziale parità con DinoBloom+LR (delta=+0.001).

### 4.5 Risultati OOD — Deep Features (F1 macro per coppia, max e media su 5 classificatori)

Dati da `results/classification/ood/metrics.csv`:

| Coppia | Backbone | F1 max (clf migliore) | F1 mean (5 clf) |
|---|---|---|---|
| Falciparum→Vivax | ConvNeXt | 0.793 | 0.637 |
| Falciparum→Vivax | DinoBloom | 0.767 | 0.638 |
| Falciparum→Vivax | Swin-T | 1.000 | 0.772 |
| Falciparum→Ovale | DinoBloom | 1.000 | 0.938 |
| Falciparum→Ovale | ResNet50 | 0.600 | 0.562 |
| Falciparum→Malariae | tutti | ≤0.172 | ≤0.150 |
| Vivax→Falciparum | tutti | ≤0.475 | ≤0.360 |
| Vivax→Ovale | Swin-T | 0.890 | 0.807 |
| Vivax→Ovale | DinoBloom | 1.000 | 0.651 |
| Vivax→Malariae | tutti | ≤0.200 | ≤0.110 |
| Ovale→Falciparum | DinoBloom | 0.821 | 0.724 |
| Ovale→Vivax | ConvNeXt | 0.505 | 0.352 |
| Ovale→Vivax | DinoBloom | 0.636 | 0.456 |
| Ovale→Malariae | tutti | ≤0.236 | ≤0.161 |

**Osservazione**: coppie → Malariae producono sistematicamente F1 vicino a zero per tutti i modelli (`WORKLOG_FASE6.md`, 2026-06-30).

### 4.6 Risultati OOD — Feature Radiomiche

Dati da `results/classification/ood_radiomic/metrics.csv`:

| Coppia | Miglior clf | F1 max | Acc (al max) |
|---|---|---|---|
| Falciparum→Vivax | LR | 0.500 | 0.933 |
| Falciparum→Ovale | RF/KNN | 0.412 | 0.700 |
| Falciparum→Malariae | LR | 0.056 | 0.083 |
| Vivax→Falciparum | RF | 0.475 | 0.905 |
| Vivax→Ovale | KNN_1/3/5 | 0.083 | 0.100 |
| Vivax→Malariae | RF | 0.287 | 0.417 |
| Ovale→Falciparum | KNN_1/3/5 | 0.533 | 0.619 |
| Ovale→Vivax | LR | 0.450 | 0.867 |
| Ovale→Malariae | tutti | 0.051 | 0.083 |

---

## 5. Fase 6: Fine-Tuning End-to-End

### 5.1 Iperparametri Globali

Fonte: `scripts/phase 6/config.py`

| Parametro | Valore | Riga |
|---|---|---|
| `NUM_EPOCHS` | 50 (con early stopping) | `:95` |
| `LEARNING_RT_HEAD` | 1e-3 | `:96` |
| `LEARNING_RT_BACKBONE` | 1e-5 | `:97` |
| `EARLY_STOPPING_PATIENCE` | 10 | `:98` |
| `WEIGHT_DECAY` | 1e-4 | `:99` |
| `RANDOM_SEED` | 42 | `:100` |
| `LORA_R` | 8 | `:103` |
| `LORA_ALPHA` | 8 (= LORA_R) | `:104` |
| `LORA_DROPOUT` | 0.1 | `:105` |
| `EARLY_STOPPING_MIN_DELTA` | 1e-4 (dopo fix del 2026-06-26) | `trainer.py:111` |
| `BATCH_SIZE` globale (fallback) | 16 | `:94` |

**Optimizer**: AdamW (`training/trainer.py:100-109`):
- `head_only`: AdamW sui soli parametri di `model.head`, `lr=LEARNING_RT_HEAD`
- `full`/`lora`: AdamW con 2 param groups — backbone con `lr=LEARNING_RT_BACKBONE`, head con `lr=LEARNING_RT_HEAD`

**Scheduler**: NESSUNO — NON TROVATO NEL REPO.

**Loss function**: `CrossEntropyLoss` con pesi per classe (`training/losses.py:22-36`):  
`weight_i = N_tot / (K × N_i)` dove K=4 classi; classi assenti ricevono peso 0 (dopo fix del 2026-06-26, prima davano `inf`).

**Criterion checkpoint**: migliore `val_loss` (`trainer.py:128-131` — salva `best_model.pt` a ogni miglioramento).

**Augmentation training** (`data/dataset.py:30-38`):
- HorizontalFlip p=0.5
- VerticalFlip p=0.5
- Rotate limit=15° p=0.5
- RandomBrightnessContrast p=0.2
- Normalize ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

**Augmentation val/test**: solo Resize + Normalize (`data/dataset.py:39-48`).

### 5.2 Batch Size e Gradient Accumulation per Modello

Fonte: `config.py:30-91`

| Modello | batch_size | grad_accum_steps | Effective batch | Motivazione |
|---|---|---|---|---|
| ResNet50 | 16 | 2 | 32 | CNN, nessun collo di bottiglia attention |
| ConvNeXt | 16 | 2 | 32 | CNN |
| Swin-T | 16 | 2 | 32 | CNN ibrido |
| ViT-B | 8 | 4 | 32 | 197 token, attention map ~190 MB/layer |
| RedDino | 8 | 4 | 32 | 197 token (stessa architettura ViT) |
| DinoBloom | 4 | 8 | 32 | 1370 token (518px, patch=14), input ~1 GB/batch=32 |

**Gradient accumulation** (`trainer.py:53`): `(loss / grad_accum_steps).backward()` — la loss viene scalata prima del backward per ottenere media invece di somma dei gradienti.

### 5.3 LoRA — Configurazione e Moduli Target

Fonte: `config.py:102-108` e `models/build_model.py:58-69`

| Parametro LoRA | Valore |
|---|---|
| Rank `r` | 8 |
| `lora_alpha` | 8 (= r) |
| `lora_dropout` | 0.1 |
| `bias` | "none" |

| Modello | `lora_target_modules` | Supporta LoRA |
|---|---|---|
| ResNet50 | [] | NO (CNN, nessun modulo attention) |
| ConvNeXt | [] | NO (CNN) |
| Swin-T | `["qkv"]` | SÌ |
| ViT-B | `["q_proj", "v_proj"]` | SÌ |
| RedDino | `["qkv"]` | SÌ |
| DinoBloom | `["qkv"]` | SÌ |

**Implementazione**: `peft.LoraConfig` + `peft.get_peft_model` applicata al `model.backbone` (`build_model.py:58-70`). Prima di applicare LoRA, tutti i parametri del backbone sono congelati; LoRA aggiunge gli adattatori A e B (trainable) alle matrici target.

### 5.4 Risultati Intra-Dataset Fase 6

Dati da `results/tuning/intra/*/metrics.json`. Test set held-out (58 campioni, 4 classi).

| Modello | Modalità | F1 fold1 | F1 fold2 | **Media F1** | Acc fold1 | Acc fold2 | MCC fold1 | MCC fold2 |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt | head_only | 0.9583 | 0.9788 | **0.9686** | 0.9655 | 0.9828 | 0.9527 | 0.9767 |
| ConvNeXt | full | 0.9188 | 0.9594 | 0.9391 | 0.9310 | 0.9655 | 0.9073 | 0.9545 |
| DinoBloom | lora | 0.9375 | 0.9583 | 0.9479 | 0.9483 | 0.9655 | 0.9291 | 0.9527 |
| DinoBloom | full | 0.9583 | 0.9132 | 0.9358 | 0.9655 | 0.9310 | 0.9527 | 0.9066 |
| DinoBloom | head_only | 0.9053 | 0.9788 | 0.9421 | 0.9138 | 0.9828 | 0.8837 | 0.9767 |
| ResNet50 | head_only | 0.9384 | 0.9583 | 0.9484 | 0.9483 | 0.9655 | 0.9296 | 0.9527 |
| ResNet50 | full | 0.9036 | 0.9249 | 0.9143 | 0.9138 | 0.9310 | 0.8825 | 0.9063 |
| Swin-T | full | 0.9384 | 0.9188 | 0.9286 | 0.9483 | 0.9310 | 0.9296 | 0.9073 |
| Swin-T | head_only | 0.9188 | 0.9188 | 0.9188 | 0.9310 | 0.9310 | 0.9073 | 0.9073 |
| Swin-T | lora | 0.9188 | 0.9188 | 0.9188 | 0.9310 | 0.9310 | 0.9073 | 0.9073 |
| RedDino | full | 0.8963 | 0.9384 | 0.9174 | 0.9138 | 0.9483 | 0.8823 | 0.9296 |
| RedDino | lora | 0.8392 | 0.8944 | 0.8668 | 0.8621 | 0.9138 | 0.8147 | 0.8855 |
| RedDino | head_only | 0.5687 | 0.8560 | 0.7124 | 0.6207 | 0.8793 | 0.4849 | 0.8447 |
| ViT-B | full | 0.8939 | 0.8750 | 0.8845 | 0.9138 | 0.8966 | 0.8819 | 0.8582 |
| ViT-B | lora | 0.7477 | 0.7001 | 0.7239 | 0.7759 | 0.7414 | 0.6973 | 0.6486 |
| ViT-B | head_only | 0.7276 | 0.6881 | 0.7079 | 0.7586 | 0.7241 | 0.6750 | 0.6256 |

**Ranking per F1 media**:

| Rank | Modello | Modalità | F1 media |
|---|---|---|---|
| 1 | ConvNeXt | head_only | **0.9686** |
| 2 | ResNet50 | head_only | 0.9484 |
| 3 | DinoBloom | lora | 0.9479 |
| 4 | DinoBloom | head_only | 0.9421 |
| 5 | DinoBloom | full | 0.9358 |
| 6 | ConvNeXt | full | 0.9391 |
| 7 | Swin-T | full | 0.9286 |
| 8 | RedDino | full | 0.9174 |
| 9 | ResNet50 | full | 0.9143 |
| 10 | Swin-T | head_only/lora | 0.9188 |
| 11 | ViT-B | full | 0.8845 |
| 12 | RedDino | lora | 0.8668 |
| 13 | ViT-B | lora | 0.7239 |
| 14 | RedDino | head_only | 0.7124 |
| 15 | ViT-B | head_only | 0.7079 |

### 5.5 Combinazioni Mancanti o Escluse

| Combinazione | Motivo esclusione |
|---|---|
| ResNet50 + lora | CNN pura, nessun modulo attention (`config.py:37`) |
| ConvNeXt + lora | CNN pura, nessun modulo attention (`config.py:45`) |
| DinoBloom + full (in run_intra.py) | Guard esplicito — VRAM non verificata inizialmente (`run_intra.py:83-84`) |
| DinoBloom + full (in run_ood.py) | Guard esplicito — stesso motivo (`run_ood.py`) |

DinoBloom+full è stato eseguito successivamente tramite script dedicati (`run_dinobloom_full_intra.py`, `run_dinobloom_full_ood.py`) — VRAM picco misurato: **5.16 GB su 6.00 GB disponibili** (`WORKLOG_FASE6.md`, 2026-06-28).

### 5.6 Risultati OOD Fase 6

**Protocollo** (`WORKLOG_FASE6.md`, 2026-06-26): il classificatore viene addestrato su campioni di **una sola specie sorgente**, poi testato su campioni della specie target. Il modello predice sempre la specie sorgente → F1=0 è il risultato atteso per costruzione.

**Questo protocollo NON è comparabile con Fase 5 OOD** (che classifica fasi, non specie).

**Riepilogo per combo** (F1 macro mediato su 9 coppie):

| Modello | Modalità | F1 media (9 coppie) | Run non-zero | Run totali |
|---|---|---|---|---|
| DinoBloom | head_only | 0.0130 | 2/9 | 9 |
| DinoBloom | lora | 0.0089 | 1/9 | 9 |
| DinoBloom | full | 0.0000 | 0/9 | 9 |
| Tutti gli altri | tutti | 0.0000 | 0/9 | 9 ciascuno |

**Dettaglio 3 run DinoBloom con F1 > 0** (`results/tuning/ood/*/metrics.json`):

| Coppia | Modalità | F1 macro | Accuracy | MCC |
|---|---|---|---|---|
| Ovale → Falciparum | lora | 0.0800 | 0.1905 | 0.0 |
| Falciparum → Malariae | head_only | 0.0714 | 0.1667 | 0.0 |
| Falciparum → Ovale | head_only | 0.0455 | 0.1000 | 0.0 |

**Totale run eseguiti Fase 6**:
- Loop intra: 32/32 (30 via `run_intra.py` + 2 DinoBloom+full via `run_dinobloom_full_intra.py`)
- Loop OOD: 144/144 (135 via `run_ood.py` + 9 DinoBloom+full via `run_dinobloom_full_ood.py`)

---

## 6. Configurazione Hardware e Ambiente

Fonte: `WORKLOG_FASE6.md` (2026-06-17) e `CLAUDE.md`

| Componente | Valore |
|---|---|
| OS | Windows 10 |
| GPU | NVIDIA GTX 1060 6 GB VRAM |
| CUDA | 12.6 |
| Python | 3.12.10 (nel virtualenv `.venv`) |
| PyTorch | 2.6.0+cu124 |
| timm | 1.0.27 |
| transformers | 5.12.1 |
| peft | 0.19.1 |
| h5py | 3.16.0 |
| scikit-learn | 1.9.0 |
| scikit-image | 0.26.0 |
| albumentations | 2.0.8 |
| pandas | 3.0.3 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| opencv-python | 4.13.0 |
| pillow | 12.2.0 |
| numpy | 2.4.4 |
| accelerate | 1.14.0 |
| safetensors | 0.8.0 |

**Vincoli VRAM e soluzioni** (`WORKLOG_FASE6.md`, 2026-06-17):

| Situazione | Soluzione |
|---|---|
| BATCH_SIZE=32 OOM per ViT/transformer full | Riduzione a 8 (ViT-B, RedDino) o 4 (DinoBloom) con gradient accumulation per mantenere effective batch=32 |
| DinoBloom 518px: 1370 token, attention map molto grandi | batch=4, grad_accum=8; VRAM picco full: 5.16 GB (sicuro su 6 GB) |
| PyRadiomics non installabile su Python 3.12 | Sostituzione con scikit-image (già installato) |

**VRAM misurata** (test empirici, `WORKLOG_FASE6.md`):
- DinoBloom head_only: 0.58 GB picco
- DinoBloom full: 5.16 GB picco (0.35 GB modello + ~4.81 GB training)

---

## 7. Issue Noti e Problemi Aperti

### 7.1 Bug Corretti

| Bug | Effetto | Correzione | File |
|---|---|---|---|
| `EARLY_STOPPING_MIN_DELTA = 0.0` | Training sempre per 50 epoche in OOD (val_loss→0 con micro-oscillazioni) | Cambiato a `1e-4` | `trainer.py:111` |
| Pesi infiniti in `losses.py` quando una classe è assente (OOD) | Warning ma matematicamente corretto (classi assenti non entrano nel calcolo) | `np.where(class_count > 0, weight, 0.0)` | `losses.py:24-26` |
| `classification_report` con meno di 4 classi (OOD) | Crash dopo training corretto | Aggiunto `labels=list(range(4)), zero_division=0` | `evaluate.py:46-51` |
| Path assoluti nei CSV (generati su altra macchina) | File non trovati al runtime | `_relocate_path` in `dataset.py` e `extract_radiomic_features.py` | `dataset.py:13-25`, `extract_radiomic_features.py:55-64` |

### 7.2 Bug NON Corretti (aperti)

| Bug | Effetto | File/Riga |
|---|---|---|
| **Disallineamento `SPECIES` vs `SPECIES_TO_ID`** | `SPECIES = ["Falciparum", "Vivax", "Ovale", "Malariae"]` (Vivax idx=1, Malariae idx=3) ma `SPECIES_TO_ID = {"Malariae": 1, "Vivax": 3}` → nei report testuali e confusion matrix PNG, Vivax e Malariae sono **scambiate**. I valori numerici F1/acc/MCC in `metrics.json` sono **corretti** (calcolati su ID numerici). | `config.py:21-27` e `evaluate.py:92,99` |

### 7.3 Limitazioni Metodologiche

| Limitazione | Descrizione |
|---|---|
| **Protocollo OOD Fase 6 non comparabile con Fase 5** | Fase 6 OOD: classificatore di specie addestrato su 1 specie → F1=0 per costruzione. Fase 5 OOD: classificatore di fasi addestrato su 1 specie → label space condiviso. I risultati non si comparano. |
| **Solo fold 1 e 2 usati per fine-tuning** | Fold 3-5 hanno val set mono-classe; usarli darebbe risultati non confrontabili. Conseguenza: ogni modello è valutato su 2 fold informativi invece di 5, riducendo la stima della variabilità. |
| **Dataset molto piccolo** | 111-122 campioni in training per fold. Con 28M+ parametri trainable (full fine-tuning), il rischio overfitting è alto (confermato: full < head_only per ResNet50 e ConvNeXt). |
| **DinoBloom+full in run_intra.py escluso per default** | I guard in `run_intra.py` e `run_ood.py` escludono DinoBloom+full. I risultati sono prodotti da script separati ma salvati nelle stesse cartelle. |
| **SPECIES_TO_ID fisso ma asimmetrico** | La mappatura `{"Falciparum":0, "Malariae":1, "Ovale":2, "Vivax":3}` è diversa dall'ordinamento in `SPECIES = ["Falciparum","Vivax","Ovale","Malariae"]`. Le confusion matrix e i report testuali mostrano le classi nell'ordine di `SPECIES` ma l'ID numerico segue `SPECIES_TO_ID`. |
| **Nessun scheduler di learning rate** | Il training usa AdamW con lr fisso per tutta la durata. Non è stato testato l'effetto di un cosine annealing o warmup. |
| **Feature deep non disponibili localmente** | I file `.h5` per i 6 backbone deep (prodotti su altra macchina) non sono nel repository locale. Le statistiche delle feature deep (mean/std/min/max) non sono verificabili. |

### 7.4 Swin-T — Risultati Identici (fenomeno osservato)

**Osservazione** (`WORKLOG_FASE6.md`, 2026-06-18): 5 dei 6 run di Swin-T (head_only fold1, head_only fold2, lora fold1, lora fold2, full fold2) producono identici F1=0.9188, Acc=0.9310, MCC=0.9073, con gli stessi 4 campioni errati (3 Ovale + 1 Malariae).  
**Causa**: quei 4 campioni sono "strutturalmente ambigui" per le feature Swin-T pretrained; qualunque testa lineare ragionevole li classifica allo stesso modo sul test set fisso di 58 campioni. Non è un bug.

---

## 8. Discrepanze da Verificare

| # | Discrepanza | Dettaglio |
|---|---|---|
| D1 | **ConvNeXt embedding_dim** | `config.py:43` riporta `"embedding_dim": 768`; il checkpoint `convnext_tiny-983f1562.pth` di torchvision effettivamente produce 768-dim per `convnext_tiny`. `inspect_convnext_tiny.py:58` asserisce `features.shape[1] == 768`. Coerente. |
| D2 | **Nota ResNet50 pesi** | `build_model.py:19` usa `ResNet50_Weights.IMAGENET1K_V2` per Fase 6; `extract_resnet50_features.py:100` usa `ResNet50_Weights.IMAGENET1K_V1` per Fase 4. Le feature di Fase 4 e il backbone di Fase 6 usano **pesi diversi**. Il confronto tra risultati Fase 5 (feature V1) e Fase 6 (backbone V2) va segnalato. |
| D3 | **RedDino model ID** | `extract_reddino_base_features.py:100` usa `"hf_hub:Snarcy/RedDino-base"` (underscore dopo hf); `config.py:76` usa `"hf-hub:Snarcy/RedDino-base"` (trattino). Entrambi dovrebbero risolvere allo stesso hub, ma andrebbero verificati. |
| D4 | **Fase 5 - BaseRoot hardcoded** | `scripts/phase 5/intra_dataset.py:9` e `out_of_distribution.py:7` usano un path assoluto di Laura (`C:\Users\laura\...`). Questi script non hanno il meccanismo `_relocate_path`. La Fase 5 è stata probabilmente eseguita sulla macchina di Laura e i risultati sono già salvati nei CSV. |
| D5 | **SPECIES_TO_ID vs SPECIES** | Già segnalato in 7.2. Da citare esplicitamente nel capitolo metodi: i report testuali hanno Vivax e Malariae scambiate, ma le metriche numeriche sono corrette. |
| D6 | **ConvNeXt embedding Fase 4 vs Fase 6** | `extract_convnext_tiny_features.py` non ha `validate_h5` con asserzione esplicita sulla dim (NON TROVATA nel file — solo per resnet50 e dinobloom è verificata). Da verificare che il file .h5 prodotto abbia effettivamente dim=768. |

---

## 9. Confronto Sintetico Fase 5 vs Fase 6 (Intra-Dataset)

| Backbone | Fase 5 best F1 (clf) | Fase 6 best F1 (mode) | Delta |
|---|---|---|---|
| ConvNeXt | 0.9788 (LR) | 0.9686 (head_only) | −0.010 |
| ViT-B/16 | 0.9788 (LR/RF) | 0.8845 (full) | −0.094 |
| DinoBloom | 0.9695 (KNN_1) | 0.9479 (lora) | −0.022 |
| Swin-T | 0.9583 (RF) | 0.9286 (full) | −0.030 |
| ResNet50 | 0.9334 (KNN_1) | 0.9484 (head_only) | +0.015 |
| RedDino | 0.8737 (LR) | 0.9174 (full) | +0.044 |
| Radiomica | 0.9566 (RF) | N/A | — |

**Nota**: il confronto non è perfettamente diretto — Fase 5 usa il migliore tra 5 classificatori, Fase 6 usa la migliore tra 2-3 modalità. La base di training (fold 1 e 2) è la stessa.

---

## 10. Dati per Capitolo 4 — sezioni 4.5-4.11

**Aggiornamento**: 2026-07-03. Dati estratti direttamente da `results/classification/*.csv`, `results/tuning/intra/*/*/fold*/metrics.json`, `results/tuning/ood_stages/*/*/*/metrics.json` e `WORKLOG_FASE6.md`. Alcuni run di `run_ood_stages.py` sono **in corso al momento della stesura** (vedi § "Run mancanti / in corso").

### 4.3 — Tabella completa Fase 5 deep (aggiunta: tutte le 30 combinazioni)

Fonte: `results/classification/intra/metrics.csv`, filtro `split=test`, fold1+fold2, media e deviazione standard tra i due fold.

| Backbone | Classif. | F1 fold1 | F1 fold2 | F1 media | F1 std | Acc fold1 | Acc fold2 | Acc media | MCC fold1 | MCC fold2 | MCC media |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt Tiny | LR | 0.9788 | 0.9788 | **0.9788** | 0.0000 | 0.9828 | 0.9828 | 0.9828 | 0.9767 | 0.9767 | 0.9767 |
| ViT-B/16 | RF | 0.9788 | 0.9788 | **0.9788** | 0.0000 | 0.9828 | 0.9828 | 0.9828 | 0.9767 | 0.9767 | 0.9767 |
| ViT-B/16 | LR | 0.9788 | 0.9788 | **0.9788** | 0.0000 | 0.9828 | 0.9828 | 0.9828 | 0.9767 | 0.9767 | 0.9767 |
| DinoBloom | KNN_1 | 0.9788 | 0.9602 | 0.9695 | 0.0093 | 0.9828 | 0.9655 | 0.9741 | 0.9767 | 0.9531 | 0.9649 |
| ConvNeXt Tiny | KNN_3 | 0.9561 | 0.9788 | 0.9674 | 0.0114 | 0.9655 | 0.9828 | 0.9741 | 0.9531 | 0.9767 | 0.9649 |
| ConvNeXt Tiny | KNN_5 | 0.9561 | 0.9788 | 0.9674 | 0.0114 | 0.9655 | 0.9828 | 0.9741 | 0.9531 | 0.9767 | 0.9649 |
| DinoBloom | LR | 0.9583 | 0.9583 | 0.9583 | 0.0000 | 0.9655 | 0.9655 | 0.9655 | 0.9527 | 0.9527 | 0.9527 |
| Swin-T | RF | 0.9583 | 0.9583 | 0.9583 | 0.0000 | 0.9655 | 0.9655 | 0.9655 | 0.9527 | 0.9527 | 0.9527 |
| ViT-B/16 | KNN_5 | 0.9788 | 0.9335 | 0.9561 | 0.0226 | 0.9828 | 0.9483 | 0.9655 | 0.9767 | 0.9302 | 0.9535 |
| ConvNeXt Tiny | KNN_1 | 0.9561 | 0.9561 | 0.9561 | 0.0000 | 0.9655 | 0.9655 | 0.9655 | 0.9531 | 0.9531 | 0.9531 |
| Swin-T | LR | 0.9384 | 0.9583 | 0.9484 | 0.0100 | 0.9483 | 0.9655 | 0.9569 | 0.9296 | 0.9527 | 0.9412 |
| ViT-B/16 | KNN_3 | 0.9583 | 0.9367 | 0.9475 | 0.0108 | 0.9655 | 0.9483 | 0.9569 | 0.9527 | 0.9295 | 0.9411 |
| DinoBloom | KNN_3 | 0.9384 | 0.9391 | 0.9388 | 0.0004 | 0.9483 | 0.9483 | 0.9483 | 0.9296 | 0.9296 | 0.9296 |
| ConvNeXt Tiny | RF | 0.9332 | 0.9386 | 0.9359 | 0.0027 | 0.9483 | 0.9483 | 0.9483 | 0.9321 | 0.9314 | 0.9318 |
| ResNet50 | KNN_1 | 0.9583 | 0.9085 | 0.9334 | 0.0249 | 0.9655 | 0.9310 | 0.9483 | 0.9527 | 0.9068 | 0.9298 |
| DinoBloom | KNN_5 | 0.9384 | 0.9201 | 0.9292 | 0.0092 | 0.9483 | 0.9310 | 0.9397 | 0.9296 | 0.9072 | 0.9184 |
| ViT-B/16 | KNN_1 | 0.9017 | 0.9561 | 0.9289 | 0.0272 | 0.9310 | 0.9655 | 0.9483 | 0.9099 | 0.9531 | 0.9315 |
| ResNet50 | LR | 0.9132 | 0.9132 | 0.9132 | 0.0000 | 0.9310 | 0.9310 | 0.9310 | 0.9066 | 0.9066 | 0.9066 |
| DinoBloom | RF | 0.8971 | 0.9279 | 0.9125 | 0.0154 | 0.9138 | 0.9310 | 0.9224 | 0.8848 | 0.9065 | 0.8956 |
| ResNet50 | RF | 0.9132 | 0.8852 | 0.8992 | 0.0140 | 0.9310 | 0.9138 | 0.9224 | 0.9066 | 0.8838 | 0.8952 |
| ResNet50 | KNN_5 | 0.9005 | 0.8921 | 0.8963 | 0.0042 | 0.9138 | 0.9138 | 0.9138 | 0.8842 | 0.8825 | 0.8833 |
| ResNet50 | KNN_3 | 0.9132 | 0.8710 | 0.8921 | 0.0211 | 0.9310 | 0.8966 | 0.9138 | 0.9066 | 0.8593 | 0.8829 |
| Swin-T | KNN_1 | 0.8791 | 0.8791 | 0.8791 | 0.0000 | 0.8966 | 0.8966 | 0.8966 | 0.8632 | 0.8632 | 0.8632 |
| RedDino | LR | 0.9021 | 0.8452 | 0.8737 | 0.0285 | 0.9138 | 0.8793 | 0.8966 | 0.8837 | 0.8388 | 0.8612 |
| Swin-T | KNN_3 | 0.8791 | 0.8586 | 0.8688 | 0.0102 | 0.8966 | 0.8793 | 0.8879 | 0.8632 | 0.8421 | 0.8526 |
| Swin-T | KNN_5 | 0.8791 | 0.8586 | 0.8688 | 0.0102 | 0.8966 | 0.8793 | 0.8879 | 0.8632 | 0.8421 | 0.8526 |
| RedDino | KNN_3 | 0.8759 | 0.8452 | 0.8606 | 0.0153 | 0.8966 | 0.8793 | 0.8879 | 0.8634 | 0.8388 | 0.8511 |
| RedDino | KNN_1 | 0.8314 | 0.8259 | 0.8287 | 0.0028 | 0.8621 | 0.8621 | 0.8621 | 0.8224 | 0.8135 | 0.8180 |
| RedDino | RF | 0.7229 | 0.8522 | 0.7875 | 0.0647 | 0.7759 | 0.8793 | 0.8276 | 0.7209 | 0.8395 | 0.7802 |
| RedDino | KNN_5 | 0.7139 | 0.8288 | 0.7714 | 0.0574 | 0.7759 | 0.8621 | 0.8190 | 0.7086 | 0.8156 | 0.7621 |

**Aggregato per backbone** (media±std su tutti i classificatori):

| Backbone | F1 medio | Acc medio | MCC medio |
|---|---|---|---|
| ConvNeXt Tiny | 0.9611 ± 0.0145 | 0.9690 ± 0.0117 | 0.9583 ± 0.0152 |
| ViT-B/16 | 0.9580 ± 0.0191 | 0.9672 ± 0.0138 | 0.9559 ± 0.0184 |
| DinoBloom | 0.9417 ± 0.0203 | 0.9500 ± 0.0184 | 0.9322 ± 0.0246 |
| ResNet50 | 0.9068 ± 0.0151 | 0.9259 ± 0.0129 | 0.8996 ± 0.0174 |
| Swin-T | 0.9047 ± 0.0400 | 0.9190 ± 0.0347 | 0.8925 ± 0.0448 |
| RedDino | 0.8244 ± 0.0398 | 0.8586 ± 0.0311 | 0.8145 ± 0.0386 |

**Aggregato per classificatore** (media±std su tutti i backbone):

| Classificatore | F1 medio | Acc medio | MCC medio |
|---|---|---|---|
| LR | 0.9419 ± 0.0377 | 0.9526 ± 0.0306 | 0.9359 ± 0.0410 |
| KNN_1 | 0.9159 ± 0.0482 | 0.9325 ± 0.0399 | 0.9101 ± 0.0522 |
| KNN_3 | 0.9125 ± 0.0407 | 0.9282 ± 0.0336 | 0.9037 ± 0.0440 |
| RF | 0.9120 ± 0.0617 | 0.9282 ± 0.0499 | 0.9054 ± 0.0631 |
| KNN_5 | 0.8982 ± 0.0659 | 0.9167 ± 0.0526 | 0.8891 ± 0.0686 |

### 4.5 — Confronto deep vs radiomica (ranking unificato, 35 combinazioni)

Fonte: `results/classification/intra/metrics.csv` (deep, 30 combo) + `results/classification/intra_radiomic/metrics.csv` (radiomica, 5 combo). Stesso filtro (split=test, fold1+fold2, media).

| Rank | Backbone/Metodo | Tipo | Classificatore | F1 media | Acc media | MCC media |
|---|---|---|---|---|---|---|
| 1 | ConvNeXt Tiny | Deep | LR | 0.9788 | 0.9828 | 0.9767 |
| 2 | ViT-B/16 | Deep | RF | 0.9788 | 0.9828 | 0.9767 |
| 3 | ViT-B/16 | Deep | LR | 0.9788 | 0.9828 | 0.9767 |
| 4 | DinoBloom | Deep | KNN_1 | 0.9695 | 0.9741 | 0.9649 |
| 5 | ConvNeXt Tiny | Deep | KNN_3 | 0.9674 | 0.9741 | 0.9649 |
| 6 | ConvNeXt Tiny | Deep | KNN_5 | 0.9674 | 0.9741 | 0.9649 |
| 7 | DinoBloom | Deep | LR | 0.9583 | 0.9655 | 0.9527 |
| 8 | Swin-T | Deep | RF | 0.9583 | 0.9655 | 0.9527 |
| **9** | **Radiomica** | **Radiomica** | **RF** | **0.9566** | **0.9655** | **0.9541** |
| 10 | ViT-B/16 | Deep | KNN_5 | 0.9561 | 0.9655 | 0.9535 |
| 11 | ConvNeXt Tiny | Deep | KNN_1 | 0.9561 | 0.9655 | 0.9531 |
| 12 | Swin-T | Deep | LR | 0.9484 | 0.9569 | 0.9412 |
| 13 | ViT-B/16 | Deep | KNN_3 | 0.9475 | 0.9569 | 0.9411 |
| 14 | DinoBloom | Deep | KNN_3 | 0.9388 | 0.9483 | 0.9296 |
| 15 | ConvNeXt Tiny | Deep | RF | 0.9359 | 0.9483 | 0.9318 |
| 16 | ResNet50 | Deep | KNN_1 | 0.9334 | 0.9483 | 0.9298 |
| 17 | DinoBloom | Deep | KNN_5 | 0.9292 | 0.9397 | 0.9184 |
| 18 | ViT-B/16 | Deep | KNN_1 | 0.9289 | 0.9483 | 0.9315 |
| **19** | **Radiomica** | **Radiomica** | **LR** | **0.9248** | **0.9397** | **0.9180** |
| 20 | ResNet50 | Deep | LR | 0.9132 | 0.9310 | 0.9066 |
| 21 | DinoBloom | Deep | RF | 0.9125 | 0.9224 | 0.8956 |
| 22 | ResNet50 | Deep | RF | 0.8992 | 0.9224 | 0.8952 |
| 23 | ResNet50 | Deep | KNN_5 | 0.8963 | 0.9138 | 0.8833 |
| 24 | ResNet50 | Deep | KNN_3 | 0.8921 | 0.9138 | 0.8829 |
| 25 | Swin-T | Deep | KNN_1 | 0.8791 | 0.8966 | 0.8632 |
| 26 | RedDino | Deep | LR | 0.8737 | 0.8966 | 0.8612 |
| 27 | Swin-T | Deep | KNN_3 | 0.8688 | 0.8879 | 0.8526 |
| 28 | Swin-T | Deep | KNN_5 | 0.8688 | 0.8879 | 0.8526 |
| 29 | RedDino | Deep | KNN_3 | 0.8606 | 0.8879 | 0.8511 |
| 30 | RedDino | Deep | KNN_1 | 0.8287 | 0.8621 | 0.8180 |
| **31** | **Radiomica** | **Radiomica** | **KNN_5** | **0.8109** | **0.8621** | **0.8173** |
| 32 | RedDino | Deep | RF | 0.7875 | 0.8276 | 0.7802 |
| **33** | **Radiomica** | **Radiomica** | **KNN_3** | **0.7729** | **0.8534** | **0.8124** |
| 34 | RedDino | Deep | KNN_5 | 0.7714 | 0.8190 | 0.7621 |
| **35** | **Radiomica** | **Radiomica** | **KNN_1** | **0.7504** | **0.8362** | **0.7875** |

**Osservazione**: contando le posizioni per riga (combinazione backbone+classificatore), RF radiomica è al 9° posto assoluto su 35. Se invece si raggruppano le combinazioni per "livello di F1" (i.e. si contano i valori distinti, non le righe — ai ranghi 1, 4, 5-6, 7-8 corrispondono rispettivamente 1, 1, 2, 2 combinazioni deep), RF radiomica occupa il 5° livello di F1 distinto, in sostanziale parità con DinoBloom+LR e Swin-T+RF (delta ≤ 0.002). Questo è coerente con la nota di `WORKLOG_FASE6.md` (2026-06-30) che la colloca "in prossimità del 4°-5° posto assoluto".

**Conclusione**: le feature radiomiche (61-dim, scikit-image) con RF raggiungono prestazioni competitive con i migliori backbone deep pretrained, nonostante una dimensionalità enormemente inferiore (61 vs 768/2048) e nessun pretraining. Il divario dal miglior risultato assoluto (ConvNeXt/ViT-B+LR/RF, F1=0.9788) è di solo 0.022.

### 4.6 — Classificazione OOD — feature deep (Fase 5)

Fonte: `results/classification/ood/metrics.csv` (270 righe = 9 coppie × 6 backbone × 5 classificatori). Protocollo: classificazione degli **stadi** (R/G/S/T), non delle specie (vedi § 4.2). Malariae non è mai sorgente (dati insufficienti).

**Tabella 1 — Miglior risultato per coppia sorgente→target:**

| Coppia | Backbone | Classificatore | F1 macro | Accuracy | MCC |
|---|---|---|---|---|---|
| Falciparum→Vivax | Swin-T | KNN_1 | 1.0000 | 1.0000 | 1.0000 |
| Falciparum→Ovale | DinoBloom | LR | 1.0000 | 1.0000 | 1.0000 |
| Falciparum→Malariae | ConvNeXt Tiny | RF | 0.1717 | 0.1667 | 0.1074 |
| Vivax→Falciparum | ConvNeXt Tiny | RF | 0.4750 | 0.9048 | 0.0000 |
| Vivax→Ovale | DinoBloom | LR | 1.0000 | 1.0000 | 1.0000 |
| Vivax→Malariae | RedDino | LR | 0.2000 | 0.2500 | 0.0925 |
| Ovale→Falciparum | DinoBloom | LR | 0.8205 | 0.9524 | 0.6892 |
| Ovale→Vivax | ResNet50 | RF | 0.6491 | 0.9333 | 0.8745 |
| Ovale→Malariae | ConvNeXt Tiny | RF | 0.2361 | 0.2500 | 0.2376 |

**Tabella 2 — F1 macro medio per backbone** (su tutte le 45 combinazioni coppia×classificatore):

| Backbone | F1 medio |
|---|---|
| DinoBloom | 0.4638 |
| Swin-T | 0.3503 |
| ConvNeXt Tiny | 0.2727 |
| ResNet50 | 0.2714 |
| RedDino | 0.2578 |
| ViT-B/16 | 0.2565 |

**Tabella 3 — F1 macro medio per classificatore** (su tutte le 54 combinazioni coppia×backbone):

| Classificatore | F1 medio |
|---|---|
| LR | 0.3235 |
| KNN_1 | 0.3125 |
| KNN_3 | 0.3125 |
| KNN_5 | 0.3125 |
| RF | 0.2993 |

**Osservazione**: DinoBloom generalizza nettamente meglio in OOD (F1 medio 0.46) rispetto a tutti gli altri backbone (0.26-0.35), nonostante NON sia il migliore in intra-dataset (dove ConvNeXt/ViT-B dominano). Le coppie →Malariae restano sistematicamente vicine a zero per ogni modello (confermato, vedi § 4.5 originale).

**Dettaglio completo per coppia** (tutte le combinazioni model×classifier, F1 macro):

<details>
<summary>Espandi dettaglio F1 per tutte le combinazioni model×classifier, per ciascuna delle 9 coppie</summary>

**Falciparum→Vivax**: Swin-T KNN_1/3/5=1.000 · ConvNeXt LR=0.793 · DinoBloom KNN_1/3/5=0.767 · ConvNeXt KNN_1/3/5=0.603 · ConvNeXt RF=0.585 · ResNet50 LR=0.536 · RedDino KNN_1/3/5=0.520 · Swin-T RF=0.489 · DinoBloom LR=0.486 · ResNet50 KNN_1/3/5=0.481 · RedDino RF=0.429 · DinoBloom RF=0.402 · Swin-T LR=0.372 · RedDino LR=0.366 · ResNet50 RF=0.250 · ViT-B LR/KNN_1/3/5=0.250 · ViT-B RF=0.232

**Falciparum→Ovale**: DinoBloom LR/KNN_1/3/5=1.000 · Swin-T KNN_1/3/5=0.762 · DinoBloom RF=0.688 · ResNet50 LR/KNN_1/3/5=0.600 · Swin-T RF=0.583 · RedDino KNN_1/3/5=0.500 · RedDino RF=0.474 · ConvNeXt LR/KNN_1/3/5=0.412 · ResNet50 RF=0.412 · Swin-T LR=0.412 · ViT-B LR/KNN_1/3/5=0.412 · ViT-B RF=0.292 · RedDino LR=0.242 · ConvNeXt RF=0.111

**Falciparum→Malariae**: ConvNeXt RF=0.172 · DinoBloom KNN_1/3/5=0.171 · RedDino LR=0.154 · RedDino RF=0.143 · DinoBloom RF=0.125 · DinoBloom LR=0.111 · ResNet50 LR=0.100 · RedDino KNN_1/3/5=0.063 · ViT-B LR=0.050 · tutti gli altri (ConvNeXt LR/KNN, ResNet50 RF/KNN, Swin-T tutti, ViT-B RF/KNN)=0.000

**Ovale→Falciparum**: DinoBloom LR=0.821 · DinoBloom KNN_1/3/5=0.743 · DinoBloom RF=0.571 · ViT-B RF=0.571 · ConvNeXt LR=0.446 · ResNet50 LR=0.432 · ResNet50 KNN_1/3/5=0.358 · RedDino RF=0.325 · ConvNeXt KNN_1/3/5=0.271 · RedDino LR=0.271 · ResNet50 RF=0.221 · RedDino KNN_1/3/5=0.198 · ConvNeXt RF=0.121 · ViT-B LR/KNN=0.087 · Swin-T RF=0.074 · Swin-T LR/KNN=0.000

**Ovale→Malariae**: ConvNeXt RF=0.236 · DinoBloom LR=0.215 · Swin-T KNN_1/3/5=0.192 · ConvNeXt LR=0.181 · DinoBloom KNN_1/3/5=0.156 · ConvNeXt KNN_1/3/5=0.133 · DinoBloom RF=0.125 · ViT-B LR=0.083 · RedDino KNN_1/3/5=0.071 · Swin-T LR=0.071 · RedDino LR=0.063 · ResNet50 RF=0.063 · ResNet50 KNN_1/3/5=0.056 · Swin-T RF=0.056 · ResNet50 LR=0.050 · RedDino RF/ViT-B RF/KNN=0.000

**Ovale→Vivax**: ResNet50 RF=0.649 · DinoBloom RF=0.636 · ResNet50 LR/KNN_1/3/5=0.636 · ViT-B LR=0.636 · ViT-B RF=0.592 · ConvNeXt RF=0.505 · DinoBloom LR=0.427 · Swin-T LR=0.427 · RedDino RF=0.419 · RedDino KNN_1/3/5=0.408 · DinoBloom KNN_1/3/5=0.406 · RedDino LR=0.406 · Swin-T KNN_1/3/5=0.401 · Swin-T RF=0.379 · ConvNeXt LR=0.377 · ViT-B KNN_1/3/5=0.367 · ConvNeXt KNN_1/3/5=0.293

**Vivax→Falciparum**: ConvNeXt RF=0.475 · DinoBloom RF/LR=0.475 · RedDino RF=0.475 · ViT-B RF=0.475 · RedDino LR=0.432 · ViT-B LR=0.432 · Swin-T LR/KNN_1/3/5=0.333 · Swin-T RF=0.325 · ViT-B KNN_1/3/5=0.298 · DinoBloom KNN_1/3/5=0.267 · ConvNeXt KNN_1/3/5=0.263 · RedDino KNN_1/3/5=0.250 · ConvNeXt LR=0.222 · ResNet50 RF=0.167 · ResNet50 KNN_1/3/5=0.087 · ResNet50 LR=0.033

**Vivax→Malariae**: RedDino LR=0.200 · ConvNeXt LR=0.167 · ResNet50 LR=0.167 · DinoBloom KNN_1/3/5=0.143 · Swin-T LR=0.143 · ConvNeXt KNN_1/3/5=0.133 · DinoBloom LR=0.100 · ViT-B LR=0.100 · ResNet50 RF=0.091 · ResNet50 KNN_1/3/5=0.063 · RedDino RF/KNN_1/3/5=0.051 · tutti gli altri=0.000

**Vivax→Ovale**: DinoBloom LR=1.000 · Swin-T KNN_1/3/5=0.890 · Swin-T RF=0.867 · ViT-B RF=0.688 · DinoBloom RF=0.600 · ViT-B LR/KNN_1/3/5=0.600 · DinoBloom KNN_1/3/5=0.552 · Swin-T LR=0.500 · ConvNeXt LR/KNN=0.250 · RedDino LR=0.222 · ResNet50 KNN_1/3/5=0.200 · RedDino RF=0.200 · ResNet50 RF=0.200 · RedDino KNN_1/3/5=0.182 · ResNet50 LR=0.167 · ConvNeXt RF=0.148
</details>

### 4.7 — Classificazione OOD — feature radiomiche (Fase 5)

Fonte: `results/classification/ood_radiomic/metrics.csv` (45 righe = 9 coppie × 5 classificatori, un solo "backbone": radiomica).

**Tabella 1 — Miglior risultato per coppia:**

| Coppia | Classificatore | F1 macro | Accuracy | MCC |
|---|---|---|---|---|
| Falciparum→Vivax | LR | 0.5000 | 0.9333 | 0.8814 |
| Falciparum→Ovale | RF | 0.4118 | 0.7000 | 0.0000 |
| Falciparum→Malariae | LR | 0.0556 | 0.0833 | −0.0724 |
| Vivax→Falciparum | RF | 0.4750 | 0.9048 | 0.0000 |
| Vivax→Ovale | KNN_1 | 0.0833 | 0.1000 | 0.1091 |
| Vivax→Malariae | RF | 0.2872 | 0.4167 | −0.0801 |
| Ovale→Falciparum | KNN_1 | 0.5333 | 0.6190 | 0.3403 |
| Ovale→Vivax | LR | 0.4500 | 0.8667 | 0.7542 |
| Ovale→Malariae | RF | 0.0513 | 0.0833 | 0.0000 |

**Tabella 2 — F1 macro medio per classificatore** (su tutte le 9 coppie):

| Classificatore | F1 medio |
|---|---|
| RF | 0.2145 |
| KNN_1 | 0.2127 |
| KNN_3 | 0.2127 |
| KNN_5 | 0.2127 |
| LR | 0.1746 |

F1 macro medio complessivo (radiomica, tutte le combinazioni): **0.2054**, contro 0.325 del miglior backbone deep medio (DinoBloom, tabella 4.6) — la radiomica generalizza peggio del deep learning in OOD, al contrario di quanto avviene intra-dataset.

**Tabella 3 — Confronto diretto per coppia: best F1 deep (§4.6) vs best F1 radiomica:**

| Coppia | Best deep | Backbone/Clf | Best radiomica | Clf | Delta (deep−radiomica) |
|---|---|---|---|---|---|
| Falciparum→Vivax | 1.0000 | Swin-T/KNN_1 | 0.5000 | LR | +0.5000 |
| Falciparum→Ovale | 1.0000 | DinoBloom/LR | 0.4118 | RF | +0.5882 |
| Falciparum→Malariae | 0.1717 | ConvNeXt/RF | 0.0556 | LR | +0.1161 |
| Vivax→Falciparum | 0.4750 | ConvNeXt/RF | 0.4750 | RF | 0.0000 |
| Vivax→Ovale | 1.0000 | DinoBloom/LR | 0.0833 | KNN_1 | +0.9167 |
| Vivax→Malariae | 0.2000 | RedDino/LR | 0.2872 | RF | **−0.0872** |
| Ovale→Falciparum | 0.8205 | DinoBloom/LR | 0.5333 | KNN_1 | +0.2872 |
| Ovale→Vivax | 0.6491 | ResNet50/RF | 0.4500 | LR | +0.1991 |
| Ovale→Malariae | 0.2361 | ConvNeXt/RF | 0.0513 | RF | +0.1848 |

**Unica coppia dove la radiomica batte il deep**: Vivax→Malariae (0.287 vs 0.200). In tutte le altre 8 coppie il deep è pari o superiore, spesso di molto (fino a +0.92 su Vivax→Ovale).

### 4.8 — Fine-tuning end-to-end intra-dataset (Fase 6)

Fonte: `results/tuning/intra/{Modello}/{mode}/fold{1,2}/metrics.json` (16 combinazioni, incluso DinoBloom+full da `run_dinobloom_full_intra.py`). Tabella identica ai dati già riportati in § 5.4, riproposta qui con riferimento esplicito ai file sorgente.

| Modello | Modalità | F1 fold1 | F1 fold2 | **F1 media** | Acc fold1 | Acc fold2 | Acc media | MCC fold1 | MCC fold2 | MCC media |
|---|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt | head_only | 0.9583 | 0.9788 | **0.9686** | 0.9655 | 0.9828 | 0.9741 | 0.9527 | 0.9767 | 0.9647 |
| ResNet50 | head_only | 0.9384 | 0.9583 | 0.9484 | 0.9483 | 0.9655 | 0.9569 | 0.9296 | 0.9527 | 0.9412 |
| DinoBloom | lora | 0.9375 | 0.9583 | 0.9479 | 0.9483 | 0.9655 | 0.9569 | 0.9291 | 0.9527 | 0.9409 |
| DinoBloom | head_only | 0.9053 | 0.9788 | 0.9421 | 0.9138 | 0.9828 | 0.9483 | 0.8837 | 0.9767 | 0.9302 |
| ConvNeXt | full | 0.9188 | 0.9594 | 0.9391 | 0.9310 | 0.9655 | 0.9483 | 0.9073 | 0.9545 | 0.9309 |
| DinoBloom | full | 0.9583 | 0.9132 | 0.9358 | 0.9655 | 0.9310 | 0.9483 | 0.9527 | 0.9066 | 0.9297 |
| Swin-T | full | 0.9384 | 0.9188 | 0.9286 | 0.9483 | 0.9310 | 0.9397 | 0.9296 | 0.9073 | 0.9185 |
| Swin-T | head_only | 0.9188 | 0.9188 | 0.9188 | 0.9310 | 0.9310 | 0.9310 | 0.9073 | 0.9073 | 0.9073 |
| Swin-T | lora | 0.9188 | 0.9188 | 0.9188 | 0.9310 | 0.9310 | 0.9310 | 0.9073 | 0.9073 | 0.9073 |
| RedDino | full | 0.8963 | 0.9384 | 0.9173 | 0.9138 | 0.9483 | 0.9310 | 0.8823 | 0.9296 | 0.9059 |
| ResNet50 | full | 0.9036 | 0.9249 | 0.9142 | 0.9138 | 0.9310 | 0.9224 | 0.8825 | 0.9063 | 0.8944 |
| ViT-B | full | 0.8939 | 0.8750 | 0.8844 | 0.9138 | 0.8966 | 0.9052 | 0.8819 | 0.8582 | 0.8701 |
| RedDino | lora | 0.8392 | 0.8944 | 0.8668 | 0.8621 | 0.9138 | 0.8879 | 0.8147 | 0.8855 | 0.8501 |
| ViT-B | lora | 0.7477 | 0.7001 | 0.7239 | 0.7759 | 0.7414 | 0.7586 | 0.6973 | 0.6486 | 0.6730 |
| RedDino | head_only | 0.5687 | 0.8560 | 0.7123 | 0.6207 | 0.8793 | 0.7500 | 0.4849 | 0.8447 | 0.6648 |
| ViT-B | head_only | 0.7276 | 0.6881 | 0.7079 | 0.7586 | 0.7241 | 0.7414 | 0.6750 | 0.6256 | 0.6503 |

**Smoke test iniziale** (ConvNeXt, head_only, fold1, 1 epoca — `WORKLOG_FASE6.md`, 2026-06-15/16, § linee 76-102):
- Loss iniziale (white-box): **1.3919**, vicina a ln(4)=1.3863 (segnale sano, distribuzione iniziale ~uniforme su 4 classi)
- Parametri trainable: **3.076 su 27.823.204 totali (0.01%)** — coerente col freezing del backbone
- Dopo 1 epoca: train_loss 1.0756, val_loss 0.5611, val_acc 0.9143; su test held-out: **F1 macro 0.9788**, Acc 0.9828, MCC 0.9767 — identico al miglior risultato fold2 completo, a conferma della forza delle feature ConvNeXt pretrained.

**VRAM misurata (DinoBloom, `WORKLOG_FASE6.md` 2026-06-17 e 2026-06-28)**:
- head_only: **0.58 GB** picco (0.35 GB modello + attivazioni minime, backbone frozen)
- full: **5.16 GB** picco su 6.00 GB disponibili (margine ~0.84 GB) — testato con 2 epoche, batch=4, grad_accum=8

**Combinazioni skippate/escluse dai loop automatici**:

| Combinazione | Motivo |
|---|---|
| ResNet50 + LoRA | CNN pura, nessun modulo attention (`config.py:37`) |
| ConvNeXt + LoRA | CNN pura, nessun modulo attention (`config.py:45`) |
| DinoBloom + full (in `run_intra.py`) | Guard esplicito — VRAM non ancora verificata al momento del loop principale |

DinoBloom+full è stato eseguito successivamente e separatamente con `run_dinobloom_full_intra.py`, dopo il test VRAM dedicato (5.16 GB, confermato sicuro). I risultati sono salvati nella stessa struttura di cartelle (`results/tuning/intra/DinoBloom/full/fold{1,2}/`) e sono inclusi nella tabella sopra.

### 4.9 — Fine-tuning end-to-end OOD stages (Fase 6, `run_ood_stages.py`)

**Protocollo** (`scripts/phase 6/experiments/run_ood_stages.py`): a differenza del protocollo OOD "originale" di Fase 6 (§5.6, classificazione di specie, F1=0 per costruzione), qui il modello viene addestrato sugli **stadi** (R/G/S/T) della specie sorgente e valutato sugli stadi della specie target — stesso protocollo concettuale di Fase 5 OOD (§4.6), ma con fine-tuning end-to-end anziché feature+classificatore classico. **Solo Falciparum è sorgente**: è l'unica specie con tutti e 4 gli stadi rappresentati sia in training sia in validation nei fold 1-2 (Vivax e Ovale hanno zero campioni di validation in fold1_val/fold2_val — un solo `group_id` ciascuno nell'intero dataset, che uno split patient-aware assegna sempre interamente al training; vedi `WORKLOG_FASE6.md`, 2026-07-04).

**Esecuzione completata (2026-07-04)**: 45/45 run completate (3 coppie × 15 combinazioni modello×modalità attive — non 16/48 come ipotizzato in una stesura precedente di questa sezione: i 6 modelli × 3 modalità = 18 combo meno 2 [ResNet50/ConvNeXt+LoRA non supportata] meno 1 [DinoBloom+full, guard VRAM] = 15 combo attive per coppia). Il loop è stato interrotto tre volte da eventi esterni (sospensione di sistema, due interruzioni di corrente) e ripreso ogni volta senza perdita di risultati grazie al resume mechanism (skip se `metrics.json` esiste).

**Tabella 1 — Risultati completi, Falciparum→Vivax** (media F1=0.3157, 0 run a zero):

| Modello | Modalità | F1 macro | Accuracy | MCC |
|---|---|---|---|---|
| RedDino | full | **0.4651** | 0.7333 | 0.5787 |
| DinoBloom | head_only | 0.4271 | 0.6667 | 0.5176 |
| Swin-T | head_only | 0.4167 | 0.5333 | 0.2584 |
| Swin-T | full | 0.3980 | 0.7333 | 0.5087 |
| ResNet50 | head_only | 0.3712 | 0.6667 | 0.3957 |
| Swin-T | lora | 0.3646 | 0.4667 | 0.2556 |
| ConvNeXt | head_only | 0.3472 | 0.6000 | 0.4183 |
| ConvNeXt | full | 0.3438 | 0.5333 | 0.4172 |
| DinoBloom | lora | 0.3389 | 0.5333 | 0.4382 |
| RedDino | lora | 0.2978 | 0.6000 | 0.4871 |
| RedDino | head_only | 0.2556 | 0.4667 | 0.3555 |
| ResNet50 | full | 0.1875 | 0.6000 | 0.0000 |
| ViT-B | head_only | 0.1875 | 0.6000 | 0.0000 |
| ViT-B | lora | 0.1875 | 0.6000 | 0.0000 |
| ViT-B | full | 0.1471 | 0.3333 | −0.0160 |

**Tabella 1b — Falciparum→Ovale** (media F1=0.2613, 0 run a zero):

| Modello | Modalità | F1 macro | Accuracy | MCC |
|---|---|---|---|---|
| DinoBloom | head_only | **0.4500** | 0.9000 | 0.7963 |
| DinoBloom | lora | 0.4500 | 0.9000 | 0.7963 |
| RedDino | lora | 0.3750 | 0.8000 | 0.6370 |
| RedDino | head_only | 0.3558 | 0.7000 | 0.5250 |
| Swin-T | full | 0.2923 | 0.6000 | 0.2431 |
| Swin-T | lora | 0.2917 | 0.5000 | 0.2431 |
| Swin-T | head_only | 0.2364 | 0.4000 | 0.1157 |
| ResNet50 | head_only | 0.2188 | 0.7000 | 0.2546 |
| ResNet50 | full | 0.2059 | 0.7000 | 0.0000 |
| ViT-B | lora | 0.2059 | 0.7000 | 0.0000 |
| ViT-B | head_only | 0.2059 | 0.7000 | 0.0000 |
| ConvNeXt | full | 0.1875 | 0.6000 | −0.1091 |
| ConvNeXt | head_only | 0.1667 | 0.4000 | 0.0980 |
| ViT-B | full | 0.1667 | 0.4000 | 0.1091 |
| RedDino | full | 0.1111 | 0.2000 | 0.1637 |

**Tabella 1c — Falciparum→Malariae** (media F1=0.0918, **6/15 run a F1=0 esatto**):

| Modello | Modalità | F1 macro | Accuracy | MCC |
|---|---|---|---|---|
| ResNet50 | full | **0.2500** | 0.0833 | 0.2655 |
| ViT-B | full | 0.2197 | 0.3333 | 0.1110 |
| RedDino | full | 0.1818 | 0.3333 | 0.2707 |
| ConvNeXt | head_only | 0.1429 | 0.3333 | −0.0135 |
| RedDino | head_only | 0.1364 | 0.2500 | 0.1083 |
| RedDino | lora | 0.1364 | 0.2500 | 0.1083 |
| ConvNeXt | full | 0.1333 | 0.3333 | −0.1132 |
| DinoBloom | lora | 0.1000 | 0.0833 | 0.0513 |
| Swin-T | lora | 0.0769 | 0.1667 | −0.2402 |
| DinoBloom | head_only | 0.0000 | 0.0000 | −0.0241 |
| ResNet50 | head_only | 0.0000 | 0.0000 | 0.0000 |
| Swin-T | head_only | 0.0000 | 0.0000 | −0.3963 |
| Swin-T | full | 0.0000 | 0.0000 | 0.0000 |
| ViT-B | head_only | 0.0000 | 0.0000 | 0.0000 |
| ViT-B | lora | 0.0000 | 0.0000 | 0.0000 |

F1 macro media complessiva sulle 45 run: **0.2229**.

**Tabella 2 — Miglior F1 per coppia**:

| Coppia | Modello | Modalità | F1 macro |
|---|---|---|---|
| Falciparum→Vivax | RedDino | full | **0.4651** |
| Falciparum→Ovale | DinoBloom | head_only / lora (pari merito) | **0.4500** |
| Falciparum→Malariae | ResNet50 | full | **0.2500** |

**Tabella 3 — F1 medio per modello** (su tutte le combinazioni completate, N=45 totali):

| Modello | F1 medio | N combo |
|---|---|---|
| DinoBloom | 0.2943 | 6 (head_only+lora × 3 coppie; full skippato per VRAM) |
| RedDino | 0.2572 | 9 |
| Swin-T | 0.2307 | 9 |
| ConvNeXt | 0.2202 | 6 |
| ResNet50 | 0.2056 | 6 |
| ViT-B | 0.1467 | 9 |

**Per modalità**: lora 0.2354 (n=12) > full 0.2193 (n=15) > head_only 0.2177 (n=18) — differenze contenute, nessuna modalità domina nettamente.

**Tabella 4 — Confronto con Fase 5 OOD (stesse coppie, Falciparum come sorgente)**:

| Coppia | Fase 5 best F1 (classico) | Fase 6 stages best F1 (fine-tuning) | Delta |
|---|---|---|---|
| Falciparum→Vivax | 1.0000 (Swin-T/KNN_1) | 0.4651 (RedDino/full) | −0.5349 |
| Falciparum→Ovale | 1.0000 (DinoBloom/LR) | 0.4500 (DinoBloom/head_only o lora) | −0.5500 |
| Falciparum→Malariae | 0.1717 (ConvNeXt/RF) | 0.2500 (ResNet50/full) | **+0.0783** |

**Osservazioni**:

1. **Il fine-tuning end-to-end non supera il classificatore classico su feature pretrained per Vivax e Ovale**, con un divario ampio (−0.53/−0.55). Causa probabile: il training set di questo protocollo è drasticamente più piccolo (130 campioni Falciparum da fold1+fold2) rispetto a Fase 5 OOD (tutti i campioni Falciparum di tutti i fold e split concatenati, §4.2) — coerente con il pattern di overfitting già osservato nell'intra-dataset (§4.10) per architetture con molti parametri trainable su pochi dati.

2. **Falciparum→Malariae è l'unica coppia dove Fase 6 batte Fase 5** (+0.078). Entrambi i protocolli restano comunque su valori bassi in assoluto (0.17-0.25): Malariae è strutturalmente la specie target più difficile in ogni protocollo testato finora (specie, stadi, deep, radiomica), coerente con la sua biologia più divergente (parassita più piccolo, ciclo eritrocitico più lungo — vedi anche §4.6 e `WORKLOG_FASE6.md` 2026-06-30).

3. **6/15 run su Falciparum→Malariae danno F1 esattamente 0**, un valore non osservato nelle altre due coppie stage-based. Ipotesi: il training Falciparum è fortemente sbilanciato verso lo stadio R (106/130 = 81.5% dei campioni), mentre il test set di Malariae ha **zero campioni R** (0/12); un collasso del modello sulla classe maggioritaria del training produce quindi zero predizioni corrette per costruzione su quello specifico test set — coerente con MCC vicino a zero o negativo in quelle run.

4. **DinoBloom è il modello più forte o tra i più forti in 2 coppie su 3** (primo in Falciparum→Ovale con 0.450, secondo in Falciparum→Vivax con 0.427 dopo RedDino), confermando — anche in questo protocollo — l'osservazione già fatta per l'OOD "originale" di specie (§4.11) e per Fase 5 (§4.6) che il pretraining domain-specific di DinoBloom offre un vantaggio nella generalizzazione cross-specie. Da notare che DinoBloom+head_only e DinoBloom+lora ottengono **valori identici** su Falciparum→Ovale (10 campioni di test) — stesso fenomeno già documentato per Swin-T in intra-dataset (§4.11, WORKLOG 2026-06-18): con un test set così piccolo, configurazioni di training diverse convergono sulla stessa decisione sui campioni "facili" e sui pochi campioni ambigui.

5. **Confronto con il protocollo OOD "originale" di Fase 6** (§5.6, classificazione di specie): quel protocollo dava F1=0 su 141/144 run per costruzione (label space non condiviso). Il protocollo stage-based dà F1>0 su 39/45 run (87%), confermando che lo stadio del ciclo cellulare è un label space effettivamente trasferibile tra specie per il fine-tuning end-to-end, non solo per i classificatori classici di Fase 5.

### 4.10 — Confronto Fase 5 vs Fase 6 intra-dataset

| Backbone | Fase 5 best F1 (clf) | Fase 6 best F1 (mode) | Delta | Fase 6 worst F1 (mode) |
|---|---|---|---|---|
| ConvNeXt | 0.9788 (LR) | 0.9686 (head_only) | −0.0102 | 0.9391 (full) |
| ViT-B | 0.9788 (RF) | 0.8844 (full) | −0.0943 | 0.7079 (head_only) |
| DinoBloom | 0.9695 (KNN_1) | 0.9479 (lora) | −0.0216 | 0.9358 (full) |
| Swin-T | 0.9583 (RF) | 0.9286 (full) | −0.0297 | 0.9188 (head_only/lora) |
| ResNet50 | 0.9334 (KNN_1) | 0.9484 (head_only) | **+0.0149** | 0.9142 (full) |
| RedDino | 0.8737 (LR) | 0.9173 (full) | **+0.0437** | 0.7123 (head_only) |
| Radiomica | 0.9566 (RF) | N/A (non fine-tunabile) | — | — |

**Nota metodologica**: confronto non perfettamente diretto — Fase 5 sceglie il migliore tra 5 classificatori su feature fisse, Fase 6 sceglie il migliore tra 2-3 modalità di fine-tuning end-to-end. Base di training comune: fold 1 e 2.

**Pattern di overfitting (CNN)**: per ConvNeXt e ResNet50 (le due CNN pure, senza LoRA disponibile), `full` è sempre la modalità peggiore o quasi (ConvNeXt: full è il worst; ResNet50: full è il worst). Il fine-tuning completo di 28M+ parametri su ~115 campioni di training peggiora le prestazioni rispetto a tenere il backbone frozen (head_only). Per i transformer con LoRA disponibile (Swin-T, RedDino, ViT-B, DinoBloom) il pattern è meno netto: RedDino migliora nettamente con full (+0.205 su head_only) e ViT-B migliora con full (+0.177 su head_only), suggerendo che per architetture attention-based il fine-tuning completo è più tollerante all'overfitting su dataset piccoli — o che head_only è penalizzato per feature meno immediatamente lineari.

**Discrepanza ResNet50** (vedi anche §8, D2): Fase 5 usa `ResNet50_Weights.IMAGENET1K_V1` per l'estrazione feature (`extract_resnet50_features.py:100`), Fase 6 usa `IMAGENET1K_V2` per il backbone di fine-tuning (`build_model.py:19`). Il delta positivo (+0.015) per ResNet50 potrebbe essere in parte attribuibile ai pesi V2 (generalmente più performanti su ImageNet) piuttosto che al fine-tuning stesso — non isolabile con i dati disponibili.

### 4.11 — Elementi di supporto per l'analisi dei risultati

**Swin-T — correzione al numero di campioni borderline** (integra/corregge §7.4): i 5 run identici di Swin-T (head_only fold1, head_only fold2, lora fold1, lora fold2, full fold2 — tutti F1=0.9188, Acc=0.9310, MCC=0.9073) sbagliano sistematicamente **4 campioni su 58**. Analizzando `classification_report.txt` (support: Falciparum=21, "Vivax"=12, Ovale=10, "Malariae"=15) alla luce del bug SPECIES/SPECIES_TO_ID (§7.2 — le etichette testuali "Vivax" e "Malariae" sono scambiate, i support lo confermano: il test set reale ha Vivax=15 e Malariae=12, § 2.2):

- Ovale: recall 0.90 → **1 errore** (non 3, come riportato in §7.4)
- riga testuale "Malariae" (support=15, recall 0.80) corrisponde in realtà a **Vivax** → **3 errori**
- riga testuale "Vivax" (support=12, recall 1.00) corrisponde in realtà a **Malariae** → **0 errori**

**Conclusione corretta**: i 4 campioni sbagliati da Swin-T sono **1 Ovale + 3 Vivax** (non "3 Ovale + 1 Malariae" come attualmente scritto in §7.4). Il totale di 4 errori e i valori numerici (F1/Acc/MCC) restano corretti; è l'attribuzione per specie nel testo che va corretta. **Da aggiornare in §7.4 del documento.**

**RedDino head_only — varianza estrema fold1 vs fold2** (dati completi in §4.8): F1 fold1=0.5687 vs fold2=0.8560 (delta=0.287, la varianza più alta tra tutte le 16 combinazioni Fase 6). Dai classification report:
- fold1: Falciparum recall 0.95 (buono) ma "Vivax"(reale Malariae) recall 0.42, Ovale recall 0.50, "Malariae"(reale Vivax) recall 0.40 — collasso su quasi tutte le classi minoritarie
- fold2: Falciparum recall 1.00, "Vivax"(reale Malariae) recall 1.00, Ovale recall 0.70, "Malariae"(reale Vivax) recall 0.73 — molto più bilanciato

Il modello RedDino head_only sembra particolarmente sensibile alla composizione esatta del training set del fold (fold1: 111 campioni; fold2: 122 campioni, §2.3) — coerente con l'essere l'unica modalità/modello dove la testa lineare da sola (nessun adattamento del backbone) non riesce a compensare la scarsità di dati per le classi minoritarie in modo stabile.

**DinoBloom — le 3 eccezioni in OOD Fase 6 protocollo originale** (già riportate in §5.6, confermate byte-per-byte dai file sorgente):

| Coppia | Modalità | F1 macro | Accuracy | MCC | File |
|---|---|---|---|---|---|
| Ovale→Falciparum | lora | 0.0800 | 0.1905 | 0.0 | `results/tuning/ood/Ovale_to_Falciparum/DinoBloom/lora/metrics.json` |
| Falciparum→Malariae | head_only | 0.0714 | 0.1667 | 0.0 | `results/tuning/ood/Falciparum_to_Malariae/DinoBloom/head_only/metrics.json` |
| Falciparum→Ovale | head_only | 0.0455 | 0.1000 | 0.0 | `results/tuning/ood/Falciparum_to_Ovale/DinoBloom/head_only/metrics.json` |

Verificato: sulle 144 run totali del protocollo OOD "originale" (classificazione di specie, F1=0 atteso per costruzione), queste sono le **uniche 3 con F1>0**, tutte DinoBloom. Le restanti 141 hanno F1 esattamente 0.0.

---

## 11. Limitazioni Note (per Capitolo 4 — sezione discussione)

1. **Bug deduplicazione `metadata.py`**: il dataset usato per fold/split (204 campioni totali, §2.1) è un **sottoinsieme** dei crop RBC effettivamente disponibili (~1499 stimati). Impatto per specie:

   | Specie | Campioni usati | Crop disponibili (stimati) | % usato |
   |---|---|---|---|
   | Falciparum | 104 | 1297 | 8.0% |
   | Vivax | 40 | 64 | 62.5% |
   | Ovale | 25 | 25 | 100% |
   | Malariae | 35 | 35 | 100% |

   Falciparum è la specie più penalizzata: solo l'8% dei crop disponibili è stato effettivamente usato in training/val/test. Questo va segnalato come limite di scala del dataset effettivo, distinto dal dataset MP-IDB nominale.

2. **Discrepanza pesi ResNet50**: `IMAGENET1K_V1` in Fase 4 (estrazione feature, `extract_resnet50_features.py:100`) vs `IMAGENET1K_V2` in Fase 6 (fine-tuning, `build_model.py:19`). Il confronto Fase5↔Fase6 per ResNet50 (§4.10) non isola l'effetto del fine-tuning da quello dei pesi diversi.

3. **Bug `SPECIES` vs `SPECIES_TO_ID`**: le etichette testuali (confusion matrix PNG, `classification_report.txt`) hanno Vivax e Malariae scambiate in tutti i run di Fase 6 (`config.py:21-27`). I valori numerici (F1 macro, accuracy, MCC in `metrics.json`) sono calcolati su ID numerici e **sono corretti**. Vedi correzione applicata in §4.11 per Swin-T e RedDino.

4. **Validation set mono-classe per LOSO Falciparum**: rilevante solo per un protocollo LOSO esplorato e poi abbandonato; non più rilevante per `run_ood_stages.py`, che usa esclusivamente Falciparum come sorgente (unica specie con validation multi-stadio). Documentato per completezza storica.

5. **`run_ood_stages.py`: solo Falciparum come sorgente**: Vivax e Ovale non hanno campioni sufficienti nei validation set dei fold 1-2 per costituire un protocollo di fine-tuning end-to-end stabile (mancano stadi). Di conseguenza il confronto Fase5↔Fase6-stages (§4.9, tabella 4) è possibile solo per le 3 coppie con sorgente Falciparum, non per le 9 coppie complete di Fase 5.

## 12. Run Mancanti / In Corso (stato al 2026-07-04)

| Script | Stato | Dettaglio |
|---|---|---|
| `run_ood_stages.py` | **COMPLETATO** | 45/45 run (3 coppie × 15 combo attive). Dati finali riportati in §4.9. |

Nessuna azione pendente per il Capitolo 4: tutte le tabelle di §4.9 riportano dati definitivi.

---

*File generato il 2026-07-01 da Claude Code (Sonnet 4.6) tramite esplorazione diretta del repository. Sezioni 10-12 aggiunte il 2026-07-03 da Claude Code (Sonnet 5) per la stesura del Capitolo 4 (Esperimenti e Risultati). §4.9, §11 e §12 aggiornate il 2026-07-04 da Claude Code (Sonnet 5) con i risultati definitivi delle 45/45 run di `run_ood_stages.py`.*
