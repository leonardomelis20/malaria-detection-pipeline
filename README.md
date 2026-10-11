# 🔬 Multi-Species Malaria Parasite Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-AI%20in%20Healthcare%20%26%20Digital%20Pathology-green)
![Status](https://img.shields.io/badge/Status-Completed%20%2F%20Research-brightgreen)
![License](https://img.shields.io/badge/License-MIT-orange)

An end-to-end computational framework for automated multi-species classification of *Plasmodium* parasites (*P. falciparum*, *P. vivax*, *P. malariae*, *P. ovale*) from microscopic red blood cell (RBC) images on the **MP-IDB** dataset.

Developed as part of the B.Sc. Thesis in Computer Science (*110/110 cum laude*) at the University of Cagliari, supervised by Prof. Cecilia Di Ruberto and Prof. Luca Zedda.

---

## 📌 Key Highlights & Methodological Contributions

- **Patient-Aware Splitting (Zero Data Leakage):** Designed manual patient-group splits (`group_id`) ensuring strict independence between training, validation, and the 58-sample held-out test set.
- **RBC Crop Extraction Pipeline:** Automated segmentation using LAB color space (A-channel), Otsu thresholding, elliptical fitting, and Circular Hough Transform as fallback.
- **Heterogeneous Backbone Benchmarking:** Evaluated 6 diverse architectures:
  - **ImageNet-pretrained:** `ResNet50`, `ConvNeXt Tiny`, `Swin Transformer (Swin-T)`, `ViT-B/16`.
  - **Domain-Specific Foundation Models:** `RedDino` (RBC-pretrained ViT) and `DinoBloom` (hematology ViT).
- **Dual Classification Paradigms:**
  1. *Traditional ML Classifiers* (Logistic Regression, Random Forest, k-NN) on pre-extracted Deep Features & Radiomic Features (scikit-image).
  2. *End-to-End Fine-Tuning* across `head_only`, `full`, and `LoRA` (Low-Rank Adaptation) regimes.
- **Robust Evaluation:** Evaluated across both **Intra-dataset** and **Out-of-Distribution (OOD)** cross-species generalization scenarios.

---

## 🏆 Key Results

| Approach | Best Configuration | Intra-Dataset Macro F1 | Key Finding |
| :--- | :--- | :---: | :--- |
| **Deep Features** | ConvNeXt Tiny + Logistic Regression | **0.9789** | Linear probing on high-quality pre-trained features outperforms full fine-tuning on small datasets. |
| **End-to-End Fine-Tuning** | ConvNeXt Tiny (`head_only`) | **0.9686** | Head-only adaptation maintains strong generalization with only 0.01% trainable parameters. |
| **Radiomic Features** | 61 Features + Random Forest | **0.9566** | Highly competitive baseline using non-deep analytical descriptors without GPU overhead. |
| **OOD Generalization** | DinoBloom (Domain-Specific) | *Top OOD* | Pre-training on biological microscopy images significantly improves cross-species adaptability. |

---

## 🏗 Architecture & Pipeline Flow

```mermaid
graph TD
    A[Raw Microscopy Images - MP-IDB] --> B[Segmentation Module: LAB Color Space & Elliptic Fit]
    B --> C[Automated RBC Crop Extraction]
    C --> D[Patient-Aware Splitting: Group-ID Partitioning]
    D --> E[Prevents Patient-Level Leakage]
    E --> F[Feature & Training Pipeline: 6 Backbones + Radiomics]
    F --> G[Dual Paradigms: Feature Extraction vs Fine-Tuning Head/Full/LoRA]
    G --> H[Evaluation & Error Analysis: Intra-dataset & OOD Metrics]```
🚀 Quickstart
Prerequisites
Python 3.12

PyTorch with CUDA support

Setup & Run
Bash
# Clone the repository
git clone https://github.com/leonardomelis20/malaria-detection-pipeline.git
cd malaria-detection-pipeline

# Install dependencies
pip install -r requirements.txt

# Inspect extracted features or run inference
python scripts/phase4/inspect_convnext_tiny.py
👨‍💻 Author
Leonardo Melis

M.Sc. Student in Artificial Intelligence and Health (AIBH) – Università degli Studi di Torino

B.Sc. Graduate in Computer Science (110/110 cum laude) – Università degli Studi di Cagliari

GitHub Profile
