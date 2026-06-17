"""Per questo test:
in run_intra.py: firma diventa
run_intra_experiment(model_name, fine_tune_mode, fold,
                       num_epochs=NUM_EPOCHS, output_subdir="intra")
e dentro: output_dir = OUTPUT_DIR / output_subdir / model_name / ...
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd

from config import *
from data.dataset import MalariaDataset, get_transforms, get_dataloader
from models.build_model import build_model
from training.trainer import train_model
from training.losses import get_loss_function
from evaluation.evaluate import run_evaluation, save_results
from run_intra import run_intra_experiment

TEST_MODELS = ["ViT-B", "RedDino"]
TEST_FOLD = 1
TEST_EPOCHS = 3

for model_name in TEST_MODELS:
    for mode in FINE_TUNE_MODES:
        if mode == "lora" and not MODEL_CONFIG[model_name]["supports_lora"]:
            continue
        print(f">>> {model_name} | {mode} | fold{TEST_FOLD}")
        run_intra_experiment(model_name, mode, TEST_FOLD,
                             num_epochs = TEST_EPOCHS,
                             output_subdir="orchestration_test")