import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from build_model import build_model

model_names = ["Swin-T"]
tune_mode = "head_only"

for m in model_names:
    print(f"\n{'='*50}")
    print(f"MODELLO: {m}")
    print(f"{'='*50}")
    model = build_model(m, tune_mode)
    for name, layer in model.backbone.named_modules():
        if name:
            print(name)