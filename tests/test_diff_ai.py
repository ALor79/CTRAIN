"""
Trains CNN7_Shi on MNIST for a short run and compares:
  - IBP bounds (baseline)
  - Zonotope bounds with crelu_boxy
  - Zonotope bounds with crelu_switch
  - Zonotope bounds with crelu_smooth
  - TODO creluNIPS
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_wrappers import DiffAIModelWrapper

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

IN_SHAPE    = [1, 28, 28]
EPS         = 0.3
NUM_EPOCHS  = 5
WARM_UP     = 1
RAMP_UP     = 2
LR          = 0.0005
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIGS = [
    dict(label='IBP (baseline)',    bound_method='ibp',      relu_transformer='boxy'),
    dict(label='Zonotope - boxy',   bound_method='zonotope', relu_transformer='boxy'),
    dict(label='Zonotope - switch', bound_method='zonotope', relu_transformer='switch'),
    dict(label='Zonotope - smooth', bound_method='zonotope', relu_transformer='smooth'),
]

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

train_loader, test_loader = load_mnist(val_split=False)

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

results = []

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")

    wrapper = DiffAIModelWrapper(
        model=CNN7_Shi(in_shape=IN_SHAPE),
        input_shape=IN_SHAPE,
        eps=EPS,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        warm_up_epochs=WARM_UP,
        ramp_up_epochs=RAMP_UP,
        bound_method=cfg['bound_method'],
        relu_transformer=cfg['relu_transformer'],
        use_errors=False,
        checkpoint_save_path=None,
        device=DEVICE,
    )

    wrapper.train_model(train_loader=train_loader)

    wrapper.eval()
    # For bound_method='zonotope', evaluate() uses bound_zonotope for certified
    # accuracy (matching the training bound). For bound_method='ibp', it falls
    # back to super().evaluate() which uses CROWN (tighter than IBP, conservative).
    std_acc, cert_acc, _ = wrapper.evaluate(test_loader=test_loader)

    results.append(dict(label=cfg['label'], std_acc=std_acc, cert_acc=cert_acc))
    print(f"  Standard accuracy : {std_acc:.4f}")
    print(f"  Certified accuracy: {cert_acc:.4f}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  Summary ({NUM_EPOCHS} epochs, eps={EPS}, MNIST)")
print(f"{'='*60}")
print(f"{'Method':<30} {'Std Acc':>10} {'Cert Acc':>10}")
print(f"{'-'*50}")
for r in results:
    print(f"{r['label']:<30} {r['std_acc']:>10.4f} {r['cert_acc']:>10.4f}")
