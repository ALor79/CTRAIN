"""
Test script for DiffAI certified training with IBP and hybrid zonotope bounds.

Trains CNN7_Shi on MNIST for a short run and compares:
  - IBP bounds (baseline)
  - Zonotope bounds with crelu_boxy
  - Zonotope bounds with crelu_switch
  - Zonotope bounds with crelu_smooth

Usage:
    python test_diff_ai.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from auto_LiRPA import BoundedModule

from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_mnist
from CTRAIN.train.certified.diff_ai import shi_train_model
from CTRAIN.eval import eval_certified

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

IN_SHAPE     = [1, 28, 28]
EPS          = 0.3
NUM_EPOCHS   = 5       # short run — increase for real training (e.g. 120)
WARM_UP      = 1
RAMP_UP      = 2
LR           = 0.0005
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIGS = [
    dict(label='IBP (baseline)',          bound_method='ibp',      relu_transformer='boxy',   use_errors=False),
    dict(label='Zonotope - boxy',         bound_method='zonotope', relu_transformer='boxy',   use_errors=False),
    dict(label='Zonotope - switch',       bound_method='zonotope', relu_transformer='switch', use_errors=False),
    dict(label='Zonotope - smooth',       bound_method='zonotope', relu_transformer='smooth', use_errors=False),
]

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

train_loader, test_loader = load_mnist(val_split=False)
eps_std = torch.tensor(EPS / train_loader.std).reshape(-1, 1, 1)

# ------------------------------------------------------------------
# Helper: build a fresh BoundedModule for each run
# ------------------------------------------------------------------

def build_models():
    model = CNN7_Shi(in_shape=IN_SHAPE).to(DEVICE)
    example_input = torch.ones([1, *IN_SHAPE], device=DEVICE)
    bounded = BoundedModule(
        model=model,
        global_input=example_input,
        bound_opts=dict(conv_mode='patches', relu='adaptive'),
        device=DEVICE,
    )
    optimizer = torch.optim.Adam(bounded.parameters(), lr=LR)
    return model, bounded, optimizer

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

results = []

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")

    model, bounded, optimizer = build_models()

    shi_train_model(
        original_model=model,
        hardened_model=bounded,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        eps=EPS,
        eps_std=eps_std,
        eps_schedule=(WARM_UP, RAMP_UP),
        optimizer=optimizer,
        n_classes=10,
        results_path=None,
        device=DEVICE,
        bound_method=cfg['bound_method'],
        relu_transformer=cfg['relu_transformer'],
        use_errors=cfg['use_errors'],
    )

    bounded.eval()
    std_acc, cert_acc = eval_certified(
        model=bounded,
        test_loader=test_loader,
        eps=eps_std,
        n_classes=10,
        device=DEVICE,
    )

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
