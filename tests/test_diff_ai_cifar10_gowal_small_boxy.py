"""
Trains GowalConvSmall on CIFAR-10 using DiffAI with zonotope bounds, crelu_boxy,
and use_errors=True (explicit error tracking — not equivalent to IBP).

Memory note: use_errors=True initialises one zonotope generator per input element
(3*32*32 = 3072 generators), so peak activation memory is ~3072x larger than
use_errors=False. BATCH_SIZE is reduced accordingly.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from CTRAIN.model_definitions import GowalConvSmall
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_wrappers import DiffAIModelWrapper

IN_SHAPE   = [3, 32, 32]
EPS        = 8 / 255
NUM_EPOCHS = 200
WARM_UP    = 1
RAMP_UP    = 2
LR         = 0.001
BATCH_SIZE = 16
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = load_cifar10(batch_size=BATCH_SIZE, val_split=False)

wrapper = DiffAIModelWrapper(
    model=GowalConvSmall(in_shape=tuple(IN_SHAPE), n_classes=10, dataset='cifar10'),
    input_shape=IN_SHAPE,
    eps=EPS,
    num_epochs=NUM_EPOCHS,
    lr=LR,
    warm_up_epochs=WARM_UP,
    ramp_up_epochs=RAMP_UP,
    end_kappa=0,
    bound_method='zonotope',
    relu_transformer='boxy',
    use_errors=True,
    checkpoint_save_path=None,
    device=DEVICE,
)

wrapper.train_model(train_loader=train_loader)

wrapper.eval()
std_acc, cert_acc, _ = wrapper.evaluate(test_loader=test_loader, test_samples=1000)

print(f"\n{'='*60}")
print(f"  GowalConvSmall — Zonotope/boxy use_errors=True — {NUM_EPOCHS} epochs, eps={EPS:.5f}, CIFAR-10")
print(f"{'='*60}")
print(f"  Standard accuracy : {std_acc:.4f}")
print(f"  Certified accuracy: {cert_acc:.4f}")
