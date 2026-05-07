"""
Trains CNN7_Shi on CIFAR-10 using DiffAI with IBP bounds (baseline).
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_wrappers import DiffAIModelWrapper

IN_SHAPE   = [3, 32, 32]
EPS        = 8 / 255
NUM_EPOCHS = 5
WARM_UP    = 1
RAMP_UP    = 2
LR         = 0.0005
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = load_cifar10(val_split=False)

wrapper = DiffAIModelWrapper(
    model=CNN7_Shi(in_shape=IN_SHAPE),
    input_shape=IN_SHAPE,
    eps=EPS,
    num_epochs=NUM_EPOCHS,
    lr=LR,
    warm_up_epochs=WARM_UP,
    ramp_up_epochs=RAMP_UP,
    bound_method='ibp',
    relu_transformer='boxy',
    use_errors=False,
    checkpoint_save_path=None,
    device=DEVICE,
)

wrapper.train_model(train_loader=train_loader)

wrapper.eval()
std_acc, cert_acc, _ = wrapper.evaluate(test_loader=test_loader, test_samples=1000)

print(f"\n{'='*60}")
print(f"  IBP (baseline) — {NUM_EPOCHS} epochs, eps={EPS:.5f}, CIFAR-10")
print(f"{'='*60}")
print(f"  Standard accuracy : {std_acc:.4f}")
print(f"  Certified accuracy: {cert_acc:.4f}")
