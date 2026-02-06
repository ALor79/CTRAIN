from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_cifar10  
from CTRAIN.model_wrappers import CrownIBPModelWrapper
import torch

train_loader, test_loader = load_cifar10(val_split=False)
in_shape = [1, 28, 28]  
model = CNN7_Shi(in_shape=in_shape)

wrapped_model = CrownIBPModelWrapper(
    model=model, 
    input_shape=in_shape, 
    eps=0.3,  
    num_epochs=70,
    
    # CROWN-IBP specific parameters
    warm_up_epochs=0,
    ramp_up_epochs=20,
    
    # Learning rate schedule
    lr=0.0005,  # 5×10^-4 initial learning rate check Mao et. al. CTBench paper
    lr_decay_milestones=(160, 180),
    lr_decay_factor=0.2,  # Decay by 0.2 at milestones check Mao et. al. CTBench paper
    
    optimizer_func=torch.optim.Adam,
    
    # Regularization
    gradient_clip=10,
    l1_reg_weight=0.00000,
    shi_reg_weight=0.5,
    shi_reg_decay=True,
    
    loss_fusion=True,

)

wrapped_model.train_model(train_loader)
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)

print(f"Standard Accuracy: {std_acc:.2%}")
print(f"Certified Accuracy: {cert_acc:.2%}")
print(f"Adversarial Accuracy: {adv_acc:.2%}")
