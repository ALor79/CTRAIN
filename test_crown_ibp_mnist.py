import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_wrappers import CrownIBPModelWrapper

train_loader, test_loader = load_mnist(val_split=False)
in_shape = [1, 28, 28]

model = CNN7_Shi(in_shape=in_shape)
wrapped_model = CrownIBPModelWrapper(model=model, input_shape=in_shape, eps=0.3, num_epochs=120,ramp_up_epochs=20,loss_fusion=False)

wrapped_model.train_model(train_loader)
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)