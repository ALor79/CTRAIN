from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_wrappers import ShiIBPModelWrapper

train_loader, test_loader = load_cifar10(val_split=False)
in_shape = [3, 32, 32]

model = CNN7_Shi(in_shape=in_shape)
wrapped_model = ShiIBPModelWrapper(model=model, input_shape=in_shape, eps=2/255, num_epochs=160)

wrapped_model.train_model(train_loader)
std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)