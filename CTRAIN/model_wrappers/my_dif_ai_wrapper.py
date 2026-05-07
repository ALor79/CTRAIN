import copy
import torch

from smac.utils.configspace import get_config_hash

from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified import shi_train_model
from CTRAIN.util import seed_ctrain

class DiffAIModelWrapper(CTRAINWrapper):
    """
    Wrapper class for training models using SHI-IBP method. For details, see Shi et al. (2021) Fast certified robust training with short warmup. https://proceedings.neurips.cc/paper/2021/file/988f9153ac4fd966ea302dd9ab9bae15-Paper.pdf
    """
    
    def __init__(
        self,
        model,
        input_shape,
        eps,
        num_epochs,
        train_eps_factor=1,
        optimizer_func=torch.optim.Adam,
        lr=0.0005,
        warm_up_epochs=1,
        ramp_up_epochs=70,
        lr_decay_factor=.2,
        lr_decay_milestones=(80, 90),
        gradient_clip=10,
        l1_reg_weight=0.000001,
        shi_reg_weight=.5,
        shi_reg_decay=True,
        checkpoint_save_path=None,
        checkpoint_save_interval=10,
        bound_opts=dict(conv_mode='patches', relu='adaptive'),
        device=torch.device('cuda')):
        """
        Initializes the DiffAIModelWrapper.

        Args:
            model (torch.nn.Module): The model to be trained.
            input_shape (tuple): Shape of the input data.
            eps (float): Epsilon value describing the perturbation the network should be certifiably robust against.
            num_epochs (int): Number of epochs for training.
            train_eps_factor (float): Factor for training epsilon.
            optimizer_func (torch.optim.Optimizer): Optimizer function.
            lr (float): Learning rate.
            warm_up_epochs (int): Number of warm-up epochs, i.e. epochs where the model is trained on clean loss.
            ramp_up_epochs (int): Number of ramp-up epochs, i.e. epochs where the epsilon is gradually increased to the target train epsilon.
            lr_decay_factor (float): Learning rate decay factor.
            lr_decay_milestones (tuple): Milestones for learning rate decay.
            gradient_clip (float): Gradient clipping value.
            l1_reg_weight (float): L1 regularization weight.
            shi_reg_weight (float): SHI regularization weight.
            shi_reg_decay (bool): Whether to decay SHI regularization during the ramp up phase.
            checkpoint_save_path (str): Path to save checkpoints.
            checkpoint_save_interval (int): Interval for saving checkpoints.
            bound_opts (dict): Options for bounding according to the auto_LiRPA documentation.
            device (torch.device): Device to run the training on.
        """
        super().__init__(model,
                        eps,
                        input_shape,
                        train_eps_factor,
                        lr,
                        optimizer_func,
                        bound_opts,
                        device,
                        checkpoint_save_path=checkpoint_save_path,
                        checkpoint_save_interval=checkpoint_save_interval)
        self.cert_train_method = 'ZONOTOPE'
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_epochs = warm_up_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_milestones = lr_decay_milestones
        self.gradient_clip = gradient_clip
        self.l1_reg_weight = l1_reg_weight
        self.shi_reg_weight = shi_reg_weight
        self.shi_reg_decay = shi_reg_decay
        self.start_kappa = 0
        self.end_kappa = 0
        self.optimizer_func = optimizer_func
        
    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        """
        Trains the model using the SHI-IBP method.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader for validation data.
            start_epoch (int, optional): Epoch to start training from. Initialises learning rate and epsilon schedulers accordingly. Defaults to 0.
            end_epoch (int, optional): Epoch to prematurely end training at. Defaults to None.

        Returns:
            (auto_LiRPA.BoundedModule): Trained model.
        """
        eps_std = self.train_eps / train_loader.std if train_loader.normalised else torch.tensor(self.train_eps)
        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1))
        
        trained_model = shi_train_model(
            original_model=self.original_model,
            hardened_model=self.bounded_model,
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            num_epochs=self.num_epochs,
            eps=self.train_eps,
            eps_std=eps_std,
            eps_schedule=(self.warm_up_epochs, self.ramp_up_epochs),
            eps_scheduler_args={'start_kappa': self.start_kappa, 'end_kappa': self.end_kappa},
            optimizer=self.optimizer,
            lr_decay_schedule=self.lr_decay_milestones,
            lr_decay_factor=self.lr_decay_factor,
            n_classes=self.n_classes,
            gradient_clip=self.gradient_clip,
            l1_regularisation_weight=self.l1_reg_weight,
            shi_regularisation_weight=self.shi_reg_weight,
            shi_reg_decay=self.shi_reg_decay,
            results_path=self.checkpoint_path,
            checkpoint_save_interval=self.checkpoint_save_interval,
            device=self.device
        )
        
        return trained_model
    
