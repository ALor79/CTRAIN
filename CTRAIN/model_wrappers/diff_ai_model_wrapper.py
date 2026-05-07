import numpy as np
import torch

from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified import diff_ai_train_model


class DiffAIModelWrapper(CTRAINWrapper):
    """
    Wrapper class for training models using DiffAI-style certified training.

    Supports IBP bounds (baseline) and hybrid zonotope bounds with configurable
    ReLU transformers (boxy, switch, smooth).

    Reference: Mirman et al. (2018) Differentiable Abstract Interpretation for
    Provably Robust Neural Networks. https://proceedings.mlr.press/v80/mirman18b.html
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
        lr_decay_factor=0.1,
        lr_decay_milestones=(15, 25),  #changed to 80,90 refrenced from __main__#444 in diffai
        gradient_clip=1, #changed from 10 taken from __main__#260 in diffai
        l1_reg_weight=0.000001,
        shi_reg_weight=1,
        shi_reg_decay=True,
        start_kappa=1,
        end_kappa=0.5,
        bound_method='ibp',
        relu_transformer='boxy',
        use_errors=True,
        checkpoint_save_path=None,
        checkpoint_save_interval=10,
        bound_opts=dict(conv_mode='patches', relu='adaptive'),
        device=torch.device('cuda'),
    ):
        """
        Args:
            model (torch.nn.Module): The model to be trained.
            input_shape (tuple): Shape of the input data (C, H, W).
            eps (float): Epsilon value describing the perturbation the network should be certifiably robust against.
            num_epochs (int): Number of epochs for training.
            train_eps_factor (float): Factor for training epsilon.
            optimizer_func (torch.optim.Optimizer): Optimizer function.
            lr (float): Learning rate.
            warm_up_epochs (int): Number of warm-up epochs, i.e. epochs where the model is trained on clean loss.
            ramp_up_epochs (int): Number of ramp-up epochs, i.e. epochs where the epsilon is gradually increased to the target train epsilon.
            lr_decay_factor (float): Learning rate decay factor.
            lr_decay_milestones (tuple): Milestones for learning rate decay.
            gradient_clip (float): Gradient clipping value. (Max gradient norm; None disables clipping)
            l1_reg_weight (float): L1 regularization weight.
            shi_reg_weight (float): Shi regularization weight.
            shi_reg_decay (bool): Whether to decay Shi regularization during the ramp up phase.
            start_kappa (float): Starting value of kappa, the clean-loss weight. Equivalent to
                        (1 - bw) in DiffAI's LinMix(a=Point(), b=Box(), bw=Lin(...)):
                        kappa=1 means train on clean loss only; kappa=0 means certified
                        loss only. Defaults to 1.
            end_kappa (float): Final kappa value once epsilon reaches its target. Defaults to 0.5,
                        matching the DiffAI paper experiments (AllExperimentsSerial.sh) which use
                        bw=Lin(0, 0.5, ...) — i.e. the certified loss weight never exceeds 0.5,
                        keeping an equal clean/certified split at full epsilon.
            bound_method (str): 'ibp' or 'zonotope'.
            relu_transformer (str): For zonotope; 'boxy', 'switch', or 'smooth'.
            use_errors (bool): Whether to track explicit error terms in zonotope.
                               Memory-intensive; keep False for CNNs.
            checkpoint_save_path (str): Path to save checkpoints.
            checkpoint_save_interval (int): Interval for saving checkpoints.
            bound_opts (dict): Options passed to BoundedModule.
            device (torch.device): Device to run the training on.
        


        """
        super().__init__(
            model,
            eps,
            input_shape,
            train_eps_factor,
            lr,
            optimizer_func,
            bound_opts,
            device,
            checkpoint_save_path=checkpoint_save_path,
            checkpoint_save_interval=checkpoint_save_interval,
        )
        self.cert_train_method = 'diff_ai'
        self.num_epochs = num_epochs
        self.warm_up_epochs = warm_up_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_milestones = lr_decay_milestones
        self.gradient_clip = gradient_clip
        self.l1_reg_weight = l1_reg_weight
        self.shi_reg_weight = shi_reg_weight
        self.shi_reg_decay = shi_reg_decay
        # kappa schedules the trade-off between clean and certified loss,
        # equivalent to (1 - bw) in DiffAI's LinMix(a=Point(), b=Box(), bw=Lin(...)).
        # start_kappa=1  -> pure clean loss at the start of ramp-up;
        # end_kappa=0.5  -> equal clean/certified split at full epsilon, matching the
        #                   DiffAI paper experiments (AllExperimentsSerial.sh: bw=Lin(0,0.5,...)).
        self.start_kappa = start_kappa
        self.end_kappa = end_kappa
        self.bound_method = bound_method
        self.relu_transformer = relu_transformer
        self.use_errors = use_errors

    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        """
        Train the model using DiffAI certified training.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            start_epoch (int): Epoch to resume from.
            end_epoch (int or None): Epoch to stop at (defaults to num_epochs).

        Returns:
            auto_LiRPA.BoundedModule: The trained bounded model.
        """
        eps_std = self.train_eps / train_loader.std if train_loader.normalised else torch.tensor(self.train_eps)
        eps_std = torch.reshape(eps_std, (*eps_std.shape, 1, 1))

        trained_model = diff_ai_train_model(
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
            eps_scheduler_args={
                # kappa = 1 - bw in DiffAI's LinMix notation.
                # Controls the clean-loss weight during the epsilon ramp-up phase.
                'start_kappa': self.start_kappa,
                'end_kappa': self.end_kappa,
            },
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
            device=self.device,
            bound_method=self.bound_method,
            relu_transformer=self.relu_transformer,
            use_errors=self.use_errors,
        )

        return trained_model

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir, cert_eval_samples=1000, nat_loss_weight=1, adv_loss_weight=1, cert_loss_weight=1):
        raise NotImplementedError
