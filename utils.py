# copied from pl_bolts but modified due to an import error in pl_bolts
# see from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import math
import copy
import warnings
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class EMA:
    def __init__(self, decay=0.99, mode='weights', warmup_epoch=0):
        self.decay = torch.tensor(decay)
        self.mode = mode
        self.model_ema = None
        self.probs_ema = None
        self.warmup_epoch = warmup_epoch

    def initialize_ema_model(self, model):
        assert model is not None, 'Model must be provided for weight-based EMA'
        self.model_ema = copy.deepcopy(model)
        # self.model_ema.to(device)
        for param in self.model_ema.parameters():
            param.requires_grad_(False)

    def put_on_device(self, device):
        self.device = device
        self.decay = self.decay.to(device)
        if self.mode == 'weights':
            self.model_ema.to(device)

    def update(self, model, probs, current_epoch):
        if current_epoch >= self.warmup_epoch:
            if self.mode == 'weights':
                if self.warmup_epoch == current_epoch:
                    self.initialize_ema_model(model)
                    self.put_on_device(next(model.parameters()).device)

                # Update EMA weights
                with torch.no_grad():
                    for (name, ema_param), (_, model_param) in zip(self.model_ema.named_parameters(),
                                                                   model.named_parameters()):
                        if 'batchnorm' in name or 'bn' in name:  # Exclude BatchNorm layers
                            continue
                        ema_param.mul_(self.decay).add_(model_param, alpha=1 - self.decay)
            elif self.mode == 'probabilities':
                if self.warmup_epoch == current_epoch:
                    self.put_on_device(next(model.parameters()).device)
                # Update EMA probabilities
                if self.probs_ema is None:
                    self.probs_ema = probs.clone()
                else:
                    self.probs_ema.mul_(self.decay).add_(probs, alpha=1 - self.decay)

    def get_probs(self, dataloader=None):
        if self.mode == 'weights':
            assert self.model_ema is not None, 'EMA model is not initialized'
            self.model_ema.eval()
            all_probs = []
            with torch.no_grad():
                for batch in dataloader:
                    inputs, _, _, _ = batch
                    inputs = inputs.to(self.device)
                    probs = torch.sigmoid(self.model_ema(inputs))
                    all_probs.append(probs)
            return torch.cat(all_probs, dim=0)
        elif self.mode == 'probabilities':
            assert self.probs_ema is not None, 'EMA probabilities are not initialized'
            return self.probs_ema.clone()


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " 
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
