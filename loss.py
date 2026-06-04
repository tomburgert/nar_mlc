import torch
import torch.nn.functional as F


class SelfAdaptiveTrainingCE(torch.nn.Module):
    def __init__(self, labels, num_classes=19, momentum=0.9, warmup_epochs=5):
        super().__init__()
        self.soft_labels = torch.Tensor(labels.copy()).cuda()
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs

    def forward(self, logits, labels, index, epoch, weight=None):
        if epoch - 1 < self.warmup_epochs:
            return F.binary_cross_entropy_with_logits(logits, labels, weight=weight)

        # obtain prob, then update running avg
        prob = torch.sigmoid(logits.detach())
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        # compute cross entropy loss, without reduction
        loss = F.binary_cross_entropy_with_logits(logits, self.soft_labels[index], weight=weight)
        return loss


class ELR(torch.nn.Module):
    """
    Early Learning Regularization (ELR) Loss.
    
    Parameters:
    -----------
    N : int
        Total number of training examples.
    num_classes : int, optional, default=19
        Number of classes in the classification problem.
    lam : float, optional, default=3
        Regularization strength.
    beta : float, optional, default=0.7
        Temporal ensembling momentum for target estimation.
    """
    def __init__(self, N, num_classes=19, lam=3, beta=0.7, device='cuda'):
        super().__init__()
        self.N = N
        self.num_classes = num_classes
        self.lam = lam
        self.beta = beta
        self.device = device
        
        # Initialize target tensor for temporal ensembling
        self.target = torch.zeros(N, num_classes, device=self.device)
        # self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, labels, index, weight=None):
        """
        Computes the Early Learning Regularization (ELR) loss.
        
        Parameters:
        -----------
        logits : torch.Tensor
            Model predictions (logits) for the current batch.
        labels : torch.Tensor
            Ground truth labels for the current batch.
        index : torch.Tensor
            Indices of the current batch samples in the dataset.
        
        Returns:
        --------
        torch.Tensor
            The computed ELR loss value.
        """
        # Sigmoid activation with clamping to avoid numerical issues
        y_pred = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
        
        # Update target tensor using exponential moving average
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * y_pred.detach()
        
        # Compute cross-entropy loss
        # ce_loss = self.bce_with_logits(logits, labels)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weight)
        
        # Compute ELR regularization term
        # elr_reg = (1 - (self.target[index] * y_pred)).log().mean()
        elr_reg = (1 - (self.target[index] * y_pred)).log()
        if weight is not None:
            elr_reg = elr_reg * weight  # Apply the weights
        elr_reg = elr_reg.mean()
        
        # Final loss
        loss = ce_loss + self.lam * elr_reg
        return loss


class ELR2(torch.nn.Module):
    def __init__(self, N, num_classes=19, lam=3.0, beta=0.7, device='cuda'):
        super().__init__()
        self.N = N
        self.num_classes = num_classes
        self.lam = lam
        self.beta = beta
        # register as buffer so it tracks device and is in state_dict
        self.register_buffer('target', torch.zeros(N, num_classes, dtype=torch.float32))

    def forward(self, logits, labels, index, weight=None):
        # predictions for teacher update must be detached
        y_pred_det = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4).detach()

        if weight is not None:
            # ensure mask is same shape, device, dtype
            m = (weight > 0).to(logits.dtype)
            old_t = self.target[index]
            self.target[index] = self.beta * old_t + (1 - self.beta) * (m * y_pred_det + (1 - m) * old_t)
        else:
            self.target[index] = self.beta * self.target[index] + (1 - self.beta) * y_pred_det

        # unreduced BCE
        bce_elts = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

        if weight is None:
            ce_loss = bce_elts.mean()
        else:
            wsum = weight.sum()
            if wsum.item() == 0:
                ce_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
            else:
                ce_loss = (bce_elts * weight).sum() / (wsum + 1e-8)

        # ELR regularizer uses non detached probabilities
        y_pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
        elr_elts = (1 - (self.target[index] * y_pred)).log()

        if weight is None:
            elr_reg = elr_elts.mean()
        else:
            wsum = weight.sum()
            if wsum.item() == 0:
                elr_reg = torch.zeros((), device=logits.device, dtype=logits.dtype)
            else:
                elr_reg = (elr_elts * weight).sum() / (wsum + 1e-8)

        return ce_loss + self.lam * elr_reg


class AsymmetricLossOptimized(torch.nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class RALossOptimized(torch.nn.Module):
    """
    Memory-aware and allocation-minimal variant of the original Ralloss.
    It mirrors the in-place operation pattern of AsymmetricLossOptimized.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        lamb: float = 1.5,
        epsilon_neg: float = 0.0,
        epsilon_pos: float = 1.0,
        epsilon_pos_pow: float = -2.5,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.lamb = lamb
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos = epsilon_pos
        self.epsilon_pos_pow = epsilon_pos_pow
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

        # Buffers reused every call to avoid fresh allocation
        self.targets = None
        self.anti_targets = None
        self.xs_pos = None
        self.xs_neg = None
        self.asymmetric_w = None
        self.loss = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: logits, shape (batch_size, num_labels)
        y: binary multi-label targets, same shape
        """
        self.targets = y
        self.anti_targets = 1.0 - y

        # Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Optional clipping for negatives
        if self.clip and self.clip > 0.0:
            self.xs_neg.add_(self.clip).clamp_(max=1.0)

        # Positive term with Taylor expansion
        one_minus_pos = 1.0 - self.xs_pos
        pos_log = torch.log(self.xs_pos.clamp(min=self.eps))
        pos_log.add_(self.epsilon_pos * one_minus_pos)
        pos_log.add_(0.5 * self.epsilon_pos_pow * one_minus_pos.pow(2))
        self.loss = self.targets * pos_log

        # Negative term with polynomial weighting
        neg_log = torch.log(self.xs_neg.clamp(min=self.eps))
        neg_log.add_(self.epsilon_neg * self.xs_neg)
        poly = (self.lamb - self.xs_pos) * self.xs_pos.pow(2) * (self.lamb - self.xs_neg)
        neg_log.mul_(poly)
        self.loss.add_(self.anti_targets * neg_log)

        # Asymmetric focal weighting
        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt = self.xs_pos * self.targets + self.xs_neg * self.anti_targets
            gamma = self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1.0 - pt, gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss.mul_(self.asymmetric_w)

        return -self.loss.sum()
