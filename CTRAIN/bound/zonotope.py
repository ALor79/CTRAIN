import torch
import torch.nn.functional as F
import torch.nn as nn


class HybridZonotope:
    """
    Hybrid zonotope abstract domain element for certified bound propagation.

    Represents the set:
        { head + errors^T z + beta * w  |  ||z||_inf <= 1, ||w||_inf <= 1 }

    where:
      - head  : center,              shape [batch, *spatial]
      - beta  : box radius,          shape [batch, *spatial]  or None
      - errors: zonotope generators, shape [n_errors, batch, *spatial]  or None

    Using beta alone (errors=None) is equivalent to IBP.
    Using errors introduces correlation tracking, giving tighter bounds at the
    cost of memory: O(n_input * batch * n_neurons) per layer.
    """

    def __init__(self, head, beta, errors):
        self.head = head
        self.beta = beta
        self.errors = errors

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_perturbation(cls, x_L, x_U, use_errors=False):
        """
        Build a HybridZonotope from a concrete L-inf interval [x_L, x_U].

        Args:
            x_L:        element-wise lower bound, shape [batch, *spatial]
            x_U:        element-wise upper bound, same shape
            use_errors: if True, expand the initial box into explicit error terms
                        (one per input element — memory-intensive for large inputs)

        Returns:
            HybridZonotope
        """
        head = (x_L + x_U) / 2.0
        beta = (x_U - x_L) / 2.0

        if not use_errors:
            return cls(head=head, beta=beta, errors=None)

        # Convert the input box to explicit error terms.
        # errors[i, b, :] = beta[b, i] * e_i  (i-th standard basis vector)
        batch = head.shape[0]
        n_elem = beta[0].numel()
        err = torch.diag(beta[0].flatten())                # [n_elem, n_elem]
        err = err.unsqueeze(1).expand(-1, batch, -1)       # [n_elem, batch, n_elem]
        err = err.contiguous().view(n_elem, batch, *head.shape[1:])
        return cls(head=head, beta=None, errors=err)

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def lb(self):
        """Element-wise lower bound."""
        result = self.head.clone()
        if self.beta is not None:
            result = result - self.beta
        if self.errors is not None:
            result = result - self.errors.abs().sum(0)
        return result

    def ub(self):
        """Element-wise upper bound."""
        result = self.head.clone()
        if self.beta is not None:
            result = result + self.beta
        if self.errors is not None:
            result = result + self.errors.abs().sum(0)
        return result

    # ------------------------------------------------------------------
    # Linear transformers (exact for affine operations)
    # ------------------------------------------------------------------

    def apply_linear(self, weight, bias):
        """
        y = x W^T + b

        head   -> W head + b       (exact)
        beta   -> |W| beta         (interval arithmetic)
        errors -> W errors[i]      (exact, preserves correlations)
        """
        new_head = F.linear(self.head, weight, bias)
        new_beta = F.linear(self.beta, weight.abs(), None) if self.beta is not None else None

        if self.errors is not None:
            n_err, batch = self.errors.shape[0], self.errors.shape[1]
            err_2d = self.errors.view(n_err * batch, -1)
            new_errors = F.linear(err_2d, weight, None).view(n_err, batch, -1)
        else:
            new_errors = None

        return HybridZonotope(new_head, new_beta, new_errors)

    def apply_conv2d(self, weight, bias, stride, padding, dilation, groups):
        """
        2D convolution.

        head   -> conv(head, W, b)         (exact)
        beta   -> conv(beta, |W|, None)    (interval arithmetic)
        errors -> conv(errors[i], W, None) (exact)
        """
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)
        new_beta = (
            F.conv2d(self.beta, weight.abs(), None, stride, padding, dilation, groups)
            if self.beta is not None else None
        )

        if self.errors is not None:
            n_err, batch = self.errors.shape[0], self.errors.shape[1]
            err_flat = self.errors.view(n_err * batch, *self.errors.shape[2:])
            new_err_flat = F.conv2d(err_flat, weight, None, stride, padding, dilation, groups)
            new_errors = new_err_flat.view(n_err, batch, *new_err_flat.shape[1:])
        else:
            new_errors = None

        return HybridZonotope(new_head, new_beta, new_errors)

    def apply_flatten(self, start_dim=1):
        """Flatten spatial dimensions."""
        new_head = self.head.flatten(start_dim)
        new_beta = self.beta.flatten(start_dim) if self.beta is not None else None
        # errors is [n_err, batch, *spatial]; spatial starts at dim 2
        new_errors = self.errors.flatten(start_dim + 1) if self.errors is not None else None
        return HybridZonotope(new_head, new_beta, new_errors)

    def apply_batch_norm(self, module):
        """
        BatchNorm as an affine map using running statistics (eval semantics):
            scale = gamma / sqrt(var + eps)
            shift = bias - mean * scale
            y     = scale * x + shift

        head   -> scale * head + shift  (exact)
        beta   -> |scale| * beta        (exact for element-wise scaling)
        errors -> scale * errors[i]     (exact)
        """
        if module.running_mean is None:
            raise ValueError(
                "BatchNorm running statistics are None — run a forward pass first."
            )

        gamma = module.weight if module.weight is not None else torch.ones_like(module.running_mean)
        bias_bn = module.bias if module.bias is not None else torch.zeros_like(module.running_mean)

        scale = gamma / torch.sqrt(module.running_var + module.eps)
        shift = bias_bn - module.running_mean * scale

        view_shape = (1, -1, 1, 1) if self.head.dim() == 4 else (1, -1)
        scale = scale.view(view_shape)
        shift = shift.view(view_shape)

        new_head = self.head * scale + shift
        new_beta = self.beta * scale.abs() if self.beta is not None else None
        new_errors = self.errors * scale.unsqueeze(0) if self.errors is not None else None

        return HybridZonotope(new_head, new_beta, new_errors)

    def apply_avg_pool2d(self, kernel_size, stride=None, padding=0):
        """Average pooling is linear — exact for zonotopes."""
        new_head = F.avg_pool2d(self.head, kernel_size, stride, padding)
        new_beta = (
            F.avg_pool2d(self.beta, kernel_size, stride, padding)
            if self.beta is not None else None
        )

        if self.errors is not None:
            n_err, batch = self.errors.shape[0], self.errors.shape[1]
            err_flat = self.errors.view(n_err * batch, *self.errors.shape[2:])
            new_err_flat = F.avg_pool2d(err_flat, kernel_size, stride, padding)
            new_errors = new_err_flat.view(n_err, batch, *new_err_flat.shape[1:])
        else:
            new_errors = None

        return HybridZonotope(new_head, new_beta, new_errors)

    def apply_c_matrix(self, c):
        """
        Apply the classification margin matrix C.

        C @ logits gives (correct_class_score - other_class_score) per class pair.

        Args:
            c: [batch, n_classes-1, n_classes]

        Returns:
            HybridZonotope with head shape [batch, n_classes-1]
        """
        new_head = torch.bmm(c, self.head.unsqueeze(-1)).squeeze(-1)
        new_beta = (
            torch.bmm(c.abs(), self.beta.unsqueeze(-1)).squeeze(-1)
            if self.beta is not None else None
        )

        if self.errors is not None:
            # errors: [n_err, batch, n_classes] -> [n_err, batch, n_classes-1]
            new_errors = torch.einsum('bkc,ebc->ebk', c, self.errors)
        else:
            new_errors = None

        return HybridZonotope(new_head, new_beta, new_errors)
