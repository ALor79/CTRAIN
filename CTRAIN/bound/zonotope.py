import torch
import torch.nn.functional as F
import torch.nn as nn

from diffai.crelu import crelu_boxy as _crelu_boxy
from diffai.crelu import crelu_switch as _crelu_switch
from diffai.crelu import crelu_smooth as _crelu_smooth

from CTRAIN.util import construct_c


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

    Unlike IBP/CROWN which delegate to auto_LiRPA's BoundedModule, hybrid zonotope
    propagation is not supported by auto_LiRPA natively. This class carries the
    abstract domain state between layers during the manual layer-by-layer walk
    performed by _propagate() and bound_zonotope(). It is an internal implementation
    detail and is not exported from bound/.
    """

    # Supported ReLU transformers — imported from diffai.crelu.
    # Each function operates at the tensor level (head, beta, errors) and is
    # wrapped here to accept/return HybridZonotope instances via apply_relu().
    RELU_TRANSFORMERS = ('boxy', 'switch', 'smooth')

    def __init__(self, head, beta, errors):
        self.head = head
        self.beta = beta
        self.errors = errors

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

    def __add__(self, other):
        """
        Add two HybridZonotope elements, as occurs at a residual skip connection.

        Heads are summed exactly. Betas are summed element-wise. Error generator
        sets are concatenated along dim 0 (the generator dimension), preserving
        all correlations from both branches.

        Adapted from diffai/ai.py HybridZonotope.__add__.
        """
        if isinstance(other, HybridZonotope):
            new_head = self.head + other.head

            if self.beta is not None and other.beta is not None:
                new_beta = self.beta + other.beta
            else:
                new_beta = self.beta if self.beta is not None else other.beta

            if self.errors is not None and other.errors is not None:
                new_errors = torch.cat([self.errors, other.errors], dim=0)
            else:
                new_errors = self.errors if self.errors is not None else other.errors

            return HybridZonotope(new_head, new_beta, new_errors)
        else:
            # Shift by a constant tensor — only the centre moves.
            return HybridZonotope(self.head + other, self.beta, self.errors)

    def apply_relu(self, relu_transformer):
        """
        Apply a ReLU crelu transformer by name.

        Delegates to the corresponding pure-tensor function from diffai.crelu,
        which implements the three variants from Mirman et al. (2018):
          - 'boxy'  : collapses crossing neurons to box [0, ub], discards errors
          - 'switch': keeps error terms for lean-positive neurons
          - 'smooth': soft interpolation between boxy and switch

        Args:
            relu_transformer (str): One of 'boxy', 'switch', 'smooth'.

        Returns:
            HybridZonotope
        """
        if relu_transformer == 'boxy':
            fn = _crelu_boxy
        elif relu_transformer == 'switch':
            fn = _crelu_switch
        elif relu_transformer == 'smooth':
            fn = _crelu_smooth
        else:
            raise ValueError(
                f"Unknown relu_transformer '{relu_transformer}'. "
                f"Choose from {self.RELU_TRANSFORMERS}."
            )
        new_head, new_beta, new_errors = fn(self.head, self.beta, self.errors)
        return HybridZonotope(new_head, new_beta, new_errors)


# auto_LiRPA does not support hybrid zonotopes natively, so bound propagation is
# implemented manually by walking the model layer by layer.
def _propagate(dom, module, relu_transformer):
    """
    Recursively propagate a HybridZonotope through a single nn.Module.

    Supported layer types:
        nn.Sequential, nn.Linear, nn.Conv2d, nn.ReLU,
        nn.Flatten, nn.BatchNorm1d, nn.BatchNorm2d,
        nn.AvgPool2d, nn.AdaptiveAvgPool2d (global only)

    Sequential-only by design. Unlike IBP and CROWN-IBP which use auto_LiRPA's
    BoundedModule to trace the full computational graph, this function walks
    layers manually and has no branch-merging logic. HybridZonotope.__add__
    implements the merge operator for skip connections (adapted from diffai/ai.py),
    but wiring it into graph traversal is left for future work. In practice,
    CTRAIN's zonotope bounds target sequential architectures (e.g. CNN7_Shi);
    residual architectures should use IBP or CROWN-IBP via auto_LiRPA.
    """
    if isinstance(module, nn.Sequential):
        for layer in module:
            dom = _propagate(dom, layer, relu_transformer)
        return dom

    if isinstance(module, nn.Linear):
        return dom.apply_linear(module.weight, module.bias)

    if isinstance(module, nn.Conv2d):
        return dom.apply_conv2d(
            module.weight, module.bias,
            module.stride, module.padding,
            module.dilation, module.groups,
        )

    if isinstance(module, (nn.ReLU, nn.ReLU6)):
        return dom.apply_relu(relu_transformer)

    if isinstance(module, nn.Flatten):
        return dom.apply_flatten(module.start_dim)

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return dom.apply_batch_norm(module)

    if isinstance(module, nn.AvgPool2d):
        return dom.apply_avg_pool2d(
            module.kernel_size, module.stride, module.padding
        )

    if isinstance(module, nn.AdaptiveAvgPool2d):
        output_size = module.output_size
        if output_size not in [1, (1, 1)]:
            raise NotImplementedError(
                f"AdaptiveAvgPool2d only supported with output_size=1 or (1,1), "
                f"got {output_size}"
            )
        h, w = dom.head.shape[-2], dom.head.shape[-1]
        return dom.apply_avg_pool2d(kernel_size=(h, w))

    if isinstance(module, nn.Identity):
        return dom

    children = list(module.children())
    if children:
        for child in children:
            dom = _propagate(dom, child, relu_transformer)
        return dom

    raise NotImplementedError(
        f"Unsupported layer type for zonotope propagation: {type(module).__name__}. "
        f"Residual/skip-connection architectures are not supported."
    )


def bound_zonotope(
    model,
    ptb,
    data,
    target,
    n_classes=10,
    relu_transformer='boxy',
    use_errors=False,
):
    """
    Compute certified output bounds using zonotope abstract interpretation.

    This is a drop-in replacement for bound_ibp that provides tighter bounds
    when use_errors=True by tracking correlations across neurons via explicit
    zonotope error terms.

    Args:
        model:            CTRAINWrapper or nn.Module. If the object has an
                          `original_model` attribute it will be used; otherwise
                          the model itself is used directly.
        ptb:              auto_LiRPA PerturbationLpNorm. Must have x_L and x_U
                          set (i.e. created with explicit lower/upper bounds).
        data:             Input tensor [batch, *spatial].
        target:           Class labels [batch].
        n_classes:        Number of output classes.
        relu_transformer: One of 'boxy', 'switch', 'smooth'.
        use_errors:       If True, initialise with explicit zonotope error terms
                          (tighter bounds, but memory scales as
                          O(n_input * batch * n_neurons) — use cautiously for
                          large inputs / CNNs).

    Returns:
        (lb, ub): Tensors of shape [batch, n_classes-1] representing lower and
                  upper bounds on class margins (correct - other).

    Memory note:
        With use_errors=False (default), propagation is equivalent to IBP.
        With use_errors=True, error terms are exact through linear/conv layers
        and are partially discarded at ReLU crossings depending on the chosen
        relu_transformer.
    """
    if relu_transformer not in HybridZonotope.RELU_TRANSFORMERS:
        raise ValueError(
            f"Unknown relu_transformer '{relu_transformer}'. "
            f"Choose from {HybridZonotope.RELU_TRANSFORMERS}."
        )

    net = getattr(model, 'original_model', model)

    if ptb.x_L is None or ptb.x_U is None:
        raise ValueError(
            "ptb.x_L and ptb.x_U must be set. "
            "Create PerturbationLpNorm with explicit x_L= and x_U= arguments."
        )
    x_L = ptb.x_L.to(data.device)
    x_U = ptb.x_U.to(data.device)
    dom = HybridZonotope.from_perturbation(x_L, x_U, use_errors=use_errors)

    dom = _propagate(dom, net, relu_transformer)

    c = construct_c(data, target, n_classes)
    dom = dom.apply_c_matrix(c)

    return dom.lb(), dom.ub()
