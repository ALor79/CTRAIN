import torch
import torch.nn as nn

from CTRAIN.bound.zonotope import HybridZonotope
from CTRAIN.bound.zono_relu import RELU_TRANSFORMERS
from CTRAIN.util import construct_c


def _propagate(dom, module, relu_fn):
    """
    Recursively propagate a HybridZonotope through a single nn.Module.

    Supported layer types:
        nn.Sequential, nn.Linear, nn.Conv2d, nn.ReLU,
        nn.Flatten, nn.BatchNorm1d, nn.BatchNorm2d,
        nn.AvgPool2d, nn.AdaptiveAvgPool2d (global only)

    Raises NotImplementedError for residual / skip-connection modules.
    """
    if isinstance(module, nn.Sequential):
        for layer in module:
            dom = _propagate(dom, layer, relu_fn)
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
        return relu_fn(dom)

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
        # Only support global average pooling (output 1x1)
        if output_size not in [1, (1, 1)]:
            raise NotImplementedError(
                f"AdaptiveAvgPool2d only supported with output_size=1 or (1,1), "
                f"got {output_size}"
            )
        h, w = dom.head.shape[-2], dom.head.shape[-1]
        return dom.apply_avg_pool2d(kernel_size=(h, w))

    if isinstance(module, nn.Identity):
        return dom

    # For modules with children (e.g. user-defined blocks with no skip connections)
    children = list(module.children())
    if children:
        for child in children:
            dom = _propagate(dom, child, relu_fn)
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
    if relu_transformer not in RELU_TRANSFORMERS:
        raise ValueError(
            f"Unknown relu_transformer '{relu_transformer}'. "
            f"Choose from {list(RELU_TRANSFORMERS.keys())}."
        )
    relu_fn = RELU_TRANSFORMERS[relu_transformer]

    # Extract plain nn.Module
    net = getattr(model, 'original_model', model)

    # Build the initial zonotope from the perturbation interval
    if ptb.x_L is None or ptb.x_U is None:
        raise ValueError(
            "ptb.x_L and ptb.x_U must be set. "
            "Create PerturbationLpNorm with explicit x_L= and x_U= arguments."
        )
    x_L = ptb.x_L.to(data.device)
    x_U = ptb.x_U.to(data.device)
    dom = HybridZonotope.from_perturbation(x_L, x_U, use_errors=use_errors)

    # Propagate through the network (use running BN stats)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        dom = _propagate(dom, net, relu_fn)
    if was_training:
        net.train()

    # Apply classification margin matrix C: [batch, n_classes-1, n_classes]
    c = construct_c(data, target, n_classes)
    dom = dom.apply_c_matrix(c)

    lb = dom.lb()
    ub = dom.ub()

    return lb, ub
